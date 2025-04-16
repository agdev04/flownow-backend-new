import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends

load_dotenv()  # Ensure environment variables are loaded from .env in all environments
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdfium2 import PdfDocument
import pytesseract
from PIL import Image
import io
import tempfile
import httpx
from models import User, ChatSession, ChatMessage, SessionLocal
from jose import jwt, JWTError
import requests
from celery_worker import save_chat_session_task, save_chat_message_task
from sqlalchemy.orm import Session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Clerk JWKS Setup ---
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")  # Set this to your actual JWKS URL

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_jwks():
    response = requests.get(CLERK_JWKS_URL)
    response.raise_for_status()
    return response.json()["keys"]


def get_public_key(token):
    unverified = jwt.get_unverified_header(token)
    kid = unverified.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Malformed token header, missing 'kid'.")
    keys = get_jwks()
    for key in keys:
        if key["kid"] == kid:
            return key
    raise HTTPException(status_code=401, detail="Public key not found for kid.")


def verify_clerk_token(request: Request, db = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = auth_header.replace("Bearer ", "")
    try:
        public_jwk = get_public_key(token)
        payload = jwt.decode(
            token,
            public_jwk,
            algorithms=public_jwk["alg"] if "alg" in public_jwk else ["RS256", "ES256"],
            options={"verify_aud": False}  # add proper audience verification for production
        )
        print("Decoded payload:", payload)
        uid = payload.get("sub")
        email = payload.get("email")

        if not uid:
            raise HTTPException(status_code=401, detail="Token did not include subject (sub).")
        user = db.query(User).filter_by(uid=uid).first()
        if not user:
            user = User(uid=uid, email=email)
            db.add(user)
            db.commit()
            print("New user created:", user)
        return user
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Clerk token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification error: {str(e)}")

@app.get("/auth-test")
def auth_test_route(current_user=Depends(verify_clerk_token)):
    return {"message": f"Authenticated! UID: {current_user.uid}, Email: {current_user.email}"}

@app.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        contents = await pdf_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(contents)
            temp_pdf_path = temp_pdf.name
        doc = PdfDocument(temp_pdf_path)
        images = []
        texts = []
        for page_number, page in enumerate(doc):
            page_image = page.render(scale=2).to_pil()
            img_bytes = io.BytesIO()
            page_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            images.append(img_bytes.getvalue())
            text = pytesseract.image_to_string(page_image)
            texts.append(text)
        os.unlink(temp_pdf_path)
        # --- Begin Pinecone & Cohere Integration ---
        import cohere
        from pinecone.grpc import PineconeGRPC as Pinecone
        from pinecone import ServerlessSpec
        from dotenv import load_dotenv
        load_dotenv()
        cohere_api_key = os.getenv("COHERE_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "sed-met")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not all([cohere_api_key, pinecone_api_key, openrouter_api_key]):
            raise HTTPException(status_code=500, detail="Missing required API keys.")
        co = cohere.Client(cohere_api_key)
        pc = Pinecone(api_key=pinecone_api_key)
        # Ensure index exists using new SDK style
        if not pc.has_index(pinecone_index_name):
            pc.create_index(
                name=pinecone_index_name,
                vector_type="dense",
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="disabled",
                tags={"environment": "development"}
            )
        index = pc.Index(pinecone_index_name)
        embeddings = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document").embeddings
        to_upsert = []
        for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
            to_upsert.append({
                "id": f"{pdf_file.filename}-page-{idx+1}",
                "values": embedding,
                "metadata": {"filename": pdf_file.filename, "page": idx+1, "text": text}
            })
        # Upsert all pages
        index.upsert(vectors=to_upsert)
        # --- End Pinecone & Cohere Integration ---
        return JSONResponse(content={"pages": len(images), "texts": texts})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/chat/")
async def chat(request: Request, current_user=Depends(verify_clerk_token), db=Depends(get_db)):
    data = await request.json()
    query = data.get("query")
    session_id = data.get("session_id")
    if not query:
        raise HTTPException(status_code=400, detail="Query required.")
    # --- Find or create a chat session for this user ---
    if session_id:
        session = db.query(ChatSession).filter_by(id=session_id, user_id=current_user.id).first()
    else:
        session_dict = {"user_id": current_user.id}
        session_id = save_chat_session_task.delay(session_dict).get(timeout=10)
        session = ChatSession(id=session_id, user_id=current_user.id)
    # Save user message asynchronously
    user_msg_dict = {"session_id": session.id, "user_id": current_user.id, "role": "user", "message": query}
    save_chat_message_task.delay(user_msg_dict)
    # --- AI processing below this ---
    import cohere
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec
    from dotenv import load_dotenv
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "sed-met")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "gryphe/mythomax-l2-13b")
    if not all([cohere_api_key, pinecone_api_key, openrouter_api_key]):
        raise HTTPException(status_code=500, detail="Missing required API keys.")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }
            classifier_prompt = f"""
                Classify the following user input strictly as \"yes\" (general knowledge) or \"no\" (not general knowledge).  
                General knowledge = factual questions (science, history, definitions, etc.).  
                Not general knowledge = personal advice (stress, emotions, success, life tips).  

                Examples:  
                1. \"How does photosynthesis work?\" → yes  
                2. \"How to be happier?\" → no  
                3. \"What is the capital of France?\" → yes  
                4. \"How to deal with anxiety?\" → no  

                Return only \"yes\" or \"no\" for this input:  
                \"{query}\"
                """
            classification_payload = {
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": "You are a topic classifier.ONLY reply Yes or No."},
                    {"role": "user", "content": classifier_prompt}
                ]
            }
            classification_resp = await client.post(openrouter_url, headers=headers, json=classification_payload)
            classification_resp.raise_for_status()
            raw = classification_resp.json()
            ai_label = raw["choices"][0]["message"]["content"].strip().lower()
            print("Classifier AI label:", ai_label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter topic classification error: {str(e)}")
    # If label is not strictly 'yes', halt here
    if not ("no" in ai_label and "yes" not in ai_label):
        ai_response = "Thank you for your question! I'm here to help with life, emotions, or meditation topics. If you have questions related to those topics, feel free to ask!"
        assistant_msg_dict = {"session_id": session.id, "user_id": current_user.id, "role": "assistant", "message": ai_response, "fault": 1}
        save_chat_message_task.delay(assistant_msg_dict)
        return {"answer": ai_response, "sources": [], "session_id": session.id}
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        if not pc.has_index(pinecone_index_name):
            raise HTTPException(status_code=500, detail="Pinecone index does not exist.")
        co = cohere.Client(cohere_api_key)
        index = pc.Index(pinecone_index_name)
        query_embedding = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query").embeddings[0]
        search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        contexts = []
        for match in search_results.matches:
            md = getattr(match, "metadata", match.get("metadata", {}))
            text = md.get("text")
            if text:
                contexts.append(text)
        prompt = (
            "You are a super friendly assistant—like a wise, supportive friend. Your job is to help users with life, emotions, or meditation questions. Always use warm, buddy-like, conversational, and encouraging language, making users feel understood, comfortable, and uplifted. Prefer plain, positive, and caring responses! Your main reference is the relevant uploaded documents provided below as context, but you can also use your own reasoning, knowledge, and positive vibes to make answers as helpful, comforting, and reassuring as possible. Here are relevant context snippets (if any):\n---\n"
            + "\n---\n".join(contexts) + f"\n\nQuestion: {query}\nAnswer:"
        )
        if not contexts:
            ai_response = "I couldn’t find any matching information in your uploaded documents to answer this question. But hey, don't worry—I'm always here for you! If you'd like to chat about life, feelings, or meditation, just let me know and I'll do my best to cheer you on and support you!"
            assistant_msg_dict = {"session_id": session.id, "user_id": current_user.id, "role": "assistant", "message": ai_response, "fault": 0}
            save_chat_message_task.delay(assistant_msg_dict)
            return {"answer": ai_response, "sources": [], "session_id": session.id}
        async with httpx.AsyncClient(timeout=60.0) as client:
            openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gryphe/mythomax-l2-13b",
                "messages": [
                    {"role": "system", "content": "You are an extremely friendly, supportive, uplifting buddy. When answering questions about life, emotions, or meditation, always use warm, conversational, and positive language, and prioritize making the user feel heard and encouraged! IMPORTANT: Users are reading your advice as a chat, so they cannot close their eyes while following your tips. Please avoid suggesting the user close their eyes for meditation or exercises, and instead offer instructions that can be followed while reading with eyes open. Use the uploaded documents as your main reference, but feel free to add encouraging words from your friendly knowledge. ALWAYS format your entire response as markdown code for improved layout and styling."},
                    {"role": "user", "content": prompt}
                ]
            }
            resp = await client.post(openrouter_url, headers=headers, json=payload)
            resp.raise_for_status()
            resp_json = resp.json()
            answer = resp_json["choices"][0]["message"]["content"]
            # Save assistant's message
            assistant_msg = ChatMessage(session_id=session.id, user_id=current_user.id, role="assistant", message=answer, fault=0)
            db.add(assistant_msg)
            db.commit()
            return {"answer": answer, "sources": contexts, "session_id": session.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter or retrieval error: {str(e)}")

@app.get("/chat_sessions/", response_model=None)
async def get_chat_sessions(current_user=Depends(verify_clerk_token), db: Session = Depends(get_db)):
    """Returns all chat sessions and their messages for the current authenticated user."""
    sessions = db.query(ChatSession).filter_by(user_id=current_user.id).all()
    session_data = []
    for session in sessions:
        messages = db.query(ChatMessage).filter_by(session_id=session.id).order_by(ChatMessage.created_at.asc()).all()
        session_data.append({
            "session_id": session.id,
            "created_at": session.created_at,
            "title": session.title,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "message": msg.message,
                    "created_at": msg.created_at,
                    "fault": msg.fault
                } for msg in messages
            ]
        })
    return {"sessions": session_data}

@app.put("/chat_sessions/{session_id}/title")
async def update_session_title(session_id: int, data: dict, current_user=Depends(verify_clerk_token), db: Session = Depends(get_db)):
    """Allows the current authenticated user to update the title of their chat session."""
    session = db.query(ChatSession).filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    new_title = data.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required.")
    session.title = new_title
    db.commit()
    db.refresh(session)
    return {"session_id": session.id, "title": session.title}

@app.delete("/chat_sessions/{session_id}")
async def delete_session(session_id: int, current_user=Depends(verify_clerk_token), db: Session = Depends(get_db)):
    """Allows the current authenticated user to delete their chat session."""
    session = db.query(ChatSession).filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    db.delete(session)
    db.commit()
    return {"message": f"Session {session_id} deleted successfully."}