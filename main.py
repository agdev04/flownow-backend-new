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

# Langchain Imports
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_community.chat_models import ChatOpenAI # Assuming OpenRouter compatibility or replace later
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # Import Document for creating Langchain documents

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

        # --- Begin Langchain Integration for PDF Upload ---
        cohere_api_key = os.getenv("COHERE_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY") # Ensure PINECONE_API_KEY is set in .env
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "sed-met")

        if not all([cohere_api_key, pinecone_api_key, pinecone_index_name]):
            raise HTTPException(status_code=500, detail="Missing required API keys or index name for PDF processing.")

        # Initialize Langchain components
        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)

        # Create Langchain Document objects
        documents = []
        for idx, text in enumerate(texts):
            metadata = {"filename": pdf_file.filename, "page": idx + 1}
            documents.append(Document(page_content=text, metadata=metadata))

        # Upsert using Langchain's PineconeVectorStore
        # This handles index creation check implicitly if using `from_documents`
        # Note: Ensure PINECONE_API_KEY is available in the environment for PineconeVectorStore
        vectorstore = PineconeVectorStore.from_documents(
            documents,
            index_name=pinecone_index_name,
            embedding=embeddings
            # pinecone_api_key=pinecone_api_key # Usually picked from env
        )
        # --- End Langchain Integration for PDF Upload ---

        # --- Old Pinecone & Cohere Integration Removed ---
        # import cohere
        # from pinecone.grpc import PineconeGRPC as Pinecone
        # from pinecone import ServerlessSpec
        # from dotenv import load_dotenv
        # load_dotenv()
        # cohere_api_key = os.getenv("COHERE_API_KEY")
        # pinecone_api_key = os.getenv("PINECONE_API_KEY")
        # pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "sed-met")
        # openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        # if not all([cohere_api_key, pinecone_api_key, openrouter_api_key]):
        #     raise HTTPException(status_code=500, detail="Missing required API keys.")
        # co = cohere.Client(cohere_api_key)
        # pc = Pinecone(api_key=pinecone_api_key)
        # # Ensure index exists using new SDK style
        # if not pc.has_index(pinecone_index_name):
        #     pc.create_index(
        #         name=pinecone_index_name,
        #         vector_type="dense",
        #         dimension=1024,
        #         metric="cosine",
        #         spec=ServerlessSpec(
        #             cloud="aws",
        #             region="us-east-1"
        #         ),
        #         deletion_protection="disabled",
        #         tags={"environment": "development"}
        #     )
        # index = pc.Index(pinecone_index_name)
        # embeddings_result = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document").embeddings
        # to_upsert = []
        # for idx, (embedding, text) in enumerate(zip(embeddings_result, texts)):
        #     to_upsert.append({
        #         "id": f"{pdf_file.filename}-page-{idx+1}",
        #         "values": embedding,
        #         "metadata": {"filename": pdf_file.filename, "page": idx+1, "text": text}
        #     })
        # # Upsert all pages
        # index.upsert(vectors=to_upsert)
        # --- End Old Pinecone & Cohere Integration ---

        return JSONResponse(content={"message": f"Successfully processed and indexed {len(documents)} pages from {pdf_file.filename}.", "pages": len(images)})
    except Exception as e:
        # Log the error appropriately
        print(f"Error during PDF processing: {e}")
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
        if not session: # Handle case where session_id is provided but doesn't exist or belong to user
             raise HTTPException(status_code=404, detail="Chat session not found or access denied.")
    else:
        session_dict = {"user_id": current_user.id}
        # Use .get() for potentially long-running task, consider timeout
        try:
            session_id = save_chat_session_task.delay(session_dict).get(timeout=10) 
            session = ChatSession(id=session_id, user_id=current_user.id) # Create a temporary object for immediate use
        except Exception as e:
            # Log the error appropriately
            print(f"Error creating chat session via Celery: {e}")
            raise HTTPException(status_code=500, detail="Failed to create chat session.")

    # Save user message asynchronously
    user_msg_dict = {"session_id": session.id, "user_id": current_user.id, "role": "user", "message": query}
    save_chat_message_task.delay(user_msg_dict)

    # --- Langchain RAG Implementation ---
    try:
        # Load API keys
        cohere_api_key = os.getenv("COHERE_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "sed-met")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        llm_model = os.getenv("LLM_MODEL", "gryphe/mythomax-l2-13b")
        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if not all([cohere_api_key, pinecone_api_key, openrouter_api_key, pinecone_index_name]):
            raise HTTPException(status_code=500, detail="Missing required API keys or index name.")

        # 1. Initialize Langchain components
        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=pinecone_index_name,
            embedding=embeddings,
            # pinecone_api_key=pinecone_api_key # Often handled by environment variables PINECONE_API_KEY
        )
        retriever = vectorstore.as_retriever()

        # Configure ChatOpenAI for OpenRouter
        llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=openrouter_api_key,
            openai_api_base=openrouter_base_url,
            # temperature=0.7, # Optional: Adjust temperature
            # max_tokens=1024, # Optional: Adjust max tokens
        )

        # 2. Define Prompt Template
        template = """
        Hey there! I'm your friendly buddy here to help. Answer the user's question based *only* on the context I've found for you. Keep it conversational and friendly, like we're just chatting! Don't mention where the info came from, just give the answer.
        Context:
        {context}

        User Question: {question}

        Your Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 3. Define RAG Chain using LCEL
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 4. Invoke Chain
        ai_response = rag_chain.invoke(query)

        # Save AI message asynchronously
        ai_msg_dict = {"session_id": session.id, "user_id": current_user.id, "role": "assistant", "message": ai_response}
        save_chat_message_task.delay(ai_msg_dict)

        return JSONResponse(content={"response": ai_response, "session_id": session.id})

    except Exception as e:
        # Log the error appropriately
        print(f"Error during Langchain RAG processing: {e}")
        # Consider more specific error handling based on potential Langchain exceptions
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

# --- Old /chat/ logic removed, replaced by Langchain implementation above ---

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