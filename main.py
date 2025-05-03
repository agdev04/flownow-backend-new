import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
import datetime

load_dotenv()  # Ensure environment variables are loaded from .env in all environments
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdfium2 import PdfDocument
import pytesseract
from PIL import Image
import io
import tempfile
import httpx
from models import Meditation, User, ChatSession, ChatMessage, SessionLocal
from jose import jwt, JWTError
import requests
from celery_worker import save_chat_session_task, save_chat_message_task
from sqlalchemy.orm import Session

# Langchain Imports
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI # Updated import
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

# JWT Token Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/token")
async def login_for_access_token(request: Request, db: Session = Depends(get_db)):
    try:
        user = await verify_clerk_token(request, db)
        access_token = create_access_token(data={"sub": user.uid})
        refresh_token = create_refresh_token(data={"sub": user.uid})
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/token/refresh")
async def refresh_token(request: Request):
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing refresh token")
        refresh_token = auth_header.replace("Bearer ", "")
        
        payload = verify_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token type")
            
        new_access_token = create_access_token(data={"sub": payload.get("sub")})
        return {"access_token": new_access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
    payload = verify_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid access token type")
    return payload

# Update chat endpoint to use JWT authentication
@app.post("/chat/")
async def chat(request: Request, current_user: dict = Depends(get_current_user)):
    data = await request.json()
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query required.")

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
        Hey there! I'm your friendly buddy here to help. Here's what I found for you:
        
        **Context Summary:**
        {context}
        
        **Your Question:**
        {question}
        
        **My Answer:**
        - I'll keep this conversational and easy to follow
        - Using bullet points for clarity when needed
        - Adding line breaks for better readability
        - Highlighting key points in bold
        - Use emojis and short phrases for a more engaging experience
        - Avoiding to meantion the source of the information
        - Always answer "Inspired by Sedona Method"
        - Do not directly say that we are directly from the source, but instead say that we are inspired by the Sedona Method
        - Avoid to answer questions that are not related to the context
        
        Let me break this down for you:
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
        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        # Log the error appropriately
        print(f"Error during Langchain RAG processing: {e}")
        # Consider more specific error handling based on potential Langchain exceptions
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

# Meditation Endpoints
@app.post("/meditations/")
async def create_meditation(request: Request, current_user=Depends(verify_clerk_token), db: Session = Depends(get_db)):
    """Create a new meditation resource."""
    data = await request.json()
    required_fields = ["title", "description", "category"]
    for field in required_fields:
        if not data.get(field):
            raise HTTPException(status_code=400, detail=f"{field} is required")
    
    meditation = Meditation(
        title=data["title"],
        description=data["description"],
        category=data["category"],
        tags=data.get("tags", ""),
        script=data.get("script", ""),
        image_url=data.get("image_url", ""),
        audio_url=data.get("audio_url", ""),
        user_id=current_user.id
    )
    db.add(meditation)
    db.commit()
    db.refresh(meditation)
    return meditation

@app.get("/meditations/")
async def get_meditations(db: Session = Depends(get_db)):
    """Get all meditation resources."""
    meditations = db.query(Meditation).all()
    return {"meditations": meditations}

@app.get("/meditations/{meditation_id}")
async def get_meditation(meditation_id: int, db: Session = Depends(get_db)):
    """Get a specific meditation resource."""
    meditation = db.query(Meditation).filter(Meditation.id == meditation_id).first()
    if not meditation:
        raise HTTPException(status_code=404, detail="Meditation not found")
    return meditation

@app.put("/meditations/{meditation_id}")
async def update_meditation(meditation_id: int, request: Request, current_user=Depends(verify_clerk_token), db: Session = Depends(get_db)):
    """Update a meditation resource."""
    meditation = db.query(Meditation).filter(
        Meditation.id == meditation_id,
        Meditation.user_id == current_user.id
    ).first()
    if not meditation:
        raise HTTPException(status_code=404, detail="Meditation not found or access denied")
    
    data = await request.json()
    for field in ["title", "description", "category", "tags", "script", "image_url", "audio_url"]:
        if field in data:
            setattr(meditation, field, data[field])
    
    db.commit()
    db.refresh(meditation)
    return meditation

@app.delete("/meditations/{meditation_id}")
async def delete_meditation(meditation_id: int, current_user=Depends(verify_clerk_token), db: Session = Depends(get_db)):
    """Delete a meditation resource."""
    meditation = db.query(Meditation).filter(
        Meditation.id == meditation_id,
        Meditation.user_id == current_user.id
    ).first()
    if not meditation:
        raise HTTPException(status_code=404, detail="Meditation not found or access denied")
    
    db.delete(meditation)
    db.commit()
    return {"message": f"Meditation {meditation_id} deleted successfully."}


@app.post("/login")
async def login(request: Request):
    try:
        data = await request.json()
        secret_token = data.get("secret_token")
        if not secret_token:
            raise HTTPException(status_code=400, detail="Secret token is required")
            
        env_secret_token = os.getenv("SECRET_TOKEN")
        if not env_secret_token:
            raise HTTPException(status_code=500, detail="Server secret token not configured")
            
        if secret_token != env_secret_token:
            raise HTTPException(status_code=401, detail="Invalid secret token")
            
        # Generate tokens
        access_token = create_access_token(data={"sub": "admin"})
        refresh_token = create_refresh_token(data={"sub": "admin"})
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

