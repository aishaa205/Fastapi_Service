import sys
sys.path.append("..") 
from sentence_transformers import SentenceTransformer , util
from fastapi import FastAPI , HTTPException ,Query, UploadFile, File, Form , Header, Depends
from database import jobs_collection ,users_collection, rag_collection, rag_names_collection, get_user_table, async_session,test_collection
from pydantic import BaseModel,validator,field_validator, Field
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import re
import fitz  # PyMuPDf
from io import BytesIO
# from googleapiclient.discovery import build
from loguru import logger
from functools import lru_cache
import requests  # To download Cloudinary files
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
# from bson.errors import InvalidId
security = HTTPBearer()
from fastapi import Request, BackgroundTasks
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from datetime import datetime
import time
import openai
from dotenv import load_dotenv
from typing import Any
from PIL import Image
from io import BytesIO
# import face_recognition
import requests
# import tempfile
# import traceback
# from deepface import DeepFace
# import easyocr
from contextlib import asynccontextmanager
from app.utils import save_temp_file, download_video_from_url
from transformers import ViTForImageClassification, ViTImageProcessor


####################################
import os
import cv2
import numpy as np
# import speech_recognition as sr
# from deepface import DeepFace
from sentence_transformers import SentenceTransformer, util
import torch
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# from PIL import Image
from fastapi import FastAPI, APIRouter, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated ,List,Optional
import logging
import traceback
# from pydub import AudioSegment
# from pydub.silence import detect_silence
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
# import noisereduce as nr
# import httpx
# from reportlab.lib.pagesizes import letter
import whisper
# import ffmpeg
# import cv2
import mediapipe as mp
# from collections import Counter
import re
from typing import Dict,TypedDict
from pydantic import BaseModel
# from torchvision import transforms
import base64
import clip
from queue_consumer import consume_queue
from queue_producer import send_to_queue
import asyncio
import traceback
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
# model = SentenceTransformer('all-MiniLM-L6-v2')
# resnet_model = ResNet50(weights='imagenet')
router = APIRouter()

# Initialize CLIP model globally to avoid reloading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print ("device",DEVICE)
CLIP_MODEL, CLIP_PREPROCESS = None, None

def load_clip_model():
    """Load CLIP model once and cache it"""
    global CLIP_MODEL, CLIP_PREPROCESS
    if CLIP_MODEL is None:
        try:
            CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIP model: {str(e)}")
            raise


# # ImageNet classes for attire analysis
# imagenet_classes = [
#     'suit', 'tie', 'shirt', 'business suit', 'bow tie', 'jeans',
#     'sweatshirt', 'T-shirt', 'sweat pants', 'pajamas', 'lab coat'
# ]








load_dotenv()
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update
# from some_fraud_detection_lib import is_fake_id
# from some_id_front_detector import is_front_side
background_tasks = BackgroundTasks()




DJANGO_AUTH_URL = "http://localhost:8000/api/token/verify/"

security = HTTPBearer()
# app = FastAPI()
#load  ats model

# model = SentenceTransformer("all-MiniLM-L6-v2")

#load recommender model
# def load_model():
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     return SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")

# @app.on_event("startup")
# async def startup_event():
#     global model
#     model = load_model()
#     logger.info("ML Model Loaded Successfully!")
#     load_dotenv()

consumer_tasks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global model, ats_model
    # Load main recommendation model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")
    logger.info("Recommendation model loaded.")
    # Load ATS model
    # ats_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    logger.info("ATS model loaded.")
    for q in ("job_queue", "user_queue", "application_queue"):
        task = asyncio.create_task(consume_queue(q))
        consumer_tasks.append(task)
        logger.info(f"Started listening on {q}")
    yield
    # Shutdown logic
    logger.info("Shutting down FastAPI application.")
    logger.info("Shutting down consumers...")
    for task in consumer_tasks:
        task.cancel()
    await asyncio.gather(*consumer_tasks, return_exceptions=True)
    logger.info("All consumers shut down.")

app = FastAPI(lifespan=lifespan)



CLOUDINARY_CLOUD_NAME ="dkvyfbtdl"
def extract_text_from_pdf_cloud(public_id: str):
    """Downloads and extracts text from a Cloudinary-hosted PDF file using its public_id."""
    try:
        pdf_url = f"https://res.cloudinary.com/{CLOUDINARY_CLOUD_NAME}/raw/upload/{public_id}.pdf"
        print("Downloading PDF from:", pdf_url)

        response = requests.get(pdf_url)
        response.raise_for_status()  # Ensure the request was successful
        print("PDF downloaded successfully.")
       
        pdf_stream = BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text = "\n".join([page.get_text("text") for page in doc])
        # print ("text",text)
        doc.close()

        return text.strip() if text else "No text found in PDF."
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {e}")   




def get_embedding(text):
    with torch.no_grad():
        return model.encode(text, convert_to_tensor=True).tolist()
    
def format_cv_url(cv_url):
    if cv_url.startswith("http"):
        return cv_url.split("/")[-1].replace(".pdf", "")
    return cv_url

def update_user(embedding, user_id, cv_url):
    users_collection.update_one(
                    {"user_id": user_id},
                    {"$set": {"embedding": embedding, "cv_url": format_cv_url(cv_url)}},
                    upsert=True
                )
def check_user_cv(user_id, cv_url):
    user_data = users_collection.find_one({"user_id": user_id})
    if not user_data:
        embedding = get_embedding(extracted_text)
        # background_tasks.add_task(update_user, embedding, user_id, cv_url)
        update_user(embedding, user_id, cv_url)
        return embedding
    if user_data and user_data.get("cv_url") == cv_url:
        print("Using stored embedding (CV unchanged)")
        return user_data["embedding"]
    else:
        print("CV updated, generating new embedding")
        extracted_text = extract_text_from_pdf_cloud(format_cv_url(cv_url))
        print(f"Extracted text length: {len(extracted_text)}")
        embedding = get_embedding(extracted_text)
        # background_tasks.add_task(update_user, embedding, user_id, cv_url)
        update_user(embedding, user_id, cv_url)
        return embedding

@app.get('/user_embedding/')
async def get_user_embedding(user_id: int, cv_url: str):
    embedding = check_user_cv(user_id, cv_url)
    return {"embedding": embedding}

@app.get("/recom/")
async def recommend_jobs(user_id: int, cv_url: str, page: int = 1, page_size: int = 10):
    try:
        
        if not cv_url:
            raise HTTPException(status_code=400, detail="CV URL is required")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")

        cv_url = format_cv_url(cv_url)
        if page < 1:
            raise HTTPException(status_code=400, detail="Page number must be 1 or higher")

        page_count = jobs_collection.count_documents({"status": "1"})
        
        if page_count == 0:
            raise HTTPException(status_code=404, detail="No jobs found")
        if page*page_size > 100:
            raise HTTPException(status_code=400, detail="Max recommendations is 100")
        if page > page_count/page_size and page > 1:
            raise HTTPException(status_code=400, detail="Page number exceeds available pages")
        
        query_vector = check_user_cv(user_id, cv_url)

        skip = (page - 1) * page_size
        pipeline = [
            {
                "$match": {
                    "status": "1",
                }
            },
            {
                "$vectorSearch": {
                    "index": "default",
                    # "index": "vector",
                    "path": "embedding",
                    # "path": "combined_embedding",
                    "queryVector": query_vector,
                    "numCandidates": 9000,
                    # "numCandidates": 500,
                    "limit": 100,
                    "metric": "cosine"
                }
                
            },
            {
                "$project": {
                    "_id": 0,
                    "id": 1,
                    "title": 1,
                    "company_logo": 1,
                    "description": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {"$skip": skip}, 
            {"$limit": page_size}
        ]

        results = list(test_collection.aggregate(pipeline))#list(jobs_collection.aggregate(pipeline))

        if not results:
            raise HTTPException(status_code=404, detail="No matching jobs found")

        return {
            "page": page,
            "page_size": page_size,
            "total_results": page_count if page_count < 100 else 100,
            "recommendations": results
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


##########################CRUD##########################
class Job(BaseModel):
    id: int
    title: str
    description: str
    location: str
    experince: str
    combined_embedding: Optional[list[float]] = None
    status: str
    type_of_job: str
    attend: str
    specialization: str
    company: int
    company_name: str
    company_logo: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # {'id': 31, 'title': 'Backend Engineer', 'description': 'Django and FastAPI experience required', 'location': 'Remote', 'status': 'open'
    #  , 'type_of_job': 'Full-time', 'experince': 'Mid-level', 'company': 8, 'company_name': 'Aisha Amr', 'company_logo': None}
    

    # @validator("title", "description")
    # def must_not_be_empty(cls, value):
    #     if not value.strip():
    #         raise ValueError("Field cannot be empty")
    #     return value

    # @validator("id")
    # def must_have_id(cls, value):
    #     if not value:
    #         raise ValueError("Must include job ID")
    #     return value
    
    @field_validator("title", "description")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("id")
    @classmethod
    def must_have_id(cls, v: int) -> int:
        if v is None:
            raise ValueError("Must include job ID")
        return v
def recommend_emails(job):
    try:
        job["combined_embedding"] = get_embedding(job["description"] + " " + " ".join(job["title"]))
        # print(job)
        inserted_job = jobs_collection.insert_one(job)
        results = list(users_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "default",
                    "queryVector": job["combined_embedding"],
                    "path": "embedding",
                    "numCandidates": 1000,
                    "limit": 10,
                    "metric": "cosine"
                }
            }
        ]))
        mail = os.getenv("MAIL_SERVICE")
        front = os.getenv("FRONT")
        requests.post(mail + "/send_recommendation", json={"emails": [user["email"] for user in results], "job_title": job["title"], 'job_link': front + "applicant/jobs/" + str(job["id"]), 'company_name': job["company_name"]})
    except Exception as e:
        print(e)
    return {"message": "Emails sent successfully"}


# @app.post("/jobs", dependencies=[Depends(verify_token)])
@app.post("/jobs")
async def create_job( request: Request, background_tasks: BackgroundTasks):
    job = await request.json()
    print(job)
    job_data = job
    background_tasks.add_task(recommend_emails, job_data)
    return {"message": "Job created successfully"}


# Commentb l7d m ashof ha7tagha wala la2
# @app.get("/jobs/")
# async def get_jobs(limit: int = 10, skip: int = 0):
#     jobs = list(jobs_collection.find({}, {"_id": 1, "title": 1, "description": 1, "title": 1})
#                 .skip(skip).limit(limit))
#     for job in jobs:
#         job["_id"] = str(job["_id"])
#     return {"jobs": jobs}


# @app.get("/jobs/{job_id}")
# async def get_job(job_id: str):
#     job = jobs_collection.find_one({"_id": ObjectId(job_id)})
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")
#     job["_id"] = str(job["_id"])
#     return job

@app.put("/jobs/{job_id}")
async def update_job(job_id: int, job: Job):
    updated_job = job.dict()
    old_job = jobs_collection.find_one({"id": job_id})
    
    if old_job and old_job['description'] != updated_job["description"]:
        updated_job["combined_embedding"] = get_embedding(updated_job["description"] + " " + " ".join(updated_job["title"]))
    
    result = jobs_collection.update_one({"id": job_id}, {"$set": updated_job}, upsert=True)

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found in mongodb")

    return {"message": "Job updated successfully"}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: int):
    print("job_id",job_id)
    existing_job = jobs_collection.find_one({"id": job_id})
    print("existing_job",existing_job)
    if not existing_job:
        raise HTTPException(status_code=404, detail="Job not found fastapi mongodb")
    result = jobs_collection.delete_one({"id": job_id})
    print(result)
    print(result)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found in mongodb")
    return {"message": "Job deleted successfully"}

##########################CRUD##########################

class ATSRequest(BaseModel):
    cv_url: str
    job_id: int
    application_id: Optional[int]


# def preprocess_text(text):
#     text = re.sub(r'\W+', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     return text.lower()
def preprocess_rag_text(text):
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def calculate_embedding_similarity(cv_embedding, job_embedding):
    similarity_score = util.pytorch_cos_sim(cv_embedding, job_embedding).item()
    return similarity_score


@app.post("/ats/{user_id}/{job_id}")
async def ats_system(user_id: int , job_id: int, request: ATSRequest, db: AsyncSession = Depends(lambda: async_session())):
    
    print(f"Received ATS request for user_id={user_id}, job_id={job_id}")
    cv_url = request.cv_url

    
    application_id = request.application_id

    job = jobs_collection.find_one({"id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_embedding = job.get("combined_embedding", None)

    if not job_embedding:
        raise HTTPException(status_code=500, detail="Job embedding missing")

    # Get CV URL
    cv_url = format_cv_url(request.cv_url)
    print(f"cv_url={cv_url}")

    cv_embedding = check_user_cv(user_id, cv_url)
    
    if not cv_embedding or not job_embedding:
        raise HTTPException(status_code=500, detail="Embedding computation failed")
    
    try:
        similarity_score = calculate_embedding_similarity(cv_embedding, job_embedding)
    except Exception as e:
        print(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail="Similarity computation failed")
    application_table = await get_user_table("applications_application")
    result = round(similarity_score * 100, 2)
    query = (
        update(application_table)
        .where(application_table.c.id == application_id)
        .values(ats_res=result)
        .execution_options(synchronize_session="fetch")
    )
    await db.execute(query)
    await db.commit()
    return {"status": "updated"}
    # return {"match_percentage": round(similarity_score * 100, 2), "message": "Higher score means a better match!"}


@app.post("/rag")
async def rag_system(pdf: UploadFile = File(...)):
    # Create the temp directory if it doesn't exist
    os.makedirs("./temp", exist_ok=True)
    rag_names_collection.insert_one({"name": pdf.filename.replace(".pdf", ""), "created_at": datetime.utcnow()})
    try:
        file_path = f"./temp/{pdf.filename}"
        with open(file_path, "wb") as f:
            f.write(pdf.file.read())
        
        start_time = time.time()
        chunks = process_pdf_and_get_chunks(file_path, pdf.filename.replace(".pdf", ""))
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken = round(time_taken, 2)
        print(f"Time taken to process PDF: {time_taken} seconds")

        rag_collection.insert_many(chunks)
        os.remove(file_path)
        return {"message": f"PDF uploaded and processed successfully in {time_taken} seconds"}
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF processing failed")

def process_pdf_and_get_chunks(file_path: str, pdf: str):
    print(f"Processing PDF with PyPDFLoader: {file_path}")
    try:
        # Load PDF using PyMuPDF
        pdf_stream = BytesIO(open(file_path, "rb").read())
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()

        if not text:
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        text = preprocess_rag_text(text)
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunked_text = text_splitter.split_text(text)
        
        chunks_with_metadata = []
        for chunk in chunked_text:
            chunk_data = {
                "text": chunk,
                "embedding": get_embedding(chunk),
                "metadata": pdf
            }
            chunks_with_metadata.append(chunk_data)
        
        print(f"Successfully split PDF into {len(chunks_with_metadata)} chunks.")
        return chunks_with_metadata
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        # Raise HTTPException to return a proper API error response
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/ask_rag")
async def ask_rag(question: str, chat_history: list[dict[str, str]] = []):
    """
    Performs a RAG query against indexed PDF data in MongoDB, includes chat history,
    and asks OpenAI.
    """
    query_text = question
    print(f"Received query: '{query_text}'")

    if not query_text:
         raise HTTPException(status_code=400, detail="Question is required.")

    try:
        # 1. Get embedding for the user query
        start_time_embedding = time.time()
        query_embedding = get_embedding(query_text)
        time_taken_embedding = round(time.time() - start_time_embedding, 2)
        print(f"Time taken to embed query: {time_taken_embedding} seconds")

        if not query_embedding:
             raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")

        # 2. Search MongoDB for relevant chunks using vector search
        start_time_search = time.time()
        search_results = list(rag_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "rag_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 1000,
                    "limit": 10,
                    "metric": "cosine"
                }
            },
            {
                 "$project": {
                    "_id": 0,
                    "text": 1,
                    "score": { "$meta": "vectorSearchScore" }
                 }
            }
        ]))
        time_taken_search = round(time.time() - start_time_search, 2)
        print(f"Time taken for vector search: {time_taken_search} seconds. Found {len(search_results)} results.")

        # 3. Format the retrieved chunks as context
        context = "\n\n---\n\n".join([doc["text"] for doc in search_results])

        # Handle case where PDF doesn't exist or no relevant chunks found for current query
        if not search_results:
            if not chat_history:
                  # No search results and no history, return specific message
                  return {"answer": f"Could not find relevant information for the question based on the content in our database. Please try rephrasing your question."}
             # If search results are empty but there IS chat history, we still proceed
            print("No new relevant chunks found, relying on history and prompt.")


        # 4. Prepare messages for OpenAI, including history and context
        messages = [
            {
                "role": "system",
                "content": (
                    "Dont metion anything about being looking at a provided context. "
                    "You are a helpful assistant that answers questions based on the provided context. "
                    "You can use external knowledge if needed but with a focus on the provided context. "
                    # "Use the chat history to understand the conversation flow and user intent. "
                    # "If the answer cannot be found in the *provided context* and the history doesn't provide enough information, respond with 'I am sorry, but the information needed to answer this question is not available in the provided document or previous conversation context.' "
                    # "Do not use external knowledge. Keep the answer concise and directly address the user's question."
                )
            }
        ]

        # Add previous chat history to the messages list
        messages.extend(chat_history)

        # Add the current user query, incorporating the retrieved context
        current_user_message_content = f"Context:\n{context}\n\nQuestion: {query_text}"

        messages.append({
            "role": "user",
            "content": current_user_message_content
        })

        print(f"Sending {len(messages)} messages to OpenAI API.")
        # print("Messages structure:", messages) # Uncomment to debug the full message structure

        # 5. Call OpenAI API
        start_time_openai = time.time()
        openai.api_key = os.getenv('OPEN_AI')
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,
            max_tokens = 384
        )
        time_taken_openai = round(time.time() - start_time_openai, 2)
        print(f"Time taken for OpenAI API call: {time_taken_openai} seconds")

        # 6. Extract the answer
        answer = response.choices[0].message.content.strip()
        print("Answer:", answer)
        # 7. Return the answer
        # The client is responsible for adding the current query and this answer to its history
        return {"answer": answer}

    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        print(f"An error occurred during RAG query with history: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

################### resume parse #######################
import json
from typing import List, Optional
from pydantic import BaseModel

class EducationItem(BaseModel):
    degree: str
    school: str
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    fieldOfStudy: Optional[str] = None

class ExperienceItem(BaseModel):
    title: str
    company: str
    startDate: Optional[str] = None
    endDate: Optional[str] = None

class CVData(BaseModel):
    about: str
    skills: List[str]
    education: List[EducationItem]
    experience: List[ExperienceItem]


def get_embedd_cv_extract(cv, user_id,cv_url):
    print("embedd dats ",cv, user_id,cv_url)
    embed= get_embedding(cv)
    users_collection.update_one(
         {"user_id": user_id},
         {"$set": {"embedding": embed, "cv_url": cv_url}}  
    )
    
    
    
@app.get("/extract-cv-data/")
async def extract_cv_data(cv_url: str , user_id: int ,background_tasks: BackgroundTasks, update: bool = True ):
    try:
        parsed = {}
        public_id = format_cv_url(cv_url)
        text = extract_text_from_pdf_cloud(public_id)

        if not text:
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        
        if update:
            prompt = f"""
                    You are an intelligent CV parser. From the following resume text, return a valid JSON object with the following keys:

                    1. "Summary": Generate a professional 5–7 midium length sentence summary based on the full CV content. This must be an original synthesis, not copied from the CV. Clearly identify the candidate's job specialization or target role based on their skills and experience (e.g., "Machine Learning Engineer" or "Full-Stack Developer"). Also include a brief summary of the types of projects they have worked on, highlighting key technologies or outcomes.
                    2. "About": Extract the **first personal paragraph or section** that appears after the contact information (name, email, phone, location). This is typically an unlabeled personal introduction, profile, or "About Me" paragraph.
                    3. "Skills": A list of technical and soft skills.
                    4. "Education": A list of objects with: degree, school, startDate, endDate, fieldOfStudy.
                    5. "Experience": A list of objects with: title, company, startDate, endDate.

                    Return only valid JSON, no markdown or explanation.

                    CV Text:
                    {text[:3000]}
                    """

            start_time = time.time()
            openai.api_key = os.getenv('OPEN_AI')
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            duration = round(time.time() - start_time, 2)
            print(f"✅ OpenAI API call succeeded in {duration} seconds")
    
            result = response.choices[0].message.content.strip()
            # print("🧠 Raw OpenAI Response:\n", result)

            # Parse JSON from response
            json_start = result.find('{')
            json_str = result[json_start:].split("```")[0]
            parsed = json.loads(json_str)
            print("🧠 Parsed OpenAI Response:\n", parsed)
            
            # Normalize missing fields
            parsed.setdefault("About", "")
            parsed.setdefault("Summary", "")
            parsed.setdefault("Skills", [])
            parsed.setdefault("Education", [])
            parsed.setdefault("Experience", [])

            # Normalize skills list
            parsed["Skills"] = [s.strip() for s in parsed["Skills"] if isinstance(s, str)]

            # Ensure Education and Experience entries have required keys
            for edu in parsed["Education"]:
                edu.setdefault("degree", "")
                edu.setdefault("school", "")
                edu.setdefault("startDate", "")
                edu.setdefault("endDate", "")
                edu.setdefault("fieldOfStudy", "")
            
            for exp in parsed["Experience"]:
                if isinstance(exp, dict):
                    exp.setdefault("title", "")
                    exp.setdefault("company", "")
                    exp.setdefault("startDate", "")
                    exp.setdefault("endDate", "")
                else:
                    # fallback in case it is just a string
                    parsed["Experience"] = [{
                        "title": exp,
                        "company": "",
                        "startDate": "",
                        "endDate": ""
                    }]        
                    # } for exp in parsed["Experience"]]
                    break
        background_tasks.add_task(get_embedd_cv_extract,text,user_id,public_id)
        return parsed if parsed else {}

    except Exception as e:
        print("❌ GPT API failed:", e)
        raise HTTPException(status_code=500, detail=f"AI parsing failed: {e}")
######### talents in company table ##############


@app.get("/top_talents")
async def top_talents(job_id: int, page: int = 1, page_size: int = 5, seniority: str = None):
    try:
        if not job_id:
            raise HTTPException(status_code=400, detail="Job ID is required")
        if page < 1 or page_size < 1:
            raise HTTPException(status_code=400, detail="Page numbers must be 1 or higher")

        page_count= users_collection.count_documents({'embedding': {'$exists': True}})

        if page_count == 0:
            raise HTTPException(status_code=404, detail="No users found")
        if page*page_size > 100:
            raise HTTPException(status_code=400, detail="Max recommendations is 100")
        if page > page_count/page_size and page > 1:
            raise HTTPException(status_code=400, detail="Page number exceeds available pages")

        job = jobs_collection.find_one({"id": job_id})
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        combined_embedding = job["combined_embedding"]
        if not combined_embedding:
            raise HTTPException(status_code=500, detail="Combined embedding not found for the job")

        # Get top talents
        skip = (page - 1) * page_size
        vector_search = [
            {
                "$vectorSearch": {
                    "index": "default",
                    "queryVector": job["combined_embedding"],
                    "path": "embedding",
                    "numCandidates": 1000,
                    "limit": 100,
                    "metric": "cosine"
                }
            },
            {
                 "$project": {
                    "_id": 0,
                    "id": '$user_id',
                    "ats_res": { "$meta": "vectorSearchScore" },
                    "name": 1,
                    "email": 1
                 }
            },
            {"$skip": skip},  # Skip previous pages
            {"$limit": page_size}  #  Limit results per page
        ]

        if seniority:
            vector_search.insert(0, {"$match": {"seniority": seniority}})

        results = list(users_collection.aggregate(vector_search))

        if not results or len(results) == 0:
            raise HTTPException(status_code=404, detail="No matching talents found")
        # print(results)
        return {
            "count": page_count if page_count < 100 else 100,
            "results": results
        }
    except Exception as e:
        print(f"An error occurred while retrieving the job: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

#################### video interview ####################







class InterviewRequest(BaseModel):
    video_url: str
    job_id: int
    application_id: int
    question_id: int
    question_text: str
    # applicant_email: str
    # question_id: int
    
class AnswerAnalysisResult(TypedDict):
    ideal_answer: str
    score: float
    feedback: str
    semantic_score: float
    gpt_score: float
    key_points_covered: List[str]
    key_points_missed: List[str]


    
class InterviewAnalyzer:
    def __init__(self):
        load_clip_model()  # Ensure CLIP model is loaded
        self.attire_classes = [
            "a professional person wearing formal business attire",
            "a professional person wearing smart casual clothing",
            "a professional person wearing casual clothing",
            "a professional person wearing inappropriate interview clothing"
        ]
        self.attire_class_weights = {
            0: 9.0,  # Formal business attire
            1: 7.0,  # Smart casual
            2: 4.0,  # Casual
            3: 2.0   # Inappropriate
        }
        
        # def _load_attire_model(self):
        #     """Load pre-trained Vision Transformer model and processor"""
        # model_name = "google/vit-base-patch16-224"
        # processor = ViTImageProcessor.from_pretrained(model_name)
        # model = ViTForImageClassification.from_pretrained(
        #     model_name,
        #     num_labels=4,  # For formal, smart casual, casual, inappropriate
        #     ignore_mismatched_sizes=True  # Allows loading fine-tuned weights
        # )
        # model.eval()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # return model.to(device), processor

    
    def compute_similarity(self, text1, embedding2):
        embedding1 = model.encode(text1, convert_to_tensor=True)
        # embedding2 = model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(embedding1, embedding2).item()  # float value between -1 and 1


    async def analyze_interview(
            self,
            video_path: str,
            job_id: int,
            application_id: int,
            question_id: int,
            question_text: str,    
            db: AsyncSession = Depends(lambda: async_session()),  # just a plain parameter now
        ):
        print ("hello from analyze interview")
        try:
            print ("video_path",video_path)
            # Validate video file
            if not os.path.isfile(video_path):
                raise ValueError("Video file does not exist")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Unable to open video file")
            cap.release()
            
            job = jobs_collection.find_one({"id": job_id})
            
            job_embedding = job['combined_embedding']
            if(not job_embedding):
                job_embedding = get_embedding(job["description"] + " " + " ".join(job["title"]))
                jobs_collection.update_one({"id": job_id}, {"$set": {"combined_embedding": job_embedding}})
            # Perform analyses
            transcript = self.transcribe_video(video_path)
            answer_analysis = self.analyze_answer(str(transcript), job_embedding,question_text)
            pronunciation_score = self.analyze_pronunciation(video_path)
            grammar_score = self.analyze_grammar(transcript)
            attire_score = self.analyze_attire(video_path)
            print (" analyzed interview transcript",transcript
                   ,"answer_score",answer_analysis['score']
                   ,"pronunciation_score",pronunciation_score
                   ,"grammar_score",grammar_score
                   ,"attire_score",attire_score)
            # Calculate total score
            total_score = (
                answer_analysis['score'] * 0.5 +
                pronunciation_score * 0.2 +
                grammar_score * 0.2 +
                attire_score * 0.1
            )
            print ("total_score",total_score)

            # return {
            #     "answer_score": round(answer_score, 2),
            #     "pronunciation_score": round(pronunciation_score, 2),
            #     "grammar_score": round(grammar_score, 2),
            #     "eye_contact_score": round(eye_contact_score, 2),
            #     "attire_score": round(attire_score, 2),
            #     "total_score": round(total_score, 2),
            #     "transcript": transcript
            # }
            
            
            application_table = await get_user_table("applications_application")
            answer_table = await get_user_table("answers_answer")
            result =float(round(total_score * 10, 2))
            print(f"Updating application_id={application_id} with screening_res={result}")

            question_text = f'Screening question for {job["title"]} job at {job["company_name"]}'
            res = {
                "question": question_text,
                "user_answer": transcript,
                # "ideal_answer": answer_analysis.get('ideal_answer',''),
                "answer_score": round(answer_analysis['score'], 2),
                "answer_score": round(answer_analysis['score'], 2),
                "semantic_score": round(answer_analysis['semantic_score'], 2),
                "pronunciation_score": round(pronunciation_score, 2),
                "grammar_score": round(grammar_score, 2),
                "attire_score": round(attire_score, 2),
                "total_score": round(total_score, 2),
                # "transcript": transcript,
                "feedback": answer_analysis.get('feedback', ''),
                "attire_feedback": self._get_attire_feedback(attire_score),
                "key_points_covered": answer_analysis.get('key_points_covered', []),
                "key_points_missed": answer_analysis.get('key_points_missed', []),
                "attire_feedback": self._get_attire_feedback(attire_score),
            }
            
            # send_to_queue('email_queue', 'post', 'send-report', res)
            url = os.getenv("MAIL_SERVICE")+"send-report"
            print("url*****************************",url)
            
            requests.post(url, json=res)
            
            print(res)
            query = (
                update(application_table)
                .where(application_table.c.id == application_id)
                .values(screening_res=result)
                .execution_options(synchronize_session="fetch")
            )
            
            # answer = answer_table(answer_text=result, application=application_id, question=question_id)
            # await db.execute(create(answer))
            await db.execute(query)
            await db.commit()
            
            return res
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def transcribe_video(self, video_path: str) -> str:
        try:
            model = whisper.load_model("medium")
            result = model.transcribe(video_path)
            print ("transcript from  transcribe_video",result['text'])
            if not result['text']:
                raise ValueError("Transcription failed")
            return result['text']
        except Exception as e:
            raise ValueError(f"Transcription failed: {str(e)}")
                    
                    
    def analyze_answer(self, transcript: str, job_embedding: list, question_text: str) -> AnswerAnalysisResult:
        try:
            # --- 1. Semantic Similarity ---
            # IS THIS EMBEDDING RIGHT ?
            
            similarity_score = self.compute_similarity(transcript, job_embedding)
            normalized_similarity_score = max(0, min(10, (similarity_score + 1) * 5))
            print("normalized_similarity_score",normalized_similarity_score)
            # --- 2. GPT Contextual Analysis ---
            openai.api_key = os.getenv("OPEN_AI")
            prompt = f"""
            You are a senior hiring expert. Evaluate the following candidate's answer to the question:
            Question:{question_text}
            Candidate Answer:
            {transcript}

            Provide:
            1. Score (0-10) for relevance, technical quality, and completeness
            2. A short paragraph of feedback
            3. Key points covered (list)
            4. Key points missed (list)

            Return in JSON:
            {{
                "score": float (0-10),
                "feedback": "...",
                "key_points_covered": [...],
                "key_points_missed": [...]
            }}
            """
            print("prompt",prompt)
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            
            result = json.loads(response.choices[0].message.content)
            print("result",result)
            gpt_score = float(result.get("score", 0))
            hybrid_score = round((gpt_score * 0.7) + (normalized_similarity_score * 0.3), 2)
            print ("gptscore",gpt_score)
            print(" hybrid_score", hybrid_score)
            return {
                "ideal_answer": "",
                "score": hybrid_score,
                "feedback": result.get("feedback", ""),
                "semantic_score": round(normalized_similarity_score, 2),
                "gpt_score": round(gpt_score, 2),
                "key_points_covered": result.get("key_points_covered", []),
                "key_points_missed": result.get("key_points_missed", []),
            }
                     
        except Exception as e:
            print(f"Hybrid answer analysis error: {str(e)}")
            return {
                "ideal_answer": "",
                "score": 3.0,
                "feedback": "Evaluation failed.",
                "semantic_score": 0.0,
                "gpt_score": 0.0,
                "key_points_covered": [],
                "key_points_missed": []
            }


    def analyze_grammar(self, transcript: str) -> float:
        try:
            
            prompt = f"Evaluate the grammar of the following text and provide ONLY a numerical score between 0 and 10:\n\n{transcript}"
            openai.api_key = os.getenv('OPEN_AI')
            response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are a grammar evaluator. Provide only a numerical score between 0-10."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            print("Grammar score:", content)
            return float(content)
        except Exception as e:
            print(f"Grammar analysis failed: {str(e)}")
            return 0.0
    def analyze_pronunciation(self, video_path: str) -> float:
        try:
            model = whisper.load_model("base")
            result = model.transcribe(video_path, word_timestamps=True)
            segments = result.get('segments', [])
            if not segments:
                return 5.0  

            LONG_WORD_THRESHOLD = 1.2  # seconds
            LOW_CONFIDENCE_THRESHOLD = 0.7
            VERY_LOW_CONFIDENCE_THRESHOLD = 0.5
            
            word_stats = {
                'total': 0,
                'unclear': 0,
                'very_unclear': 0,
                'total_duration': 0.0
            }

            for segment in segments:
                words = segment.get('words', [])
                for word_info in words:
                    word_stats['total'] += 1
                    duration = word_info.get('end', 0) - word_info.get('start', 0)
                    confidence = word_info.get('confidence', 1)
                    
                    # Count problematic words
                    if duration > LONG_WORD_THRESHOLD:
                        word_stats['unclear'] += 1
                    if confidence < LOW_CONFIDENCE_THRESHOLD:
                        word_stats['unclear'] += 1
                    if confidence < VERY_LOW_CONFIDENCE_THRESHOLD:
                        word_stats['very_unclear'] += 1
                    
                    word_stats['total_duration'] += duration

            if word_stats['total'] == 0:
                return 5.0

            # Calculate clarity metrics
            avg_word_duration = word_stats['total_duration'] / word_stats['total']
            unclear_ratio = word_stats['unclear'] / word_stats['total']
            very_unclear_ratio = word_stats['very_unclear'] / word_stats['total']

            # Score calculation (0-10 scale)
            base_score = 8.0  # Starting from slightly positive
            base_score -= (unclear_ratio * 4)  # Subtract up to 4 points for unclear words
            base_score -= (very_unclear_ratio * 3)  # Subtract up to 3 more for very unclear
            base_score -= max(0, (avg_word_duration - 0.8) * 2)  # Penalize slow speech
            
            # Ensure score is within bounds
            final_score = max(1.0, min(10.0, base_score))
            
            return round(final_score, 2)

        except Exception as e:
            print(f"Pronunciation analysis failed: {str(e)}")
            return 5.0  # Neutral score on error
    
    def _get_attire_feedback(self, score: float) -> str:
        """Generate feedback based on attire score"""
        if score >= 8.5:
            return "Excellent professional appearance"
        elif score >= 7:
            return "Appropriate professional attire"
        elif score >= 5:
            return "Could improve professional appearance"
        else:
            return "Inappropriate attire for professional interview"

    def _extract_key_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """Extract evenly spaced frames from video focusing on upper body"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            
            # Get evenly spaced frame indices
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert to RGB and crop to upper body (assuming person is centered)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = frame.shape[:2]
                    upper_body = frame[:int(height*0.7), :]  # Take top 70% of image
                    frames.append(upper_body)
                    
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Frame extraction failed: {str(e)}")
            return []
    
    def analyze_attire(self, video_path: str) -> float:
        """Analyze attire formality using CLIP"""
        if CLIP_MODEL is None:
            return 1.0  # Neutral score if model failed to load
            
        try:
            frames = self._extract_key_frames(video_path, num_frames=3)
            if not frames:
                return 5.0
                
            # Prepare text inputs
            text_inputs = torch.cat([clip.tokenize(c) for c in self.attire_classes]).to(DEVICE)
            print ("text inputs",text_inputs)
            
            print("\nAttire classes being evaluated:")
            for i, class_desc in enumerate(self.attire_classes):
                print(f"{i}: {class_desc}")
                
            scores = []
            for frame_idx, frame in enumerate(frames):
                print(f"\nAnalyzing frame {frame_idx + 1}:")
                # Preprocess image
                img = Image.fromarray(frame)
                image_input = CLIP_PREPROCESS(img).unsqueeze(0).to(DEVICE)
                
                # Calculate features
                with torch.no_grad():
                    image_features = CLIP_MODEL.encode_image(image_input)
                    text_features = CLIP_MODEL.encode_text(text_inputs)
                
                # Get similarity scores
                logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = logits_per_image.cpu().numpy()[0]
                 # Apply smoothing to prevent extreme probabilities
                probs = np.clip(probs, 0.05, 0.95)
                probs = probs / probs.sum()
                print ("probs",probs)
                print("logits_per_image",logits_per_image)
                print("Class probabilities:")
                for i, prob in enumerate(probs):
                   print(f"- {self.attire_classes[i]}: {prob:.2%}")
            
                # Calculate weighted score
                frame_score = sum(
                    self.attire_class_weights[i] * probs[i] 
                    for i in range(len(self.attire_classes)))
                scores.append(frame_score) 
                print
            # Average scores across frames and clamp to 0-10 range
            avg_score = np.mean(scores)
            final_score = min(max(avg_score, 0), 10)
            print(f"\nFinal attire score: {final_score:.2f}")
            return final_score
            
        except Exception as e:
            print(f"CLIP attire analysis failed: {str(e)}")
            return 1.0








def delete_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Failed to delete temp file {path}: {e}")


@app.post("/analyze-interview/")
async def analyze_interview_endpoint(request: InterviewRequest, background_tasks: BackgroundTasks,db: AsyncSession = Depends(lambda: async_session()))-> Any:  #
    print("hello")
    try:
        # Download video
        video_path = download_video_from_url(request.video_url)
        print ("video path",video_path)
        print ("hello after download video")
        # Step 2: Schedule deletion in background
        background_tasks.add_task(delete_file, video_path)

        # Step 3: Run analyzer
        analyzer = InterviewAnalyzer()
        print ("hello after analyzer")
        # Analyze interview
        results = await analyzer.analyze_interview(
            video_path=video_path,
            job_id=request.job_id,
            application_id=request.application_id,
            question_id=request.question_id,
            question_text=request.question_text,
            db=db,
        )
        # results = await analyzer.analyze_interview(
        #     video_path=video_path,
        #     question=request.question,
        #     job_description=request.job_description
        # )
        print ("hello after analyze interview")
        print ("intervew results",results)
        
        res = requests.post(os.getenv("MAIL_SERVICE") + "send_report", json=results)
        if res.status_code != 200:
            print("Failed to send email report")
            # raise HTTPException(status_code=500, detail="Failed to send email report")
        
        print("Analysis complete. Report will be sent via email.")

        return results
        #return {"message": "Analysis complete. Report will be sent via email."}

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))  
    
    
    
    
    
    
    
    # def analyze_answer(self,transcript: str, job_embedding: list) -> dict:
    #     try:
    #         # Calculate similarity between answer and job description
    #         similarity_score = self.compute_similarity(transcript, job_embedding)
              
    #         # Normalize similarity score to 0-10 scale (cosine similarity is -1 to 1, but typically 0-1 for text)
    #         normalized_score = max(0, min(10, (similarity_score + 1) * 5))  # maps [-1,1] to [0,10]
            
    #         # Generate feedback based on similarity
    #         if similarity_score < 0.3:
    #             feedback = "The answer shows little relevance to the job requirements."
    #             score = 3.0
    #         elif similarity_score < 0.5:
    #             feedback = "The answer has some relevance but could better address the job requirements."
    #             score = 5.0
    #         elif similarity_score < 0.7:
    #             feedback = "The answer is relevant to the job requirements but could be more specific."
    #             score = 7.0
    #         else:
    #             feedback = "The answer demonstrates strong relevance to the job requirements."
    #             score = 9.0
            
    #         return {
    #             'similarity_score': similarity_score,
    #             'score': score,
    #             'normalized_score': normalized_score,
    #             'feedback': feedback
    #         }
            
    #     except Exception as e:
    #         print(f"Answer analysis error: {str(e)}")
    #         return {
    #             'similarity_score': 0.0,
    #             'score': 2.0,
    #             'normalized_score': 2.0,
    #             'feedback': 'Evaluation could not be completed'
    #         }   
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
           
    # def analyze_attire(self, video_path: str) -> float:
    #     try:
            
    #         print("response",response)
    #         # Initialize MediaPipe solutions outside the processing loop
    #         mp_pose = mp.solutions.pose
    #         mp_selfie_segmentation = mp.solutions.selfie_segmentation
            
    #         # Use context managers to ensure proper cleanup
    #         with mp_pose.Pose(
    #             static_image_mode=False, 
    #             min_detection_confidence=0.5
    #         ) as pose, mp_selfie_segmentation.SelfieSegmentation(
    #             model_selection=1
    #         ) as selfie_segmentation:
                
    #             cap = cv2.VideoCapture(video_path)
    #             if not cap.isOpened():
    #                 return 0.0

    #             frame_count = 0
    #             formal_score = 0
    #             sample_rate = 5  # Process every 5th frame

    #             # Define ImageNet classes for formal attire identification
    #             imagenet_classes = [
    #                 'suit', 'tie', 'shirt', 'business suit', 'bow tie', 'jeans',
    #                 'sweatshirt', 'T-shirt', 'sweat pants', 'pajamas', 'lab coat'
    #             ]

    #             def is_formal_attire(detected_classes):
    #                 # print("detected_classes",detected_classes)
    #                 formal_keywords = {'suit', 'tie', 'shirt', 'business suit', 'bow tie'}
    #                 return any(keyword in detected_classes for keyword in formal_keywords)

    #             while cap.isOpened():
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     break
                        
    #                 frame_count += 1
    #                 if frame_count % sample_rate != 0:
    #                     continue  # Sample every 5th frame for efficiency

    #                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 results = pose.process(rgb_frame)
    #                 seg_results = selfie_segmentation.process(rgb_frame)
    #                 segmentation_mask = seg_results.segmentation_mask > 0.5
    #                 if frame_count == 3:
    #                         try:
    #                             try:
    #                                 response = requests.post('https://detect.roboflow.com/detection-z9fo6/8?api_key=RNnorhEJpl4HX25mDAj4',data=self.image_to_base64(rgb_frame),
    #                                     headers={"Content-Type": "application/json"})
    #                                 response.raise_for_status()
    #                             except requests.exceptions.RequestException as e:
    #                                 print("Attire analysis error:",e)
    #                                 return 0.0
    #                         except Exception as e:
    #                             print("Attire analysis error:",e)
    #                 if results.pose_landmarks:
    #                     landmarks = results.pose_landmarks.landmark
    #                     h, w, _ = frame.shape
                        
    #                     # Get upper body region
    #                     left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    #                     right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    #                     nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    #                     x1 = int(left_shoulder.x * w)
    #                     x2 = int(right_shoulder.x * w)
    #                     y1 = int(nose.y * h)
    #                     y2 = int((left_shoulder.y * h + right_shoulder.y * h)/2 + 0.2 * h)

    #                     # Ensure coordinates are valid
    #                     x1, x2 = sorted((max(0, x1), min(w, x2)))
    #                     y1, y2 = max(0, y1), min(h, y2)

    #                     # Apply segmentation mask to focus on person
    #                     chest_region = frame[y1:y2, x1:x2]
    #                     if chest_region.size == 0:
    #                         continue

    #                     # Apply mask to region
    #                     region_mask = segmentation_mask[y1:y2, x1:x2]
    #                     masked_region = cv2.bitwise_and(
    #                         chest_region, 
    #                         chest_region, 
    #                         mask=region_mask.astype(np.uint8)
    #                     )
                        

    #                     # Simulate detection of attire classes (placeholder for actual model)
    #                     detected_classes = self.detect_classes(masked_region, imagenet_classes)
                        
    #                     # Check for formal attire
    #                     if is_formal_attire(detected_classes):
    #                         # if is_formal_color(avg_color) or has_tie(masked_region):
    #                         formal_score += 1

    #             cap.release()

    #             if frame_count == 0:
    #                 return 0.0
    #             # print("formal_score",formal_score)
    #             # Calculate percentage of frames with formal attire
    #             percentage = (formal_score / (frame_count // sample_rate)) * 10
    #             return round(percentage, 2)

    #     except Exception as e:
    #         print(f"Attire analysis error: {str(e)}")
    #         return 0  # Return a default score if analysis fails

    # def image_to_base64(self, image):
    #     _, encoded_image = cv2.imencode('.jpg', image)
    #     img_str = base64.b64encode(encoded_image).decode('utf-8')
    #     print("img_str",img_str)
    #     return img_str
    # def detect_classes(self, image, classes):
    #     # Placeholder for actual model-based class detection logic
    #     # This should be replaced with a model inference call
    #     # preprocessed_image = preprocess_input(image.img_to_array(image))
    #     # expanded_image = np.expand_dims(image, axis=0)
    #     image = cv2.resize(image, (224, 224))
    #     image = image.reshape((1, 224, 224, 3))
    #     predictions = resnet_model.predict(image)
    #     decoded_predictions = decode_predictions(predictions, top=5)[0]
    #     # print("decoded_predictions",decoded_predictions)
    #     detected_classes = [class_name for (_, class_name, _) in decoded_predictions if class_name in classes]
    #     # print("detected_classes",detected_classes)
    #     return detected_classes
    #     # return ['suit', 'tie']  # Example detected classes

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def analyze_answer(self, transcript: str, question: str, job_description: str) -> AnswerAnalysisResult:
    #     try:
    #         # # Default response
    #         # default_response = {
    #         #     'ideal_answer': 'Error generating ideal answer',
    #         #     'score': 10.0,
    #         #     'feedback': 'Evaluation failed or could not be completed'
    #         # }

    #         # prompt = f"""Evaluate this interview answer based on relevance to the question and job requirements:
    #         # Job: {job_description}
    #         # Question: {question}
    #         # Answer: {transcript}

    #         # Provide evaluation in this exact format:
    #         # Ideal Answer: [what a strong answer would include]
    #         # Relevance Score: [0-10 score based on how relevant the answer is to the question]
    #         # Content Score: [0-10 score based on quality of content]
    #         # Feedback: [specific feedback on what's good and what needs improvement]

    #         # Scoring Guidelines:
    #         # - 0-3: Completely unrelated or nonsensical answer
    #         # - 4-6: Somewhat related but missing key points
    #         # - 7-8: Good answer with minor improvements needed
    #         # - 9-10: Excellent, comprehensive answer
            
    #         # Final Score: [average of Relevance and Content scores, 0-10]"""
        
    #         prompt = f"""**Interview Answer Evaluation**
        
    #         **Job Requirements:** {job_description}
    #         **Question Asked:** {question}
    #         **Candidate's Answer:** {transcript}

    #         Perform a strict evaluation with these criteria:
    #         1. RELEVANCE (0-10): How directly the answer addresses the question
    #         - 0-3: Completely irrelevant/off-topic
    #         - 4-6: Somewhat related but misses key points
    #         - 7-8: Mostly relevant with minor digressions
    #         - 9-10: Perfectly on-point
            
    #         2. CONTENT QUALITY (0-10): Depth and accuracy of information
    #         3. STRUCTURE (0-10): Logical flow and organization

    #         Provide output in EXACTLY this format:
    #         RELEVANCE_SCORE: [0-10]
    #         CONTENT_SCORE: [0-10]
    #         STRUCTURE_SCORE: [0-10]
    #         FINAL_SCORE: [average of above]
    #         IDEAL_ANSWER: [2-3 sentence model answer]
    #         FEEDBACK: [3-4 specific bullet points:
    #                 - What was good
    #                 - What needs improvement
    #                 - Specific suggestions]
            
    #         Important: Be strict about relevance. If answer is completely off-topic, 
    #         RELEVANCE_SCORE must be below 3 and FINAL_SCORE below 5."""
    #         openai.api_key = os.getenv('OPEN_AI') 
    #         response = openai.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {"role": "system", "content": "You are an interview evaluator.Provide scores between 0-10."},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             max_tokens=500,  # Increased from 20
    #             temperature=0.3
    #         )

    #         content = response.choices[0].message.content.strip()
    #         print(f"GPT Response:\n{content}")

    #         # More robust parsing
    #         result = {
    #             'ideal_answer': '',
    #             'score': 0.0,
    #             'feedback': ''
    #         }

    #         # current_field = None
    #         # for line in content.split('\n'):
    #         #     line = line.strip()
    #         #     if line.startswith("Ideal Answer:"):
    #         #         current_field = 'ideal_answer'
    #         #         parts[current_field] = line[13:].strip()
    #         #     elif line.startswith("Score:"):
    #         #         current_field = 'score'
    #         #         try:
    #         #             parts[current_field] = float(line[6:].strip())
    #         #         except ValueError:
    #         #             parts[current_field] = 0.0
    #         #     elif line.startswith("Feedback:"):
    #         #         current_field = 'feedback'
    #         #         parts[current_field] = line[9:].strip()
    #         #     elif current_field:
    #         #         parts[current_field] += "\n" + line
           
           
    #         # current_field = None
    #         # for line in content.split('\n'):
    #         #     line = line.strip()
    #         #     if line.startswith("Ideal Answer:"):
    #         #         current_field = 'ideal_answer'
    #         #         result[current_field] = line[13:].strip()
    #         #     elif line.startswith("Score:"):
    #         #         current_field = 'score'
    #         #         try:
    #         #             result[current_field] = float(line[6:].strip())
    #         #         except ValueError:
    #         #             result[current_field] = 50.0
    #         #     elif line.startswith("Feedback:"):
    #         #         current_field = 'feedback'
    #         #         result[current_field] = line[9:].strip()
    #         #     elif current_field:
    #         #         result[current_field] += " " + line


    #              # Ensure score is between 0-100
    #             # result['score'] = max(0, min(100, result['score']))
    #             # return result
                
    #              # Parse each component
    #         for line in content.split('\n'):
    #             line = line.strip()
    #             if line.startswith("Ideal Answer:"):
    #                 result['ideal_answer'] = line[13:].strip()
    #             elif line.startswith("Final Score:"):
    #                 try:
    #                     score = float(line[12:].strip())
    #                     result['score'] = min(10, max(0, score))  # Ensure within bounds
    #                 except ValueError:
    #                     pass
    #             elif line.startswith("Feedback:"):
    #                 result['feedback'] = line[9:].strip()

    #             return result
    #     except Exception as e:
    #         print(f"Answer analysis error: {str(e)}")
    #         return {
    #             'ideal_answer': 'Error in evaluation',
    #             'score': 5.0,
    #             'feedback': 'Evaluation could not be completed'
    #         }    

    #     # except Exception as e:
    #     #     print(f"Answer analysis error: {str(e)}")
    #     #     return default_response

    
    
    
    
    
   
@app.post("/register-user/")
async def register_user( cvs:  List[UploadFile] = File(...)):
    try:
        # Generate a new user ID (e.g., use a simple auto-increment or UUID)
        for cv in cvs:
            user_id = users_collection.count_documents({}) + 200
            
            # Read and process the CV file
            cv_content = await cv.read()
            pdf_stream = BytesIO(cv_content)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            text = "\n".join([page.get_text("text") for page in doc])
            # print ("text",text)
            doc.close()
            extracted_text = text.strip() if text else "No text found in PDF."
            cv_embedding = get_embedding(extracted_text)

            # Insert the new user into the database
            users_collection.insert_one({
                "user_id": user_id,
                "name": cv.filename.replace(".pdf", ""),
                "email": cv.filename.replace(".pdf", "").replace(" ", "_") + "@gmail.com",
                "embedding": cv_embedding
            })
            print("added name", cv.filename.replace(".pdf", ""))
        
        return {"message": "User registered successfully"}
    
    except Exception as e:
        print(f"An error occurred while registering the user: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

    