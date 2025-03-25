import sys
sys.path.append("..") 


from sentence_transformers import SentenceTransformer , util
from fastapi import FastAPI , HTTPException ,Query, UploadFile, File, Form , Header, Depends
from database import jobs_collection 
from pydantic import BaseModel,validator
from bson import ObjectId
import torch
import joblib
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import re
import fitz  # PyMuPDF
import spacy
import requests
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from loguru import logger
from functools import lru_cache
import requests  # To download Cloudinary files
from pdf2image import convert_from_bytes  # Convert scanned PDF to images
import pytesseract  # OCR for text extraction



# Security: API Key Middleware
def verify_api_key(api_key: str = Header(...)):
    if api_key != "your_secure_api_key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key






app = FastAPI()
#load  ats model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


#load recommender model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

@app.on_event("startup")
async def startup_event():
    #Preload model before serving requests
    global model
    model = load_model()
    logger.info("ML Model Loaded Successfully!")

def get_embedding(text):
    with torch.no_grad():
        return model.encode(text, convert_to_tensor=True).tolist()


async def recommend_jobs_background(user_skills: str):
    query_vector = get_embedding(user_skills)
    return {"embedding": query_vector}  



@app.get("/recom/")
async def recommend_jobs(user_skills: str, page: int = 1, page_size: int = 5):
    """AI-powered job recommendations with pagination"""
    
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be 1 or higher")
    
    query_vector = get_embedding(user_skills)
    skip = (page - 1) * page_size

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector",
                "path": "combined_embedding",
                "queryVector": query_vector,
                "numCandidates": 500,
                "limit": page_size,
                "metric": "cosine"
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "description": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {"$skip": skip},  # Skip previous pages
        {"$limit": page_size}  #  Limit results per page
    ]

    results = list(jobs_collection.aggregate(pipeline))

    if not results:
        raise HTTPException(status_code=404, detail="No matching jobs found")

    return {
        "page": page,
        "page_size": page_size,
        "total_results": len(results),
        "recommendations": results
    }



class Job(BaseModel):
    title: str
    description: str
    skills_required: list[str]

    @validator("title", "description")
    def must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Field cannot be empty")
        return value

    @validator("skills_required")
    def must_have_skills(cls, value):
        if len(value) == 0:
            raise ValueError("Must include at least one skill")
        return value


@app.post("/jobs/")
async def create_job(job: Job):
    job_data = job.dict()
    job_data["combined_embedding"] = get_embedding(job_data["description"] + " " + " ".join(job_data["skills_required"]))
    inserted_job = jobs_collection.insert_one(job_data)
    return {"id": str(inserted_job.inserted_id), "message": "Job created successfully"}

@app.get("/jobs/")
async def get_jobs(limit: int = 10, skip: int = 0):
    jobs = list(jobs_collection.find({}, {"_id": 1, "title": 1, "description": 1, "skills_required": 1})
                .skip(skip).limit(limit))
    for job in jobs:
        job["_id"] = str(job["_id"])
    return {"jobs": jobs}




@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["_id"] = str(job["_id"])
    return job

@app.put("/jobs/{job_id}")
async def update_job(job_id: str, job: Job):
    updated_job = job.dict()
    updated_job["combined_embedding"] = get_embedding(updated_job["description"] + " " + " ".join(updated_job["skills_required"]))
    result = jobs_collection.update_one({"_id": ObjectId(job_id)}, {"$set": updated_job})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job updated successfully"}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    result = jobs_collection.delete_one({"_id": ObjectId(job_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}






# Caching job embeddings
@lru_cache(maxsize=1000)
def get_job_embedding(job_text: str):
    return model.encode(job_text, convert_to_tensor=True).tolist()

def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file (in-memory processing)"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = " ".join(page.get_text("text") for page in doc)
    
    return text
# def extract_text_with_ocr(file):
#     """Extract text from scanned PDF using OCR (Tesseract + pdf2image)"""
#     images = convert_from_bytes(file.read())  # Convert PDF pages to images
#     text = " ".join(pytesseract.image_to_string(img) for img in images)
#     return text

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])



def extract_file_id_from_url(drive_url):
    """ Extracts Google Drive file ID from the provided URL """
    match = re.search(r"[-\w]{25,}", drive_url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid Google Drive link")
    return match.group(0)

def download_cv_from_drive_public(drive_url):
    """ Downloads a PDF from a public Google Drive link """
    file_id = extract_file_id_from_url(drive_url)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(download_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download CV. Ensure the link is public.")

        file_stream = BytesIO(response.content)
        return extract_text_from_pdf(file_stream)
    except Exception as e:
        logger.error(f"Error downloading CV: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process CV from Google Drive")


def calculate_similarity(cv_text, job_text):
   # similarity between CV and job description
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=50000,
        use_idf=True
    )
    vectors = vectorizer.fit_transform([cv_text, job_text])
    return cosine_similarity(vectors[0], vectors[1])[0][0] * 100  # Return percentage match
  


@app.post("/ats/")
async def ats_system(
    job_id: str = Form(...),
    cv_drive_link: str = Form(...)
):
# async def ats_system(job_id: str = Form(...), cv_cloudinary_url: str = Form(...)):
    """ Matches a CV against a job description using TF-IDF """
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_description = job.get("description", "")
    cv_text = download_cv_from_drive_public(cv_drive_link)
    

    
    #Preprocess texts
    cv_processed = preprocess_text(cv_text)
    job_processed = preprocess_text(job_description)

    # Calculate similarity
    similarity_score = calculate_similarity(cv_processed, job_processed)

    logger.info(f"ATS Match Score for Job {job_id}: {similarity_score:.3f}%")
    return {"match_percentage": round(similarity_score, 2), "message": "Higher score means a better match!"}

@app.get("/ats/")
async def get_ats_info():
    """ Health check endpoint """
    return {"message": "ATS API is running! Use POST /ats/ to check CV-job match."}

def download_cv_from_drive(file_id):
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    SERVICE_ACCOUNT_FILE = "path/to/service_account.json"
    
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds)
    request = service.files().get_media(fileId=file_id)
    file_stream = BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    done = False
    try:
        while not done:
            status, done = downloader.next_chunk()
        file_stream.seek(0)
        return extract_text_from_pdf(file_stream)
    except Exception as e:
        logger.error(f"Error downloading CV: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download CV from Google Drive")




































































































































































































# @app.post("/ats/")
# async def ats_system(cv_file: UploadFile = File(...), job_description: str = Form(...)):
#     """Process CV and job description and return match percentage"""
#     cv_text = extract_text_from_pdf(cv_file)
#     cv_processed = preprocess_text(cv_text)
#     job_processed = preprocess_text(job_description)
#     similarity_score = calculate_similarity(cv_processed, job_processed)

#     return {
#         "match_percentage": round(similarity_score, 2),
#         "message": "Higher score means a better match!"
#     }
# # el part da mahtag yt3adel 
# def download_cv_from_drive(file_id):
#     SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
#     SERVICE_ACCOUNT_FILE = "path/to/service_account.json"
    
#     creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#     service = build("drive", "v3", credentials=creds)
#     request = service.files().get_media(fileId=file_id)
#     file_stream = BytesIO()
#     downloader = MediaIoBaseDownload(file_stream, request)
#     done = False
#     while not done:
#         status, done = downloader.next_chunk()
    
#     file_stream.seek(0)
#     return extract_text_from_pdf(file_stream)



# @app.post("/ats/")
# async def ats_system(job_id: str = Form(...), cv_drive_link: str = Form(...)):
#     job = jobs_collection.find_one({"_id": ObjectId(job_id)})
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job_description = job.get("description", "")
#     match = re.search(r"[-\w]{25,}", cv_drive_link)
#     if not match:
#         raise HTTPException(status_code=400, detail="Invalid Google Drive link")
    
#     cv_file_id = match.group(0)
#     cv_text = download_cv_from_drive(cv_file_id)
#     cv_processed = preprocess_text(cv_text)
#     job_processed = preprocess_text(job_description)
#     similarity_score = calculate_similarity(cv_processed, job_processed)

#     return {"match_percentage": round(similarity_score, 2), "message": "Higher score means a better match!"}

# @app.get("/ats/")
# async def get_ats_info():
#     return {"message": "ATS API is running! Use POST /ats/ to check CV-job match."}
