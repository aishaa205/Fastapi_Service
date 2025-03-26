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
import requests
from io import BytesIO
from googleapiclient.discovery import build
from loguru import logger
from functools import lru_cache
import requests  # To download Cloudinary files
from pdf2image import convert_from_bytes  # Convert scanned PDF to images


from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer()



DJANGO_AUTH_URL = "http://localhost:8000/api/token/verify/"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    response = requests.post(DJANGO_AUTH_URL, json={"token": token})
    if response.status_code != 200:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    return response.json()  # or return the user details


app = FastAPI()
#load  ats model

model_1 = SentenceTransformer("all-MiniLM-L6-v2")


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





CLOUDINARY_CLOUD_NAME ="dkvyfbtdl"
def extract_text_from_pdf_cloud(public_id: str):
    """Downloads and extracts text from a Cloudinary-hosted PDF file using its public_id."""
    try:
        pdf_url = f"https://res.cloudinary.com/{CLOUDINARY_CLOUD_NAME}/raw/upload/{public_id}.pdf"
        print("Downloading PDF from:", pdf_url)

        response = requests.get(pdf_url)
        response.raise_for_status()  # Ensure the request was successful
        print("PDF downloaded successfully.")
       
         #Use BytesIO to open PDF from memory
        pdf_stream = BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()

        return text.strip() if text else "No text found in PDF."
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {e}")   


@app.get("/recom/")
async def recommend_jobs(user_skills: str, page: int = 1, page_size: int = 5):
    """AI-powered job recommendations with pagination"""
    print("user_skills",user_skills)
    print("page",page)
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be 1 or higher")
    
    
    extracted_text = extract_text_from_pdf_cloud(user_skills)
    
    query_vector = get_embedding(extracted_text)
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
    id: str
    title: str
    description: str

    @validator("title", "description")
    def must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Field cannot be empty")
        return value

    @validator("id")
    def must_have_id(cls, value):
        if not value:
            raise ValueError("Must include job ID")
        return value




# @app.post("/jobs", dependencies=[Depends(verify_token)])
@app.post("/jobs")
async def create_job(job: Job):
    print("job",job)
    job_data = job.dict()
    print("job_data",job_data)
    job_data["combined_embedding"] = get_embedding(job_data["description"] + " " + " ".join(job_data["title"]))
    inserted_job = jobs_collection.insert_one(job_data)
    print("inserted_job",inserted_job)
    return {"id": str(inserted_job.inserted_id), "message": "Job created successfully"}


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
async def update_job(job_id: str, job: Job):
    updated_job = job.dict()
    updated_job["combined_embedding"] = get_embedding(updated_job["description"] + " " + " ".join(updated_job["title"]))
    
    existing_job = jobs_collection.find_one({"id": job_id})
    print(existing_job)
    if not existing_job:
            raise HTTPException(status_code=404, detail="Job not found")
    result = jobs_collection.update_one({"id": job_id}, {"$set": updated_job})
    print (result)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Job updated successfully"}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    existing_job = jobs_collection.find_one({"id": job_id})
    print("existing_job",existing_job)
    if not existing_job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = jobs_collection.delete_one({"id": job_id})
    print(result)
    print(result)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}



def get_embedding_ats(text):
    with torch.no_grad():
        return model_1.encode(text, convert_to_tensor=True).tolist()


# def extract_text_from_pdf(file):
#     """Extract text from an uploaded PDF file (in-memory processing)"""
#     doc = fitz.open(stream=file.read(), filetype="pdf")
#     text = " ".join(page.get_text("text") for page in doc)
#     return text

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()


def calculate_embedding_similarity(cv_text, job_text):
    """Calculate similarity using sentence embeddings"""
    cv_embedding = model_1.encode(cv_text, convert_to_tensor=True)
    job_embedding = model_1.encode(job_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(cv_embedding, job_embedding).item()
    return similarity_score * 100  # Convert to percentage




@app.post("/ats/{job_id}/")
async def ats_system(
    job_id: str = Form(...),
    cv_drive_link: str = Form(...)
):
    """Matches a CV against a job description using sentence embeddings"""
    #de kda mn mongodb not django 
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_description = job.get("description", "")
    cv_text = extract_text_from_pdf_cloud(cv_drive_link)
    
    # Preprocess texts
    cv_processed = preprocess_text(cv_text)
    job_processed = preprocess_text(job_description)

    # Calculate similarity
    similarity_score = calculate_embedding_similarity(cv_processed, job_processed)

    logger.info(f"ATS Match Score for Job {job_id}: {similarity_score:.3f}%")
    return {"match_percentage": round(similarity_score, 2), "message": "Higher score means a better match!"}

@app.get("/ats/")
async def get_ats_info():
    """Health check endpoint"""
    return {"message": "ATS API is running! Use POST /ats/ to check CV-job match."}






















#matnse4 split to 2 different files + add test cases file for each 


# def extract_text_from_pdf(file):
#     """Extract text from an uploaded PDF file (in-memory processing)"""
#     doc = fitz.open(stream=file.read(), filetype="pdf")
#     text = " ".join(page.get_text("text") for page in doc)
    
#     return text
# # def extract_text_with_ocr(file):
# #     """Extract text from scanned PDF using OCR (Tesseract + pdf2image)"""
# #     images = convert_from_bytes(file.read())  # Convert PDF pages to images
# #     text = " ".join(pytesseract.image_to_string(img) for img in images)
# #     return text

# def preprocess_text(text):
#     text = re.sub(r'\W+', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     doc = nlp(text.lower())
#     return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])



# def extract_file_id_from_url(drive_url):
#     """ Extracts Google Drive file ID from the provided URL """
#     match = re.search(r"[-\w]{25,}", drive_url)
#     if not match:
#         raise HTTPException(status_code=400, detail="Invalid Google Drive link")
#     return match.group(0)

# def download_cv_from_drive_public(drive_url):
#     """ Downloads a PDF from a public Google Drive link """
#     file_id = extract_file_id_from_url(drive_url)
#     download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
#     try:
#         response = requests.get(download_url)
#         if response.status_code != 200:
#             raise HTTPException(status_code=400, detail="Failed to download CV. Ensure the link is public.")

#         file_stream = BytesIO(response.content)
#         return extract_text_from_pdf(file_stream)
#     except Exception as e:
#         logger.error(f"Error downloading CV: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to process CV from Google Drive")


# def calculate_similarity(cv_text, job_text):
#    # similarity between CV and job description
#     vectorizer = TfidfVectorizer(
#         stop_words='english',
#         ngram_range=(1,2),
#         max_features=50000,
#         use_idf=True
#     )
#     vectors = vectorizer.fit_transform([cv_text, job_text])
#     return cosine_similarity(vectors[0], vectors[1])[0][0] * 100  # Return percentage match
  


# @app.post("/ats/")
# async def ats_system(
#     job_id: str = Form(...),
#     cv_drive_link: str = Form(...)
# ):
# # async def ats_system(job_id: str = Form(...), cv_cloudinary_url: str = Form(...)):
#     """ Matches a CV against a job description using TF-IDF """
#     job = jobs_collection.find_one({"_id": ObjectId(job_id)})
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job_description = job.get("description", "")
#     cv_text = download_cv_from_drive_public(cv_drive_link)
    

    
#     #Preprocess texts
#     cv_processed = preprocess_text(cv_text)
#     job_processed = preprocess_text(job_description)

#     # Calculate similarity
#     similarity_score = calculate_similarity(cv_processed, job_processed)

#     logger.info(f"ATS Match Score for Job {job_id}: {similarity_score:.3f}%")
#     return {"match_percentage": round(similarity_score, 2), "message": "Higher score means a better match!"}

# @app.get("/ats/")
# async def get_ats_info():
#     """ Health check endpoint """
#     return {"message": "ATS API is running! Use POST /ats/ to check CV-job match."}

# def download_cv_from_drive(file_id):
#     SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
#     SERVICE_ACCOUNT_FILE = "path/to/service_account.json"
    
#     creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#     service = build("drive", "v3", credentials=creds)
#     request = service.files().get_media(fileId=file_id)
#     file_stream = BytesIO()
#     downloader = MediaIoBaseDownload(file_stream, request)
#     done = False
#     try:
#         while not done:
#             status, done = downloader.next_chunk()
#         file_stream.seek(0)
#         return extract_text_from_pdf(file_stream)
#     except Exception as e:
#         logger.error(f"Error downloading CV: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to download CV from Google Drive")



# from fastapi import FastAPI, Header, HTTPException
# from JobRecommender import router as job_recommender_router
# from ats_system import router as ats_router
# from loguru import logger

# app = FastAPI()

# # Security: API Key Middleware
# def verify_api_key(api_key: str = Header(...)):
#     if api_key != "your_secure_api_key":
#         raise HTTPException(status_code=403, detail="Invalid API Key")
#     return api_key

# @app.on_event("startup")
# async def startup_event():
#     logger.info("Server started successfully!")

# # Include routers
# app.include_router(job_recommender_router, prefix="/jobs", tags=["Job Recommender"])
# app.include_router(ats_router, prefix="/ats", tags=["ATS System"])

# @app.get("/")
# async def root():
#     return {"message": "Job Recommender & ATS System API Running"}










































