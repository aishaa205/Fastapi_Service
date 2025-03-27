import sys
sys.path.append("..") 
from sentence_transformers import SentenceTransformer , util
from fastapi import FastAPI , HTTPException ,Query, UploadFile, File, Form , Header, Depends
from database import jobs_collection ,users_collection
from pydantic import BaseModel,validator
from bson import ObjectId
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import fitz  # PyMuPDF
import requests
from io import BytesIO
from googleapiclient.discovery import build
from loguru import logger
from functools import lru_cache
import requests  # To download Cloudinary files
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()



DJANGO_AUTH_URL = "http://localhost:8000/api/token/verify/"

# async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     token = credentials.credentials
#     response = requests.post(DJANGO_AUTH_URL, json={"token": token})
#     if response.status_code != 200:
#         raise HTTPException(status_code=403, detail="Invalid authentication token")
#     return response.json()  # or return the user details


app = FastAPI()
#load  ats model

model_1 = SentenceTransformer("all-MiniLM-L6-v2")


#load recommender model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

@app.on_event("startup")
async def startup_event():
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
       
        pdf_stream = BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()

        return text.strip() if text else "No text found in PDF."
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {e}")   




def get_embedding(text):
    with torch.no_grad():
        return model.encode(text, convert_to_tensor=True).tolist()


@app.get("/recom/")
async def recommend_jobs(user_id: str, cv_url: str, page: int = 1, page_size: int = 5):
    print("user_id",user_id)
    print("cv_url",cv_url)
    print("page",page)
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be 1 or higher")

    user_data = users_collection.find_one({"user_id": user_id})
    if user_data:
        stored_cv_url = user_data.get("cv_url")
        if stored_cv_url == cv_url:
            print("Using stored embedding (CV unchanged)")
            query_vector = user_data["embedding"]
        else:
            print("CV updated, generating new embedding")
            extracted_text = extract_text_from_pdf_cloud(cv_url)
            query_vector = get_embedding(extracted_text)
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"embedding": query_vector, "cv_url": cv_url}}  
            )
    else:
        print("New user, extracting CV text")
        extracted_text = extract_text_from_pdf_cloud(cv_url)
        query_vector = get_embedding(extracted_text)
        users_collection.insert_one(
            {"user_id": user_id, "embedding": query_vector, "cv_url": cv_url} 
        )
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
    id: int
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

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()


def calculate_embedding_similarity(cv_text, job_text):
    cv_embedding = model_1.encode(cv_text, convert_to_tensor=True)
    job_embedding = model_1.encode(job_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(cv_embedding, job_embedding).item()
    return similarity_score * 100  

@app.post("/ats/{user_id}/{job_id}/")
async def ats_system(user_id: str, job_id: str, cv_url: str):
    """Matches a CV against a job description using sentence embeddings."""
    
    # Check if job exists
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        logger.info(f"Job {job_id} not found. Creating new job entry.")
        new_job = {"_id": ObjectId(job_id), "description": "", "title": "Unknown", "combined_embedding": None}
        jobs_collection.insert_one(new_job)
        job = new_job  # Use newly created job entry
    
    job_description = job.get("description", "")

    # Check if user exists in DB
    user_data = users_collection.find_one({"user_id": user_id})
    
    if user_data:
        stored_cv_url = user_data.get("cv_url")
        if stored_cv_url == cv_url:
            print("Using stored embedding (CV unchanged)")
            cv_embedding = user_data["embedding"]
        else:
            print("CV updated, generating new embedding")
            extracted_text = extract_text_from_pdf_cloud(cv_url)
            cv_embedding = get_embedding_ats(extracted_text)
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"embedding": cv_embedding, "cv_url": cv_url}}
            )
    else:
        print("New user, extracting CV text")
        extracted_text = extract_text_from_pdf_cloud(cv_url)
        cv_embedding = get_embedding_ats(extracted_text)
        users_collection.insert_one(
            {"user_id": user_id, "embedding": cv_embedding, "cv_url": cv_url}
        )

    # Preprocess texts
    job_processed = preprocess_text(job_description)

    # Calculate similarity
    similarity_score = calculate_embedding_similarity(cv_embedding, job_processed)

    logger.info(f"ATS Match Score for Job {job_id}: {similarity_score:.3f}%")
    return {"match_percentage": round(similarity_score, 2), "message": "Higher score means a better match!"}


# @app.post("/ats/{job_id}/")
# async def ats_system(
#     job_id: str ,
#     cv_drive_link: str = Form(...)
# ):
#     """Matches a CV against a job description using sentence embeddings"""
#     #de kda mn mongodb not django 
#     job = jobs_collection.find_one({"_id": ObjectId(job_id)})
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job_description = job.get("description", "")
#     cv_text = extract_text_from_pdf_cloud(cv_drive_link)
    
#     # Preprocess texts
#     cv_processed = preprocess_text(cv_text)
#     job_processed = preprocess_text(job_description)

#     # Calculate similarity
#     similarity_score = calculate_embedding_similarity(cv_processed, job_processed)

#     logger.info(f"ATS Match Score for Job {job_id}: {similarity_score:.3f}%")
#     return {"match_percentage": round(similarity_score, 2), "message": "Higher score means a better match!"}























#matnse4 split to 2 different files + add test cases file for each 









































