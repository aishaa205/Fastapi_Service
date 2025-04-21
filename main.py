import sys
sys.path.append("..") 
from sentence_transformers import SentenceTransformer , util
from fastapi import FastAPI , HTTPException ,Query, UploadFile, File, Form , Header, Depends
from database import jobs_collection ,users_collection
from pydantic import BaseModel,validator
# from bson import ObjectId
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
from fastapi import Request
import json
import os

from interview import router as interview_router





app = FastAPI()
#load  ats model

model_1 = SentenceTransformer("all-MiniLM-L6-v2")


app.include_router(interview_router, prefix="/api/v1")


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
    
def format_cv_url(cv_url):
    if cv_url.startswith("http"):
        return cv_url.split("/")[-1]
    return cv_url


@app.get("/recom/")
async def recommend_jobs(user_id: str, cv_url: str, page: int = 1, page_size: int = 6):
    print("user_id",user_id)
    cv_url = format_cv_url(cv_url)
    print("cv_url",cv_url)
    print("page",page)
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be 1 or higher")

    user_data = users_collection.find_one({"user_id": user_id})
    page_count= jobs_collection.count_documents({}) // page_size + 1
    if page*page_size > 100:
        raise HTTPException(status_code=400, detail="Max recommendations is 100")
    if page > page_count/page_size:
        raise HTTPException(status_code=400, detail="Page number exceeds available pages")
    if not cv_url:
        raise HTTPException(status_code=400, detail="CV URL is required")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
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
                "limit": 100,
                "metric": "cosine"
            }
        },
        {
            "$project": {
                "_id": 0,
                "id": 1,
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
        "total_results": page_count if page_count < 100 else 100,
        "recommendations": results
    }

class Job(BaseModel):
    id: int
    title: str
    description: str
    location: str
    experince: str
    status: str
    type_of_job: str
    company: int
    company_name: str
    company_logo: Optional[str] = None
    {'id': 31, 'title': 'Backend Engineer', 'description': 'Django and FastAPI experience required', 'location': 'Remote', 'status': 'open'
     , 'type_of_job': 'Full-time', 'experince': 'Mid-level', 'company': 8, 'company_name': 'Aisha Amr', 'company_logo': None}
    

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
async def create_job(job: Job, request: Request):
    print("job",job)
    job_data = job.dict()
    print("job_data",job_data)
    job_data["combined_embedding"] = get_embedding(job_data["description"] + " " + " ".join(job_data["title"]))
    inserted_job = jobs_collection.insert_one(job_data)
    print("inserted_job",inserted_job)
    return {"id": str(inserted_job.inserted_id),
            "message": "Job created successfully",
            #"mongodb_id": str(inserted_job.inserted_id)
            }


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
    updated_job["combined_embedding"] = get_embedding(updated_job["description"] + " " + " ".join(updated_job["title"]))
    print(job_id)
    existing_job = jobs_collection.find_one({"id": job_id})
    print(existing_job)
    if not existing_job:
            result = jobs_collection.insert_one(updated_job)
            return {"message": "Job created successfully", "id": str(result.inserted_id)}
    result = jobs_collection.update_one({"id": job_id}, {"$set": updated_job})
    print (result)
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


class ATSRequest(BaseModel):
    cv_url: str
    job_id: int

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
async def ats_system(user_id: str, job_id: int, request: ATSRequest):
    
    print(f"Received request for user_id={user_id}, job_id={job_id}")

    try:
        job_object_id = job_id
    except Exception as e:
        print(f"Invalid job_id: {job_id} - Error: {e}")
        raise HTTPException(status_code=400, detail="Invalid job ID")


    job = jobs_collection.find_one({"id": job_object_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_embedding = job.get("combined_embedding", None)

    if not job_embedding:
        raise HTTPException(status_code=500, detail="Job embedding missing")

    # Get CV URL
    cv_url = format_cv_url(request.cv_url)
    print(f"cv_url={cv_url}")

    
    user_data = users_collection.find_one({"user_id": user_id})

    if user_data and user_data.get("cv_url") == cv_url:
        print("Using stored embedding (CV unchanged)")
        cv_embedding = user_data["embedding"]
    else:
        try:
            extracted_text = extract_text_from_pdf_cloud(cv_url)
            print(f"Extracted text length: {len(extracted_text)}")
        except Exception as e:
            print(f"Error extracting text: {e}")
            raise HTTPException(status_code=500, detail="CV extraction failed")

        try:
            cv_embedding = get_embedding(extracted_text)
            print(f"Generated embedding length: {len(cv_embedding)}")
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="Embedding generation failed")

        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"embedding": cv_embedding, "cv_url": cv_url}},
            upsert=True
        )

    
    if not cv_embedding or not job_embedding:
        raise HTTPException(status_code=500, detail="Embedding computation failed")
    
    try:
        similarity_score = util.pytorch_cos_sim(
            torch.tensor(cv_embedding), torch.tensor(job_embedding)
        ).item()
    except Exception as e:
        print(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail="Similarity computation failed")

    return {"match_percentage": round(similarity_score * 100, 2), "message": "Higher score means a better match!"}
    
    
    
    
#matnse4 split to 2 different files + add test cases file for each 



@app.post("/analyze-interview/")
async def analyze_interview_endpoint(
    video: UploadFile = File(...),
    question: str = Form(...),
    job_description: str = Form(...)
):
    """
    Enhanced interview analysis endpoint that provides:
    - Content relevance to job description
    - Language proficiency assessment
    - Presentation skills evaluation
    - Voice tone analysis
    """
    try:
        # Save video temporarily
        temp_video_path = await save_temp_file(video)
        
        # Perform analysis
        results = await analyzer.analyze_interview(
            video_path=temp_video_path,
            question=question,
            job_description=job_description
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)