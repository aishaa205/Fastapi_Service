import torch
from fastapi import APIRouter, HTTPException, Query
from sentence_transformers import SentenceTransformer
from database import jobs_collection
from bson import ObjectId
from functools import lru_cache

router = APIRouter()    

def load_recommender_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

model = load_recommender_model()



def get_embedding(text):
    with torch.no_grad():
        return model.encode(text, convert_to_tensor=True).tolist()

@router.get("/recom/")
async def recommend_jobs(user_skills: str, page: int = 1, page_size: int = 5):
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
        {"$project": {"_id": 0, "title": 1, "description": 1, "score": {"$meta": "vectorSearchScore"}}},
        {"$skip": skip},
        {"$limit": page_size}
    ]

    results = list(jobs_collection.aggregate(pipeline))

    if not results:
        raise HTTPException(status_code=404, detail="No matching jobs found")

    return {"page": page, "page_size": page_size, "total_results": len(results), "recommendations": results}

@router.post("/jobs/")
async def create_job(job: dict):
    """Create a new job"""
    job["combined_embedding"] = get_embedding(job["description"] + " " + " ".join(job["skills_required"]))
    inserted_job = jobs_collection.insert_one(job)
    return {"id": str(inserted_job.inserted_id), "message": "Job created successfully"}

@router.get("/jobs/")
async def get_jobs(limit: int = 10, skip: int = 0):
    """Get all jobs with pagination"""
    jobs = list(jobs_collection.find({}, {"_id": 1, "title": 1, "description": 1, "skills_required": 1}).skip(skip).limit(limit))
    print("Jobs found in database:", jobs) 
    for job in jobs:
        job["_id"] = str(job["_id"])
    return {"jobs": jobs}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        job = jobs_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        job["_id"] = str(job["_id"])
        return job
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Job ID: {str(e)}")