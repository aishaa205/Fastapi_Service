import sys
sys.path.append("..") 


from sentence_transformers import SentenceTransformer
from fastapi import FastAPI , HTTPException ,Query, BackgroundTasks
from database import jobs_collection 
from pydantic import BaseModel,validator
from bson import ObjectId
import torch
import joblib
import os


app = FastAPI()


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

@app.on_event("startup")
async def startup_event():
    #Preload model before serving requests
    global model
    model = load_model()

def get_embedding(text):
    with torch.no_grad():
        return model.encode(text, convert_to_tensor=True).tolist()


async def recommend_jobs_background(user_skills: str):
    query_vector = get_embedding(user_skills)
    return {"embedding": query_vector}  

# @app.get("/recom/")
# async def recommend_jobs(user_skills: str, background_tasks: BackgroundTasks):
    # job recommendation m4 hytzhar
    # background_tasks.add_task(recommend_jobs_background, user_skills)
    # return {"message": "Processing recommendation in the background"}


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

