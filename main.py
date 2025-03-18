import sys
sys.path.append("..") 

from fastapi import FastAPI , HTTPException ,Query
from database import jobs_collection 
import joblib
from sentence_transformers import SentenceTransformer
import torch


app= FastAPI()



# Load Sentence Transformer Model with GPU Support (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

# Function to encode text efficiently
def get_embedding(text):
    with torch.no_grad():  # Disable gradient tracking for efficiency
        return model.encode(text, convert_to_tensor=True).tolist()

@app.get("/recom/")
async def recommend_jobs(job_title: str = Query(...), user_skills: str = Query(...)):#user_skills: str, limit: int = 10):
    """AI-powered job recommendations using MongoDB Atlas vector search."""

    query_vector = get_embedding(user_skills)

    # MongoDB Atlas vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector",  
                "path": "combined_embedding",
                "queryVector": query_vector,
                "numCandidates": 500,  
                "limit": 10,
                "metric": "cosine"
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude MongoDB ID
                "title": 1,
                "description": 1,
                "score": {"$meta": "vectorSearchScore"}  # Include similarity score
            }
        }
    ]

    results = list(jobs_collection.aggregate(pipeline))

    if not results:
        raise HTTPException(status_code=404, detail="No matching jobs found")

    return {"recommendations": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)








# model = joblib.load("models/job_recommendation.pkl")
# @app.get("/recommend/{user_id}")
# async def recommend_jobs(user_id: int):
#     jobs = list(jobs_collection.find({}, {"_id": 0}))  # Fetch jobs from MongoDB
#     recommended_jobs = jobs[:10]  # Mock recommendation logic
#     return {"user_id": user_id, "recommendations": recommended_jobs}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)

