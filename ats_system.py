import torch
import re
import requests
from fastapi import APIRouter, HTTPException, Form
from sentence_transformers import SentenceTransformer, util
from database import jobs_collection
from bson import ObjectId
from io import BytesIO
import fitz  # PyMuPDF

router = APIRouter()

# Load ATS model
model_ats = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding_ats(text):
    with torch.no_grad():
        return model_ats.encode(text, convert_to_tensor=True).tolist()

def extract_text_from_pdf(file):
    """Extract text from PDF"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = " ".join(page.get_text("text") for page in doc)
    return text

def preprocess_text(text):
    """Clean text"""
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_file_id_from_url(drive_url):
    """Extracts Google Drive file ID"""
    match = re.search(r"[-\w]{25,}", drive_url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid Google Drive link")
    return match.group(0)

def download_cv_from_drive(drive_url):
    """Download CV from Google Drive"""
    file_id = extract_file_id_from_url(drive_url)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download CV. Ensure the link is public.")
    file_stream = BytesIO(response.content)
    return extract_text_from_pdf(file_stream)

def calculate_embedding_similarity(cv_text, job_text):
    """Calculate similarity between CV and job description"""
    cv_embedding = model_ats.encode(cv_text, convert_to_tensor=True)
    job_embedding = model_ats.encode(job_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(cv_embedding, job_embedding).item()
    return similarity_score * 100

@router.post("/ats/")
async def ats_system(
    job_id: str = Form(...),
    cv_drive_link: str = Form(...)
):
    """Matches a CV against a job description"""
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_description = job.get("description", "")
    cv_text = download_cv_from_drive(cv_drive_link)

    # Preprocess texts
    cv_processed = preprocess_text(cv_text)
    job_processed = preprocess_text(job_description)

    # Calculate similarity
    similarity_score = calculate_embedding_similarity(cv_processed, job_processed)

    return {"match_percentage": round(similarity_score, 2), "message": "Higher score means a better match!"}
