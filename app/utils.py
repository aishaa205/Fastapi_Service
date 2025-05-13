import os
import requests
import tempfile
from fastapi import HTTPException

def save_temp_file(file_content: bytes, file_extension: str = '.webm') -> str:
    """Save uploaded file content to a temporary file."""
    try:
        temp_dir = 'temp/'
        os.makedirs(temp_dir, exist_ok=True)  # Ensure the temp directory exists
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir=temp_dir) as temp_file:
            temp_file.write(file_content)
            print("temp",temp_file)
            return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")


def download_video_from_url(url: str) -> str:
    """Download video from URL and save it temporarily."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        print ("response video",response)
        video_path = save_temp_file(response.content)
        return video_path
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")
