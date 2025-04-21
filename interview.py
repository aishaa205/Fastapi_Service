# import os
# import tempfile
# import subprocess
# import cv2
# import numpy as np
# import speech_recognition as sr
# from deepface import DeepFace
# from sentence_transformers import SentenceTransformer, util
# import torch
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# from fastapi import FastAPI, APIRouter, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Annotated
# import logging
# import traceback
# from pydub import AudioSegment
# from pydub.silence import detect_silence
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
# import noisereduce as nr
# import httpx


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)



# # Initialize models
# model = SentenceTransformer('all-MiniLM-L6-v2')
# resnet_model = ResNet50(weights='imagenet')
# router = APIRouter()

# # ImageNet classes for attire analysis
# imagenet_classes = [
#     'suit', 'tie', 'shirt', 'business suit', 'bow tie', 'jeans',
#     'sweatshirt', 'T-shirt', 'sweat pants', 'pajamas', 'lab coat'
# ]

# class InterviewAnalyzer:
#     async def analyze_interview(self, video_path: str, question: str, job_description: str):
#         """Main analysis entry point"""
#         try:
#             # Validate video file
#             if not os.path.isfile(video_path):
#                 raise ValueError("Video file does not exist")
            
#             # Basic video file validation
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 raise ValueError("Unable to open video file")
#             cap.release()

#        # try:
#             # Analyze answer content
#             transcript = self.transcribe_video(video_path)
#             answer_score = self.analyze_answer(transcript, question, job_description)
            
#             # Analyze pronunciation and grammar
#             pronunciation_score = self.analyze_pronunciation(video_path)
            
#             # Analyze eye contact
#             eye_contact_score = self.analyze_eye_contact(video_path)
            
#             # Analyze attire
#             attire_score = self.analyze_attire(video_path)
            
#             # Calculate total score
#             total_score = (
#                 answer_score * 0.4 +
#                 pronunciation_score * 0.3 +
#                 eye_contact_score * 0.15 +
#                 attire_score * 0.15
#             )
            
#             return {
#                 "answer_score": round(answer_score, 2),
#                 "pronunciation_score": round(pronunciation_score, 2),
#                 "eye_contact_score": round(eye_contact_score, 2),
#                 "attire_score": round(attire_score, 2),
#                 "total_score": round(total_score, 2),
#                 "transcript": transcript
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

#     def transcribe_video(self, video_path: str) -> str:
#         """Convert speech to text"""
#         r = sr.Recognizer()
#         audio_path = self.extract_audio(video_path)
        
#         try:
#             # Check audio quality before processing
#             audio = AudioSegment.from_wav(audio_path)
            
#             # 1. Noise reduction
#             audio_array = np.array(audio.get_array_of_samples())
#             reduced_noise = nr.reduce_noise(
#                 y=audio_array,
#                 sr=audio.frame_rate,
#                 stationary=True
#             )
#             cleaned_audio = AudioSegment(
#                 reduced_noise.tobytes(),
#                 frame_rate=audio.frame_rate,
#                 sample_width=audio.sample_width,
#                 channels=audio.channels
#             )
#             # 2. Silence detection
#             non_silent = detect_nonsilent(
#                 cleaned_audio,
#                 min_silence_len=500,
#                 silence_thresh=-40
#             )
            
#             if not non_silent:
#                 raise HTTPException(400, "No speech detected in audio")
            
            
#             # 3. Audio normalization
#             normalized = cleaned_audio.normalize()

#             # Save processed audio
#             processed_path = audio_path.replace(".wav", "_processed.wav")
#             normalized.export(processed_path, format="wav")

#             # Transcription with adjusted parameters
#             with sr.AudioFile(processed_path) as source:
#                 audio_data = r.record(source)
#                 return r.recognize_google(
#                     audio_data,
#                     language="en-US",
#                     show_all=False,
#                     pfilter=1  # Enable profanity filter
#                 )
#         except sr.UnknownValueError:
#                 raise HTTPException(400, "Could not understand audio (low quality)")
#         except sr.RequestError as e:
#                 raise HTTPException(500, f"Speech service error: {str(e)}")
#         finally:
#                 for f in [audio_path, processed_path]:
#                     if os.path.exists(f):
#                         os.remove(f)        
            
#         #     # Check duration
#         #     if len(audio) < 1000:  # 1 second minimum
#         #         raise HTTPException(status_code=400, detail="Audio too short")
#         #     # Transcribe audio
#         #     with sr.AudioFile(audio_path) as source:
#         #         audio = r.record(source)
#         #     return r.recognize_google(audio)
#         # except sr.UnknownValueError:
#         #     raise HTTPException(status_code=400, detail="Could not understand audio")
#         # except sr.RequestError as e:
#         #     raise HTTPException(status_code=500, detail=f"Speech recognition error: {e}")
#         # finally:
#         #     if os.path.exists(audio_path):
#         #         os.remove(audio_path)

#     def analyze_answer(self, transcript: str, question: str, job_desc: str) -> float:
#         """Semantic analysis of answer content"""
#         expected_answer = self.generate_expected_answer(question, job_desc)
#         transcript_embedding = model.encode(transcript)
#         expected_embedding = model.encode(expected_answer)
#         return util.pytorch_cos_sim(transcript_embedding, expected_embedding).item() * 100

#     def generate_expected_answer(self, question: str, job_desc: str) -> str:
#         """Generate ideal answer template"""
#         return f"A strong answer would demonstrate experience with {job_desc} " \
#                f"and specifically address {question}. Ideal responses should include " \
#                "relevant technical skills, problem-solving examples, and team collaboration experiences."

#     def analyze_pronunciation(self, video_path: str) -> float:
#         """Pronunciation and grammar analysis"""
#         try:
#             audio_path = self.extract_audio(video_path)
#             r = sr.Recognizer()
            
#             with sr.AudioFile(audio_path) as source:
#                 audio = r.record(source)
#                 text = r.recognize_google(audio)
                
#             # Grammar check
#             sentence_count = len(text.split('.'))
#             word_count = len(text.split())
#             complexity_score = min((word_count / sentence_count) / 10 * 100, 100)
            
#             return complexity_score * 0.6 + 40  # Base score + complexity
            
#         except Exception as e:
#             return 65  # Minimum score if analysis fails
#         finally:
#             if os.path.exists(audio_path):
#                 os.remove(audio_path)

#    # Updated eye contact analysis
#     def analyze_eye_contact(self, video_path: str) -> float:
#         cap = cv2.VideoCapture(video_path)
#         total_frames = 0
#         looking_frames = 0
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
#             if len(faces) > 0:
#                 # Calculate eye region
#                 (x,y,w,h) = faces[0]
#                 eye_region = frame[y:y+h//2, x:x+w]
                
#                 # Detect pupils using HoughCircles
#                 circles = cv2.HoughCircles(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY),
#                     cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, 
#                     minRadius=0, maxRadius=0)
                
#                 if circles is not None:
#                     looking_frames += 1
                    
#             total_frames += 1
            
#         return (looking_frames / total_frames) * 100 if total_frames > 0 else 0

#     def extract_audio(self, video_path: str) -> str:
#         """Extract audio using ffmpeg"""
#         try:
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
#                 command = [
#                     'ffmpeg',
#                     '-y',
#                     '-i', video_path,
#                     '-vn', '-acodec', 'pcm_s16le',
#                     '-acodec', 'pcm_s16le',
#                     '-ar', '16000', '-ac', '1',
#                     '-loglevel', 'debug',  # Add detailed logging
#                     temp_audio.name
#                 ]
#                 logger.info(f"Audio stored at: {temp_audio.name}")
#                 result = subprocess.run(
#                     command,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True
#                 )
                
#                 if result.returncode != 0:
#                     logger.error(f"FFmpeg Error: {result.stderr}")
#                     raise RuntimeError(f"Audio extraction failed: {result.stderr}")
                    
#                 # Verify output file
#                 if os.path.getsize(temp_audio.name) == 0:
#                     raise RuntimeError("Empty audio file generated")
            
#                 return temp_audio.name
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Audio extraction failed: {str(e)}")


#     # Improved attire analysis
#     def analyze_attire(self, video_path: str) -> float:
#         professional_classes = {
#             'suit': 0.7, 'tie': 0.6, 'dress_shirt': 0.8, 
#             'business_suit': 0.9, 'formal_wear': 0.85
#         }
        
#         cap = cv2.VideoCapture(video_path)
#         total_score = 0
#         sampled_frames = 0
        
#         while cap.isOpened() and sampled_frames < 30:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             if sampled_frames % 10 == 0:  # Sample every 10th frame
#                 img = Image.fromarray(frame).resize((224, 224))
#                 x = image.img_to_array(img)
#                 x = preprocess_input(x)
#                 x = np.expand_dims(x, axis=0)
                
#                 preds = resnet_model.predict(x)
#                 decoded_preds = decode_predictions(preds, top=5)[0]
                
#                 for p in decoded_preds:
#                     if p[1] in professional_classes:
#                         total_score += professional_classes[p[1]] * p[2]
                
#                 sampled_frames += 1
                
#         return (total_score / sampled_frames) * 100 if sampled_frames > 0 else 0


# analyzer = InterviewAnalyzer()

# async def save_temp_file(file: UploadFile) -> str:
#     """Save uploaded file to temporary location"""
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#             logger.info(f"Video stored at: {temp_file.name}")
#             return temp_file.name
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")

# @router.post("/analyze-interview/")
# async def analyze_interview_endpoint(
#     video_url: str = Body(..., embed=True),
#     question: str = Body(...),
#     job_description: str = Body(...)
# ):
#     """Endpoint for video interview analysis with Cloudinary URL"""
#     temp_video_path = None
#     try:
#         # Validate Cloudinary URL
#         if not video_url.startswith(('http://', 'https://')):
#             raise HTTPException(400, "Invalid video URL format")
        
#         logger.info(f"Received Cloudinary video URL: {video_url}")

#         # Download video from Cloudinary
#         temp_video_path = await download_video_from_url(video_url)
#         logger.info(f"Downloaded temporary file to: {temp_video_path}")

#         # Verify downloaded file
#         if not os.path.exists(temp_video_path):
#             raise HTTPException(500, "Failed to download video file")
#         if os.path.getsize(temp_video_path) == 0:
#             raise HTTPException(400, "Downloaded video file is empty")

#         # Process analysis
#         logger.info("Starting video analysis...")
#         results = await analyzer.analyze_interview(
#             video_path=temp_video_path,
#             question=question,
#             job_description=job_description
#         )
        
#         logger.info("Analysis completed successfully")
#         return results

#     except HTTPException as he:
#         logger.error(f"HTTP Error {he.status_code}: {he.detail}")
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(500, "Internal processing error") from e
#     finally:
#         if temp_video_path and os.path.exists(temp_video_path):
#             try:
#                 os.remove(temp_video_path)
#                 logger.info(f"Cleaned up temporary file: {temp_video_path}")
#             except Exception as cleanup_error:
#                 logger.error(f"Cleanup failed: {str(cleanup_error)}")

# async def download_video_from_url(url: str) -> str:
#     """Download video from URL to temporary file"""
#     try:
#         async with httpx.AsyncClient(timeout=60.0) as client:
#             response = await client.get(url)
#             response.raise_for_status()
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#                 temp_file.write(response.content)
#                 return temp_file.name
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(502, f"Failed to download video: {e.response.text}")
#     except httpx.RequestError as e:
#         raise HTTPException(503, f"Video download service unavailable: {str(e)}")
    
# app = FastAPI()

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include router
# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)














































# import os
# import tempfile
# import subprocess
# import speech_recognition as sr
# from sentence_transformers import SentenceTransformer, util
# import torch
# import logging
# import traceback
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
# import noisereduce as nr
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import httpx
# import numpy as np
# from fastapi import Body 
# from io import BytesIO 

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize models
# model = SentenceTransformer('all-MiniLM-L6-v2')

# class AudioAnalyzer:
#     async def analyze_interview(self, audio_path: str, question: str, job_description: str):
#         """Main analysis entry point"""
#         try:
#             if not os.path.isfile(audio_path):
#                 raise ValueError("Audio file does not exist")

#             # Analyze audio content
#             transcript = self.transcribe_audio(audio_path)
#             answer_score = self.analyze_answer(transcript, question, job_description)
#             pronunciation_score = self.analyze_pronunciation(audio_path)
#             print("answer_score:", answer_score)
#             print("pronunciation_score:", pronunciation_score)
#             print("transcript:", transcript)
#             # Calculate total score
#             total_score = (
#                 answer_score * 0.7 +
#                 pronunciation_score * 0.3
#             )
#             print("total_score:", total_score)
            
#             return {
#                 "answer_score": round(answer_score, 2),
#                 "pronunciation_score": round(pronunciation_score, 2),
#                 "total_score": round(total_score, 2),
#                 "transcript": transcript
#             }
           
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

#     def transcribe_audio(self, audio_path: str) -> str:
#         """Convert speech to text"""
#         r = sr.Recognizer()
        
#         try:
#             audio = AudioSegment.from_file(io.BytesIO(audio_content))
            
#             # Audio processing
#             audio_array = np.array(audio.get_array_of_samples())
#             reduced_noise = nr.reduce_noise(
#                 y=audio_array,
#                 sr=audio.frame_rate,
#                 stationary=True
#             )
            
#             cleaned_audio = AudioSegment(
#                 reduced_noise.tobytes(),
#                 frame_rate=audio.frame_rate,
#                 sample_width=audio.sample_width,
#                 channels=audio.channels
#             )
            
#              # Use in-memory bytes
#             with io.BytesIO() as buffer:
#                     cleaned_audio.export(buffer, format="wav")
#                     buffer.seek(0)
                    
#                     with sr.AudioFile(buffer) as source:
#                         audio_data = r.record(source)
#                         return r.recognize_google(
#                             audio_data,
#                             language="en-US",
#                             show_all=False,
#                             pfilter=1
#                         )
                        
#         except sr.UnknownValueError:
#                 raise HTTPException(400, "Could not understand audio (low quality)")
#         except sr.RequestError as e:
#                 raise HTTPException(500, f"Speech service error: {str(e)}")
                
            
            
#         #     # Save processed audio
#         #     processed_path = audio_path.replace(".wav", "_processed.wav")
#         #     cleaned_audio.export(processed_path, format="wav")

#         #     with sr.AudioFile(processed_path) as source:
#         #         audio_data = r.record(source)
#         #         return r.recognize_google(
#         #             audio_data,
#         #             language="en-US",
#         #             show_all=False,
#         #             pfilter=1
#         #         )
#         # except sr.UnknownValueError:
#         #     raise HTTPException(400, "Could not understand audio (low quality)")
#         # except sr.RequestError as e:
#         #     raise HTTPException(500, f"Speech service error: {str(e)}")
#         # finally:
#         #     for f in [audio_path, processed_path]:
#         #         if os.path.exists(f):
#         #             os.remove(f)

#     def analyze_answer(self, transcript: str, question: str, job_desc: str) -> float:
#         """Semantic analysis of answer content"""
#         expected_answer = self.generate_expected_answer(question, job_desc)
#         transcript_embedding = model.encode(transcript)
#         expected_embedding = model.encode(expected_answer)
        
#         print("transcript:", transcript)
#         print("question:", question)
#         print("expected_answer:", expected_answer)
   
#         return util.pytorch_cos_sim(transcript_embedding, expected_embedding).item() * 100

#     def generate_expected_answer(self, question: str, job_desc: str) -> str:
#         """Generate ideal answer template"""
#         return f"A strong answer would demonstrate experience with {job_desc} " \
#                f"and specifically address {question}. Ideal responses should include " \
#                "relevant technical skills, problem-solving examples, and team collaboration experiences."

#     def analyze_pronunciation(self, audio_path: str) -> float:
#         """Pronunciation analysis"""
#         try:
#             audio = AudioSegment.from_file(audio_path)
#             r = sr.Recognizer()
            
#             with sr.AudioFile(audio_path) as source:
#                 audio_data = r.record(source)
#                 text = r.recognize_google(audio_data)
                
#             # Calculate speech metrics
#             duration = len(audio) / 1000  # in seconds
#             words = len(text.split())
#             speaking_rate = words / duration if duration > 0 else 0
            
#             # Score calculation
#             rate_score = max(0, 100 - abs(120 - speaking_rate)*2)  # Ideal 120 wpm
#             return min(rate_score * 0.6 + 40, 100)
#         except Exception as e:
#             return 65  # Minimum score if analysis fails

# analyzer = AudioAnalyzer()

# app = FastAPI()

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/analyze-interview/")
# async def analyze_interview_endpoint(
#      audio_url: str = Body(..., embed=True),
#     question: str = Body(...),
#     job_description: str = Body(...)
# ):
#     """Endpoint for audio interview analysis"""
#     temp_audio_path = None
#     try:
#         # Stream audio directly without saving to disk
#         async with httpx.AsyncClient(timeout=60.0) as client:
#             response = await client.get(audio_url)
#             response.raise_for_status()
#             audio_content = response.content
            
#             results = await analyzer.analyze_interview(
#                 audio_content=audio_content,
#                 question=question,
#                 job_description=job_description
#             )
#             return results
            
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(502, f"Failed to fetch audio: {e.response.text}")
#     except httpx.RequestError as e:
#         raise HTTPException(503, f"Audio download service unavailable: {str(e)}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(500, "Internal processing error") from e
    
    
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



















# # Updated FastAPI service for audio analysis


# import os
# import tempfile
# import subprocess
# import speech_recognition as sr
# from sentence_transformers import SentenceTransformer, util
# import torch
# from fastapi import FastAPI, APIRouter, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Annotated
# import logging
# import traceback
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
# import noisereduce as nr
# import httpx

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# model = SentenceTransformer('all-MiniLM-L6-v2')
# router = APIRouter()

# class AudioAnalyzer:
#     async def analyze_interview(self, audio_path: str, question: str, job_description: str):
#         try:
#             if not os.path.isfile(audio_path):
#                 raise ValueError("Audio file does not exist")

#             transcript = self.transcribe_audio(audio_path)
#             answer_score = self.analyze_answer(transcript, question, job_description)
#             pronunciation_score = self.analyze_pronunciation(audio_path)
            
#             total_score = (
#                 answer_score * 0.7 +
#                 pronunciation_score * 0.3
#             )
            
#             return {
#                 "answer_score": round(answer_score, 2),
#                 "pronunciation_score": round(pronunciation_score, 2),
#                 "total_score": round(total_score, 2),
#                 "transcript": transcript
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

#     def transcribe_audio(self, audio_path: str) -> str:
#         r = sr.Recognizer()
#         try:
#             audio = AudioSegment.from_file(audio_path)
#             processed_path = self.process_audio(audio)
            
#             with sr.AudioFile(processed_path) as source:
#                 audio_data = r.record(source)
#                 return r.recognize_google(
#                     audio_data,
#                     language="en-US",
#                     show_all=False,
#                     pfilter=1
#                 )
#         except sr.UnknownValueError:
#             raise HTTPException(400, "Could not understand audio (low quality)")
#         finally:
#             if os.path.exists(processed_path):
#                 os.remove(processed_path)

#     def process_audio(self, audio: AudioSegment) -> str:
#         # Noise reduction
#         audio_array = np.array(audio.get_array_of_samples())
#         reduced_noise = nr.reduce_noise(
#             y=audio_array,
#             sr=audio.frame_rate,
#             stationary=True
#         )
        
#         cleaned_audio = AudioSegment(
#             reduced_noise.tobytes(),
#             frame_rate=audio.frame_rate,
#             sample_width=audio.sample_width,
#             channels=audio.channels
#         )

#         # Normalization
#         normalized = cleaned_audio.normalize()
#         processed_path = os.path.join(tempfile.gettempdir(), "processed_audio.wav")
#         normalized.export(processed_path, format="wav")
#         return processed_path

#     def analyze_answer(self, transcript: str, question: str, job_desc: str) -> float:
#         expected_answer = self.generate_expected_answer(question, job_desc)
#         transcript_embedding = model.encode(transcript)
#         expected_embedding = model.encode(expected_answer)
#         return util.pytorch_cos_sim(transcript_embedding, expected_embedding).item() * 100

#     def generate_expected_answer(self, question: str, job_desc: str) -> str:
#         return f"A strong answer would demonstrate experience with {job_desc} " \
#                f"and specifically address {question}. Include relevant technical skills " \
#                "and problem-solving examples."

# analyzer = AudioAnalyzer()

# async def save_temp_file(file: UploadFile) -> str:
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#             return temp_file.name
#     except Exception as e:
#         raise HTTPException(500, detail=f"File save failed: {str(e)}")

# @router.post("/analyze_interview/")
# async def analyze_interview_endpoint(
#     audio_file: UploadFile = File(...),
#     question: str = Form(...),
#     job_description: str = Form(...)
# ):
#     temp_audio_path = None
#     try:
#         temp_audio_path = await save_temp_file(audio_file)
#         results = await analyzer.analyze_interview(
#             audio_path=temp_audio_path,
#             question=question,
#             job_description=job_description
#         )
#         return results
#     finally:
#         if temp_audio_path and os.path.exists(temp_audio_path):
#             os.remove(temp_audio_path)

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(router)


# import os
# import tempfile
# import subprocess
# import cv2
# import numpy as np
# import speech_recognition as sr
# from deepface import DeepFace
# from sentence_transformers import SentenceTransformer, util
# import torch
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# from fastapi import FastAPI, APIRouter, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Annotated
# import logging
# import traceback
# from pydub import AudioSegment
# from pydub.silence import detect_silence
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
# import noisereduce as nr
# import httpx


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)



# # Initialize models
# model = SentenceTransformer('all-MiniLM-L6-v2')
# resnet_model = ResNet50(weights='imagenet')
# router = APIRouter()

# # ImageNet classes for attire analysis
# imagenet_classes = [
#     'suit', 'tie', 'shirt', 'business suit', 'bow tie', 'jeans',
#     'sweatshirt', 'T-shirt', 'sweat pants', 'pajamas', 'lab coat'
# ]

# class InterviewAnalyzer:
#     async def analyze_interview(self, video_path: str, question: str, job_description: str):
#         """Main analysis entry point"""
#         try:
#             # Validate video file
#             if not os.path.isfile(video_path):
#                 raise ValueError("Video file does not exist")
            
#             # Basic video file validation
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 raise ValueError("Unable to open video file")
#             cap.release()

#        # try:
#             # Analyze answer content
#             transcript = self.transcribe_video(video_path)
#             answer_score = self.analyze_answer(transcript, question, job_description)
            
#             # Analyze pronunciation and grammar
#             pronunciation_score = self.analyze_pronunciation(video_path)
            
#             # Analyze eye contact
#             eye_contact_score = self.analyze_eye_contact(video_path)
            
#             # Analyze attire
#             attire_score = self.analyze_attire(video_path)
            
#             # Calculate total score
#             total_score = (
#                 answer_score * 0.4 +
#                 pronunciation_score * 0.3 +
#                 eye_contact_score * 0.15 +
#                 attire_score * 0.15
#             )
            
#             return {
#                 "answer_score": round(answer_score, 2),
#                 "pronunciation_score": round(pronunciation_score, 2),
#                 "eye_contact_score": round(eye_contact_score, 2),
#                 "attire_score": round(attire_score, 2),
#                 "total_score": round(total_score, 2),
#                 "transcript": transcript
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

#     def transcribe_video(self, video_path: str) -> str:
#         """Convert speech to text"""
#         r = sr.Recognizer()
#         audio_path = self.extract_audio(video_path)
        
#         try:
#             # Check audio quality before processing
#             audio = AudioSegment.from_wav(audio_path)
            
#             # 1. Noise reduction
#             audio_array = np.array(audio.get_array_of_samples())
#             reduced_noise = nr.reduce_noise(
#                 y=audio_array,
#                 sr=audio.frame_rate,
#                 stationary=True
#             )
#             cleaned_audio = AudioSegment(
#                 reduced_noise.tobytes(),
#                 frame_rate=audio.frame_rate,
#                 sample_width=audio.sample_width,
#                 channels=audio.channels
#             )
#             # 2. Silence detection
#             non_silent = detect_nonsilent(
#                 cleaned_audio,
#                 min_silence_len=500,
#                 silence_thresh=-40
#             )
            
#             if not non_silent:
#                 raise HTTPException(400, "No speech detected in audio")
            
            
#             # 3. Audio normalization
#             normalized = cleaned_audio.normalize()

#             # Save processed audio
#             processed_path = audio_path.replace(".wav", "_processed.wav")
#             normalized.export(processed_path, format="wav")

#             # Transcription with adjusted parameters
#             with sr.AudioFile(processed_path) as source:
#                 audio_data = r.record(source)
#                 return r.recognize_google(
#                     audio_data,
#                     language="en-US",
#                     show_all=False,
#                     pfilter=1  # Enable profanity filter
#                 )
#         except sr.UnknownValueError:
#                 raise HTTPException(400, "Could not understand audio (low quality)")
#         except sr.RequestError as e:
#                 raise HTTPException(500, f"Speech service error: {str(e)}")
#         finally:
#                 for f in [audio_path, processed_path]:
#                     if os.path.exists(f):
#                         os.remove(f)        
            
#         #     # Check duration
#         #     if len(audio) < 1000:  # 1 second minimum
#         #         raise HTTPException(status_code=400, detail="Audio too short")
#         #     # Transcribe audio
#         #     with sr.AudioFile(audio_path) as source:
#         #         audio = r.record(source)
#         #     return r.recognize_google(audio)
#         # except sr.UnknownValueError:
#         #     raise HTTPException(status_code=400, detail="Could not understand audio")
#         # except sr.RequestError as e:
#         #     raise HTTPException(status_code=500, detail=f"Speech recognition error: {e}")
#         # finally:
#         #     if os.path.exists(audio_path):
#         #         os.remove(audio_path)

#     def analyze_answer(self, transcript: str, question: str, job_desc: str) -> float:
#         """Semantic analysis of answer content"""
#         expected_answer = self.generate_expected_answer(question, job_desc)
#         transcript_embedding = model.encode(transcript)
#         expected_embedding = model.encode(expected_answer)
#         return util.pytorch_cos_sim(transcript_embedding, expected_embedding).item() * 100

#     def generate_expected_answer(self, question: str, job_desc: str) -> str:
#         """Generate ideal answer template"""
#         return f"A strong answer would demonstrate experience with {job_desc} " \
#                f"and specifically address {question}. Ideal responses should include " \
#                "relevant technical skills, problem-solving examples, and team collaboration experiences."

#     def analyze_pronunciation(self, video_path: str) -> float:
#         """Pronunciation and grammar analysis"""
#         try:
#             audio_path = self.extract_audio(video_path)
#             r = sr.Recognizer()
            
#             with sr.AudioFile(audio_path) as source:
#                 audio = r.record(source)
#                 text = r.recognize_google(audio)
                
#             # Grammar check
#             sentence_count = len(text.split('.'))
#             word_count = len(text.split())
#             complexity_score = min((word_count / sentence_count) / 10 * 100, 100)
            
#             return complexity_score * 0.6 + 40  # Base score + complexity
            
#         except Exception as e:
#             return 65  # Minimum score if analysis fails
#         finally:
#             if os.path.exists(audio_path):
#                 os.remove(audio_path)

#    # Updated eye contact analysis
#     def analyze_eye_contact(self, video_path: str) -> float:
#         cap = cv2.VideoCapture(video_path)
#         total_frames = 0
#         looking_frames = 0
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
#             if len(faces) > 0:
#                 # Calculate eye region
#                 (x,y,w,h) = faces[0]
#                 eye_region = frame[y:y+h//2, x:x+w]
                
#                 # Detect pupils using HoughCircles
#                 circles = cv2.HoughCircles(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY),
#                     cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, 
#                     minRadius=0, maxRadius=0)
                
#                 if circles is not None:
#                     looking_frames += 1
                    
#             total_frames += 1
            
#         return (looking_frames / total_frames) * 100 if total_frames > 0 else 0

#     def extract_audio(self, video_path: str) -> str:
#         """Extract audio using ffmpeg"""
#         try:
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
#                 command = [
#                     'ffmpeg',
#                     '-y',
#                     '-i', video_path,
#                     '-vn', '-acodec', 'pcm_s16le',
#                     '-acodec', 'pcm_s16le',
#                     '-ar', '16000', '-ac', '1',
#                     '-loglevel', 'debug',  # Add detailed logging
#                     temp_audio.name
#                 ]
#                 logger.info(f"Audio stored at: {temp_audio.name}")
#                 result = subprocess.run(
#                     command,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True
#                 )
                
#                 if result.returncode != 0:
#                     logger.error(f"FFmpeg Error: {result.stderr}")
#                     raise RuntimeError(f"Audio extraction failed: {result.stderr}")
                    
#                 # Verify output file
#                 if os.path.getsize(temp_audio.name) == 0:
#                     raise RuntimeError("Empty audio file generated")
            
#                 return temp_audio.name
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Audio extraction failed: {str(e)}")


#     # Improved attire analysis
#     def analyze_attire(self, video_path: str) -> float:
#         professional_classes = {
#             'suit': 0.7, 'tie': 0.6, 'dress_shirt': 0.8, 
#             'business_suit': 0.9, 'formal_wear': 0.85
#         }
        
#         cap = cv2.VideoCapture(video_path)
#         total_score = 0
#         sampled_frames = 0
        
#         while cap.isOpened() and sampled_frames < 30:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             if sampled_frames % 10 == 0:  # Sample every 10th frame
#                 img = Image.fromarray(frame).resize((224, 224))
#                 x = image.img_to_array(img)
#                 x = preprocess_input(x)
#                 x = np.expand_dims(x, axis=0)
                
#                 preds = resnet_model.predict(x)
#                 decoded_preds = decode_predictions(preds, top=5)[0]
                
#                 for p in decoded_preds:
#                     if p[1] in professional_classes:
#                         total_score += professional_classes[p[1]] * p[2]
                
#                 sampled_frames += 1
                
#         return (total_score / sampled_frames) * 100 if sampled_frames > 0 else 0


# analyzer = InterviewAnalyzer()

# async def save_temp_file(file: UploadFile) -> str:
#     """Save uploaded file to temporary location"""
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#             logger.info(f"Video stored at: {temp_file.name}")
#             return temp_file.name
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")

# @router.post("/analyze-interview/")
# async def analyze_interview_endpoint(
#     video_url: str = Body(..., embed=True),
#     question: str = Body(...),
#     job_description: str = Body(...)
# ):
#     """Endpoint for video interview analysis with Cloudinary URL"""
#     temp_video_path = None
#     try:
#         # Validate Cloudinary URL
#         if not video_url.startswith(('http://', 'https://')):
#             raise HTTPException(400, "Invalid video URL format")
        
#         logger.info(f"Received Cloudinary video URL: {video_url}")

#         # Download video from Cloudinary
#         temp_video_path = await download_video_from_url(video_url)
#         logger.info(f"Downloaded temporary file to: {temp_video_path}")

#         # Verify downloaded file
#         if not os.path.exists(temp_video_path):
#             raise HTTPException(500, "Failed to download video file")
#         if os.path.getsize(temp_video_path) == 0:
#             raise HTTPException(400, "Downloaded video file is empty")

#         # Process analysis
#         logger.info("Starting video analysis...")
#         results = await analyzer.analyze_interview(
#             video_path=temp_video_path,
#             question=question,
#             job_description=job_description
#         )
        
#         logger.info("Analysis completed successfully")
#         return results

#     except HTTPException as he:
#         logger.error(f"HTTP Error {he.status_code}: {he.detail}")
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(500, "Internal processing error") from e
#     finally:
#         if temp_video_path and os.path.exists(temp_video_path):
#             try:
#                 os.remove(temp_video_path)
#                 logger.info(f"Cleaned up temporary file: {temp_video_path}")
#             except Exception as cleanup_error:
#                 logger.error(f"Cleanup failed: {str(cleanup_error)}")

# async def download_video_from_url(url: str) -> str:
#     """Download video from URL to temporary file"""
#     try:
#         async with httpx.AsyncClient(timeout=60.0) as client:
#             response = await client.get(url)
#             response.raise_for_status()
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#                 temp_file.write(response.content)
#                 return temp_file.name
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(502, f"Failed to download video: {e.response.text}")
#     except httpx.RequestError as e:
#         raise HTTPException(503, f"Video download service unavailable: {str(e)}")
    
# app = FastAPI()

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include router
# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import tempfile
import subprocess
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydub import AudioSegment
import noisereduce as nr

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class AudioAnalyzer:
    async def analyze_audio(self, audio_path: str, question: str, job_desc: str):
        try:
            # Audio preprocessing
            cleaned_path = self.preprocess_audio(audio_path)
            
            # Transcription
            transcript = self.transcribe_audio(cleaned_path)
            
            # Pronunciation analysis
            pronunciation_score = self.analyze_pronunciation(cleaned_path)
            
            # Grammar analysis
            grammar_score = self.analyze_grammar(transcript)
            
            # Answer relevance
            answer_score = self.analyze_answer_relevance(transcript, question, job_desc)
            
            total_score = (
                pronunciation_score * 0.4 +
                grammar_score * 0.3 +
                answer_score * 0.3
            )
            
            return {
                "pronunciation_score": round(pronunciation_score, 2),
                "grammar_score": round(grammar_score, 2),
                "answer_score": round(answer_score, 2),
                "total_score": round(total_score, 2),
                "transcript": transcript
            }
            
        finally:
            for f in [audio_path, cleaned_path]:
                if os.path.exists(f):
                    os.remove(f)

    def preprocess_audio(self, audio_path: str) -> str:
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Noise reduction
            samples = np.array(audio.get_array_of_samples())
            reduced_noise = nr.reduce_noise(
                y=samples.astype(np.float32),
                sr=audio.frame_rate,
                stationary=True
            )
            
            cleaned_audio = AudioSegment(
                reduced_noise.astype(np.int16).tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=2,
                channels=1
            )
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                cleaned_audio.export(f.name, format="wav")
                return f.name
                
        except Exception as e:
            raise HTTPException(500, f"Audio processing failed: {str(e)}")

    def transcribe_audio(self, audio_path: str) -> str:
        r = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio = r.record(source)
                return r.recognize_google(audio)
        except sr.UnknownValueError:
            raise HTTPException(400, "Could not understand audio")
        except sr.RequestError as e:
            raise HTTPException(500, f"Speech service error: {e}")

    def analyze_pronunciation(self, audio_path: str) -> float:
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            # Example: Analyze speaking rate
            intervals = librosa.effects.split(y, top_db=20)
            speech_duration = sum(end - start for start, end in intervals) / sr
            total_duration = len(y) / sr
            speaking_ratio = speech_duration / total_duration
            
            # Pitch analysis
            pitches = librosa.yin(y, fmin=80, fmax=400)
            pitch_variation = np.std(pitches)
            
            # Combine features into score
            score = min(100, (speaking_ratio * 70) + (30 - min(30, pitch_variation)))
            return score
            
        except Exception as e:
            return 70  # Fallback score

    def analyze_grammar(self, text: str) -> float:
        # Implement your grammar checking logic here
        # This could integrate with an external API or NLP model
        return 85  # Placeholder

    def analyze_answer_relevance(self, transcript: str, question: str, job_desc: str) -> float:
        expected_answer = self.generate_expected_answer(question, job_desc)
        transcript_embed = model.encode(transcript)
        expected_embed = model.encode(expected_answer)
        return util.pytorch_cos_sim(transcript_embed, expected_embed).item() * 100

    def generate_expected_answer(self, question: str, job_desc: str) -> str:
        return f"A strong answer would address {question} while demonstrating skills relevant to {job_desc}."

analyzer = AudioAnalyzer()

@app.post("/analyze_audio/")
async def analyze_audio_endpoint(
    question: str,
    job_description: str,
    audio: UploadFile = File(...)
):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        return await analyzer.analyze_audio(tmp_path, question, job_description)
        
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    
