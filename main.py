from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from genre_classifier import predict_genre
from stems_checker import analyze_track
import shutil
import os
import uuid

app = FastAPI()

# CORS to allow Next.js frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save uploaded audio to a temp path
    temp_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    temp_path = f"temp/{temp_id}.{ext}"
    os.makedirs("temp", exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run genre classification
    try:
        genre = predict_genre(temp_path)
    except Exception as e:
        genre = "unknown"
        print("Genre error:", e)

    # Run stem analysis
    try:
        stem_results = analyze_track(temp_path)
        looking_for = [stem for stem, rms in stem_results.items() if rms < 0.01]
    except Exception as e:
        looking_for = []
        print("Stems error:", e)

    # Clean up
    os.remove(temp_path)

    return {
        "genre": genre,
        "looking_for": looking_for
    }
