# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from parser import parse_resume, save_parsed_data
from vector import generate_vector_store
from gemini_integration import get_question_answer_chain  # ✅ Updated backend now uses model param
from monitor import log_event
import os
from dotenv import load_dotenv

# ✅ Load .env file before reading model
load_dotenv()

app = FastAPI(
    title="Resumizer AI - Resume Analyzer API",
    description="Upload a resume and get AI responses.",
    version="1.0.0"
)

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".pdf", ".docx"]:
        return JSONResponse(status_code=400, content={"error": "Only .pdf and .docx are supported"})

    resume_path = os.path.join("resume_samples", file.filename)
    os.makedirs("resume_samples", exist_ok=True)

    with open(resume_path, "wb") as f:
        f.write(await file.read())

    log_event("upload", f"{file.filename} uploaded")

    try:
        # ✅ Parse the resume into text
        text = parse_resume(resume_path)
        save_parsed_data(text, file.filename.split('.')[0])
        vector_store = generate_vector_store(text)

        # ✅ ✅ ✅ Load model from .env and pass it to get_question_answer_chain
        model = os.getenv("OLLAMA_MODEL", "mistral")
        qa_chain = get_question_answer_chain(vector_store, model=model)

        questions = [
            "What are the candidate's key skills?",
            "Does the resume mention leadership or team management?",
            "What industries or domains has the candidate worked in?",
            "Suggest 5 ways to improve this resume.",
        ]

        results = {}
        for q in questions:
            results[q] = qa_chain.run(q)

        return {"filename": file.filename, "analysis": results}

    except Exception as e:
        log_event("error", f"Processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})