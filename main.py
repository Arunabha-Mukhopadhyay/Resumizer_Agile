# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from parser import parse_resume, save_parsed_data
from vector import generate_vector_store
from gemini_integration import get_question_answer_chain  # ✅ Updated backend now uses model param
from monitor import log_event
import os
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel

# ✅ Load .env file before reading model
load_dotenv()

app = FastAPI(
    title="Resumizer AI - Resume Analyzer API",
    description="Upload a resume and get AI responses.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class QuestionRequest(BaseModel):
    resume_id: str
    question: str
    model: str | None = None

_resume_store = {}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend not found.</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

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

        model = os.getenv("OLLAMA_MODEL", "mistral")
        qa_chain = get_question_answer_chain(vector_store, model=model)

        questions = [
            "What are the candidate's key skills?",
            "Does the resume mention leadership or team management?",
            "What industries or domains has the candidate worked in?",
            "Suggest 5 ways to improve this resume.",
            "what is the ATS score?"
        ]

        results = {}
        for q in questions:
            results[q] = qa_chain.run(q)

        resume_id = str(uuid.uuid4())
        _resume_store[resume_id] = {
            "vector_store": vector_store,
            "model": model,
            "filename": file.filename
        }

        return {"filename": file.filename, "resume_id": resume_id, "analysis": results}

    except Exception as e:
        log_event("error", f"Processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    resume_data = _resume_store.get(request.resume_id)
    if not resume_data:
        raise HTTPException(status_code=404, detail="Resume not found. Please upload a resume first.")

    model = request.model or resume_data["model"]
    qa_chain = get_question_answer_chain(resume_data["vector_store"], model=model)

    log_event("question", f"{request.resume_id}: {request.question}")

    try:
        answer = qa_chain.run(request.question)
    except Exception as e:
        log_event("error", f"Q&A failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "model": model}
