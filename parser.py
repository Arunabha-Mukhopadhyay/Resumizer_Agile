# parser.py
import os
import json
import pdfplumber
from docx import Document
from unstructured.partition.auto import partition

def parse_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())
    return text

def parse_docx(file_path):
    doc = Document(file_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text

def parse_resume(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

def save_parsed_data(text, filename):
    out_path = os.path.join("data", f"{filename}.json")
    os.makedirs("data", exist_ok=True)
    json_data = {"content": text}
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=4)
    return out_path