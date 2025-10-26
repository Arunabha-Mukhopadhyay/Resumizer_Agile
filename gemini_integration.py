import ollama

class OllamaQA:
    def __init__(self, model="mistral", resume_text=""):
        self.model = model
        self.context = resume_text

    def run(self, question):
        prompt = f"""
        You are an AI assistant helping analyze a candidate's resume.

        Resume:
        ---------
        {self.context}

        Question: {question}

        Answer:
        """

        print(f"[🧠] Asking Ollama model: '{self.model}'") 

        try:
            response = ollama.chat(
                model=self.model.strip(),
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"⚠️ Error calling Ollama (model='{self.model}'): {str(e)}"
        



def get_question_answer_chain(vector_store, model="mistral"):
    try:
        docs = vector_store.docstore._dict.values()
        context = "\n".join(doc.page_content for doc in docs)
    except Exception:
        context = "Candidate resume not found."

    return OllamaQA(model=model, resume_text=context)