from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import json
import numpy as np
import requests
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()
data = []
embeddings = None
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

def get_embedding(text):
    """Fetch embedding for text using AI Proxy."""
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {"model": "text-embedding-3-small", "input": text[:8192]}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.on_event("startup")
def load_data():
    """Load content and embeddings from SQLite on startup."""
    global data, embeddings
    try:
        conn = sqlite3.connect("tds_data.db")
        c = conn.cursor()
        c.execute("SELECT id, source, content, url, title, embedding FROM content")
        rows = c.fetchall()
        conn.close()
        data = [{"id": r[0], "source": r[1], "content": r[2], "url": r[3], "title": r[4], "embedding": json.loads(r[5]) if r[5] else None} for r in rows]
        embeddings = np.array([d["embedding"] for d in data if d["embedding"]])
        print(f"Loaded {len(data)} items from database")
    except Exception as e:
        print(f"Database load error: {e}")

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Handle POST requests to answer questions."""
    try:
        # Compute question embedding
        q_embedding = get_embedding(request.question)
        
        # Find top-5 similar content
        similarities = np.dot(embeddings, q_embedding)
        k = min(5, len(embeddings))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_data = [data[i] for i in top_k_indices]
        
        # Create context
        context = "\n".join([d["content"] for d in top_k_data])
        
        # Prepare AI Proxy request
        messages = [
            {"role": "system", "content": "You are a Teaching Assistant for the Tools in Data Science course at IIT Madras. Answer questions based on the provided course content and Discourse posts. Provide concise answers and avoid generating links."},
            {"role": "user", "content": [{"type": "text", "text": f"Based on the following course content and Discourse posts, answer the question.\n\nCourse content:\n{context}\n\nQuestion: {request.question}"}]}
        ]
        if request.image:
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{request.image}"}})
        
        # Call AI Proxy
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AIPROXY_TOKEN}"}
        payload = {"model": "gpt-4o-mini", "messages": messages}
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        # Prepare links
        links = [{"url": d["url"], "text": d["title"]} for d in top_k_data]
        
        return {"answer": answer, "links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
