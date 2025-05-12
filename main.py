from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    goal: str
    tone: str
    context: str

@app.post("/api/generate-email")
async def generate_email(request: EmailRequest):
    prompt = (
        f"You are an expert sales/recruiting assistant. Write a short, tailored cold outreach email. "
        f"The goal is to {request.goal}. Tone: {request.tone}. Context: {request.context}. "
        f"Keep it under 120 words. Include a subject line."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful outbound email generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=300
        )

        email = response.choices[0].message.content.strip()
        return {"email": email}

    except Exception as e:
        return {"error": str(e)}
