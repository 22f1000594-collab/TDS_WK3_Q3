from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class CommentRequest(BaseModel):
    comment: str


@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the given comment and respond ONLY with a JSON object. "
                        "The JSON must have exactly two fields:\n"
                        "- sentiment: one of 'positive', 'negative', or 'neutral'\n"
                        "- rating: an integer from 1 to 5 where 5=highly positive, 1=highly negative, 3=neutral\n"
                        "Example: {\"sentiment\": \"positive\", \"rating\": 5}\n"
                        "Do not include any explanation or extra text."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        raw = response.choices[0].message.content
        result = json.loads(raw)

        sentiment = result.get("sentiment", "neutral").lower()
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"

        rating = int(result.get("rating", 3))
        rating = max(1, min(5, rating))

        return JSONResponse(content={"sentiment": sentiment, "rating": rating})

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse model response as JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
