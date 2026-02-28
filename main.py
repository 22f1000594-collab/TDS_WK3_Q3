from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import os
import json

app = FastAPI()

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class CommentRequest(BaseModel):
    comment: str


SENTIMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "rating": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5]
                }
            },
            "required": ["sentiment", "rating"],
            "additionalProperties": False
        }
    }
}


@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the given comment and return:\n"
                        "- sentiment: 'positive', 'negative', or 'neutral'\n"
                        "- rating: integer 1-5 where 5=highly positive, 1=highly negative, 3=neutral"
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format=SENTIMENT_SCHEMA
        )

        result = json.loads(response.choices[0].message.content)
        return JSONResponse(content=result)

    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to OpenAI API")
    except openai.AuthenticationError:
        raise HTTPException(status_code=500, detail="Invalid OpenAI API key")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded")
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
