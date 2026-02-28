from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    video_url: str
    topic: str


def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID")


def get_transcript(video_id: str) -> list:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in transcript_list:
                return t.fetch()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not get transcript: {str(e)}")


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_timestamp_with_gemini(transcript: list, topic: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Format transcript as text with timestamps
    transcript_text = "\n".join([
        f"[{seconds_to_hhmmss(entry['start'])}] {entry['text']}"
        for entry in transcript
    ])

    prompt = f"""You are analyzing a YouTube video transcript with timestamps.
Find the FIRST moment where this topic is spoken or discussed: "{topic}"

TRANSCRIPT:
{transcript_text[:50000]}

Return the timestamp (HH:MM:SS) of when this topic first appears in the video.
If not found, return the closest related timestamp."""

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "timestamp": types.Schema(
                        type=types.Type.STRING,
                        description="Timestamp in HH:MM:SS format"
                    )
                },
                required=["timestamp"]
            )
        )
    )

    import json
    result = json.loads(response.text)
    timestamp = result.get("timestamp", "00:00:00")

    # Ensure HH:MM:SS format
    parts = timestamp.split(":")
    if len(parts) == 2:
        timestamp = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    elif len(parts) == 3:
        timestamp = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    else:
        timestamp = "00:00:00"

    return timestamp


@app.post("/ask")
async def ask(request: AskRequest):
    if not request.video_url or not request.topic:
        raise HTTPException(status_code=422, detail="video_url and topic are required")

    try:
        video_id = extract_video_id(request.video_url)
        transcript = get_transcript(video_id)
        timestamp = find_timestamp_with_gemini(transcript, request.topic)

        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
