from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys
import traceback
from io import StringIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception as e:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{tb}

Return the line number(s) where the error is located."""

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER)
                    )
                },
                required=["error_lines"]
            )
        )
    )

    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines


@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=422, detail="Code cannot be empty")

    execution = execute_python_code(request.code)

    if execution["success"]:
        return JSONResponse(content={"error": [], "result": execution["output"]})
    else:
        try:
            error_lines = analyze_error_with_ai(request.code, execution["output"])
        except Exception as e:
            error_lines = []
        return JSONResponse(content={"error": error_lines, "result": execution["output"]})


@app.get("/health")
async def health():
    return {"status": "ok"}
