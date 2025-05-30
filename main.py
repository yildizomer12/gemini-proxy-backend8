import os
import uuid
import time
import json
import asyncio
import base64
from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

# FastAPI app
app = FastAPI(title="Gemini Backend for Vercel")

# API Keys - production'da environment variable kullanın
API_KEYS = [
    "AIzaSyCT1PXjhup0VHx3Fz4AioHbVUHED0fVBP4",
    "AIzaSyArNqpA1EeeXBx-S3EVnP0tzao6r4BQnO0",
    "AIzaSyCXICPfRTnNAFwNQMmtBIb3Pi0pR4SydHg",
    "AIzaSyDiLvp7CU443luErAz3Ck0B8zFdm8UvNRs",
    "AIzaSyBzqJebfbVPcBXQy7r4Y5sVgC499uV85i0",
    "AIzaSyD6AFGKycSp1glkNEuARknMLvo93YbCqH8",
    "AIzaSyBTara5UhTbLR6qnaUI6nyV4wugycoABRM",
    "AIzaSyBI2Jc8mHJgjnXnx2udyibIZyNq8SGlLSY",
    "AIzaSyAcgdqbZsX9UOG4QieFSW7xCcwlHzDSURY",
    "AIzaSyAwOawlX-YI7_xvXY-A-3Ks3k9CxiTQfy4",
    "AIzaSyCJVUeJkqYeLNG6UsF06Gasn4mvMFfPhzw",
    "AIzaSyBFOK0YgaQOg5wilQul0P2LqHk1BgeYErw",
    "AIzaSyBQRsGHOhaiD2cNb5F68hI6BcZR7CXqmwc",
    "AIzaSyCIC16VVTlFGbiQtq7RlstTTqPYizTB7yQ",
    "AIzaSyCIlfHXQ9vannx6G9Pae0rKwWJpdstcZIM",
    "AIzaSyAUIR9gx08SNgeHq8zKAa9wyFtFu00reTM",
    "AIzaSyAST1jah1vAcnLfmofR4DDw0rjYkJXJoWg",
    "AIzaSyAV8OU1_ANXTIvkRooikeNrI1EMR3IbTyQ"
]

# Simple state management (in-memory)
current_key_index = 0
key_usage = {key: 0 for key in API_KEYS}

# Pydantic Models
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

def get_next_key():
    """Simple round-robin key selection"""
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    key_usage[key] = key_usage.get(key, 0) + 1
    return key

def process_content(content):
    """Convert OpenAI format to Gemini format"""
    if isinstance(content, str):
        return [{"text": content}]
    
    parts = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append({"text": item.text})
        elif item.type == "image_url" and item.image_url:
            try:
                # Handle base64 images
                if item.image_url.url.startswith("data:"):
                    header, base64_data = item.image_url.url.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    })
            except:
                parts.append({"text": "[Image processing error]"})
    
    return parts or [{"text": ""}]

def convert_messages(messages):
    """Convert OpenAI messages to Gemini format"""
    return [
        {
            "role": "user" if msg.role == "user" else "model",
            "parts": process_content(msg.content)
        }
        for msg in messages
    ]

async def stream_gemini_response(gemini_response: httpx.Response, model: str):
    """Streams Gemini API response in OpenAI format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # Initial chunk - role başlangıcı
    initial_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    try:
        yield f"data: {json.dumps(initial_chunk)}\n\n"
    except BrokenPipeError:
        print("Client disconnected during initial chunk (BrokenPipeError).")
        return

    async for chunk in gemini_response.aiter_bytes():
        try:
            # Each chunk from Gemini is a complete JSON object, but the stream
            # might concatenate them. We need to split them.
            # Gemini's SSE stream uses 'data: ' prefix.
            data_str = chunk.decode('utf-8')
            for line in data_str.splitlines():
                if line.startswith("data: "):
                    json_str = line[len("data: "):]
                    if json_str.strip(): # Ensure it's not an empty data line
                        try:
                            gemini_chunk = json.loads(json_str)
                            if gemini_chunk.get("candidates"):
                                for candidate in gemini_chunk["candidates"]:
                                    if candidate.get("content") and candidate["content"].get("parts"):
                                        for part in candidate["content"]["parts"]:
                                            if part.get("text"):
                                                content_chunk = {
                                                    "id": chunk_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"content": part["text"]},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(content_chunk)}\n\n"
                        except json.JSONDecodeError:
                            print(f"JSON Decode Error for chunk: {json_str}")
                elif line.strip() == "": # End of data block
                    pass
        except BrokenPipeError:
            print("Client disconnected during content chunk (BrokenPipeError).")
            return
        except Exception as e:
            print(f"Error processing Gemini stream chunk: {e}")
            error_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error"
                }]
            }
            try:
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except BrokenPipeError:
                print("Client disconnected during error handling (BrokenPipeError).")
            return

    # Final chunk - streaming tamamlandı sinyali
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    try:
        yield f"data: {json.dumps(final_chunk)}\n\n"
    except BrokenPipeError:
        print("Client disconnected during final chunk (BrokenPipeError).")
        return

async def make_gemini_request(api_key: str, model: str, messages: list, generation_config: dict, stream: bool = False):
    """Gemini API'ye request yapar"""
    async with httpx.AsyncClient(timeout=45) as client:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        if stream:
            url += "?alt=sse" # Use Server-Sent Events for streaming

        response = await client.post(
            url,
            json={
                "contents": messages,
                "generationConfig": generation_config
            },
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
        )
        return response

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # Convert messages
        gemini_messages = convert_messages(request.messages)
        generation_config = {
            "temperature": request.temperature,
            "topK": 40,
            "topP": 0.95
        }
        
        if request.max_tokens:
            generation_config["maxOutputTokens"] = min(request.max_tokens, 8192)
        
        # API key seç
        api_key = get_next_key()
        
        # Gemini API çağrısı
        response = await make_gemini_request(
            api_key,
            request.model,
            gemini_messages,
            generation_config,
            stream=request.stream # Pass stream parameter
        )
        
        if not response.is_success:
            error_detail = f"Gemini API error: {response.status_code} - {response.text}"
            if response.status_code == 429:
                error_detail = "Rate limit exceeded, trying another key"
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        if request.stream:
            return StreamingResponse(
                stream_gemini_response(response, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )

        # Regular response (non-streaming)
        result = response.json()
        text = ""
        if result.get("candidates") and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if candidate.get("content") and candidate["content"].get("parts"):
                text = "".join(part.get("text", "") for part in candidate["content"]["parts"])

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(str(request.messages)),
                "completion_tokens": len(text),
                "total_tokens": len(str(request.messages)) + len(text)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "total_requests": sum(key_usage.values()),
        "active_keys": len([k for k, v in key_usage.items() if v > 0])
    }

@app.get("/")
async def root():
    return {
        "message": "Gemini Proxy API",
        "version": "1.0.0",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"]
    }

# CORS Middleware
@app.middleware("http")
async def cors_handler(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    
    # OPTIONS request handling
    if request.method == "OPTIONS":
        response.status_code = 200
    
    return response