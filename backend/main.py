import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import os

app = FastAPI()

# Get allowed origins from environment variable, default to localhost for development
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8081").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest):
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")  # Default to ollama service
    logging.debug(f"Using OLLAMA_URL: {ollama_url}")
    headers = {"Content-Type": "application/json"}
    sanitized_query = request.query.strip().replace("\n", " ").replace("\t", " ")
    logging.debug(f"Sanitized query: {sanitized_query}")
    payload = {"model": "llama3.2:1b", "prompt": sanitized_query}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(ollama_url, json=payload, headers=headers, timeout=120.0)
            response.raise_for_status()
            full_response = []
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk.strip()
                try:
                    while "\n" in buffer:
                        json_chunk, buffer = buffer.split("\n", 1)
                        parsed_chunk = json.loads(json_chunk.strip())
                        if "response" in parsed_chunk:
                            full_response.append(parsed_chunk["response"])
                except (json.JSONDecodeError, ValueError):
                    pass
            final_response = "".join(full_response).strip()
            logging.debug(f"Final response: {final_response}")
            return {"response": final_response}
    except httpx.RequestError as e:
        logging.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP status error: {e}")
        raise HTTPException(status_code=500, detail=f"HTTP error from Ollama: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")