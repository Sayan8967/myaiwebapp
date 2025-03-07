import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import os

# Initialize the FastAPI app
app = FastAPI()

# Get allowed origins from environment variable, default to localhost for development
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8081").split(",")

# Add CORS middleware to allow cross-origin requests from specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging to debug level for detailed output
logging.basicConfig(level=logging.DEBUG)

# Define the request model for the /query endpoint
class QueryRequest(BaseModel):
    query: str

# Define the /query endpoint to handle POST requests
@app.post("/query")
async def process_query(request: QueryRequest):
    # Get the Ollama service URL from environment variable or use default
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
    logging.debug(f"Using OLLAMA_URL: {ollama_url}")
    
    # Set headers for the request to Ollama
    headers = {"Content-Type": "application/json"}
    
    # Sanitize the query by removing extra spaces and newlines
    sanitized_query = request.query.strip().replace("\n", " ").replace("\t", " ")
    logging.debug(f"Sanitized query: {sanitized_query}")
    
    # Prepare the payload for the Ollama service
    payload = {"model": "llama3.2:1b", "prompt": sanitized_query}

    try:
        # Use httpx.AsyncClient to make an asynchronous POST request to Ollama
        async with httpx.AsyncClient() as client:
            # Increased timeout to 300 seconds to avoid 504 Gateway Timeout error
            response = await client.post(ollama_url, json=payload, headers=headers, timeout=300.0)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)
            
            # Initialize variables to collect the full response
            full_response = []
            buffer = ""
            
            # Process the streaming response from Ollama
            async for chunk in response.aiter_text():
                buffer += chunk.strip()
                try:
                    while "\n" in buffer:
                        json_chunk, buffer = buffer.split("\n", 1)
                        parsed_chunk = json.loads(json_chunk.strip())
                        if "response" in parsed_chunk:
                            full_response.append(parsed_chunk["response"])
                except (json.JSONDecodeError, ValueError):
                    pass  # Ignore incomplete or invalid chunks
            
            # Combine all response parts into the final response
            final_response = "".join(full_response).strip()
            logging.debug(f"Final response: {final_response}")
            
            # Return the response to the client
            return {"response": final_response}
    
    # Handle request errors (e.g., network issues)
    except httpx.RequestError as e:
        logging.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")
    
    # Handle HTTP status errors from Ollama (e.g., 500 from Ollama)
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP status error: {e}")
        raise HTTPException(status_code=500, detail=f"HTTP error from Ollama: {str(e)}")
    
    # Handle any other unexpected errors
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")