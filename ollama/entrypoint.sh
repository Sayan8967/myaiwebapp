#!/bin/sh
set -e

echo "Starting Ollama server..."
ollama serve &

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Ollama server not ready yet, waiting..."
    sleep 2
done

echo "Checking for llama3.2:1b model..."
if ! ollama list | grep -q "llama3.2:1b"; then
    echo "Pulling llama3.2:1b model in the background..."
    ollama pull llama3.2:1b &
else
    echo "llama3.2:1b model already exists."
fi

echo "Keeping Ollama server running..."
wait  # Wait for background processes to complete