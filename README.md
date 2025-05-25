# Document Summarization Tool

A Python tool for extracting text from documents and generating intelligent summaries using Ollama's phi3.5 model with RAG (Retrieval-Augmented Generation).

## Features

- **Multi-format Support**: PDF, DOCX, Excel files
- **RAG-based Summarization**: Uses vector embeddings for context-aware summaries
- **Optimized for phi3.5**: Specifically tuned for Ollama's phi3.5:3.8b model
- **Intelligent Chunking**: Smart text segmentation for better processing
- **Memory Efficient**: Optimized for local execution

## Prerequisites

1. **Python 3.8+**
2. **Ollama installed and running**
3. **phi3.5:3.8b model downloaded**

# Install Ollama

## On macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

## On Windows
### Download from https://ollama.ai/download

### Download the model
ollama pull phi3.5:3.8b


# Create virtual environment
python -m venv venv
### On Windows
venv\Scripts\activate
### On macOS/Linux
source venv/bin/activate


# Install dependencies
pip install -r requirements.txt