import os
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import re

# Use more stable LangChain imports
try:
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import FakeEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    # Fallback to older imports if community package not available
    from langchain.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import FakeEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain

# For document extraction - optimized imports
try:
    import fitz  # PyMuPDF - faster than PyPDF2
    PYMUPDF_AVAILABLE = True
except ImportError:
    print("PyMuPDF not available, falling back to PyPDF2")
    from PyPDF2 import PdfReader
    PYMUPDF_AVAILABLE = False

import docx
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union

# Initialize model with optimized settings for phi3.5:3.8b - OPTIMIZED FOR LENGTH CONTROL
try:
    llm = Ollama(
        model="phi3.5:3.8b", 
        temperature=0.2,  # Lower temperature for more consistent output
        **{
            "num_predict": 1024,     # Reduced to prevent overly long responses
            "top_k": 20,            # Reduced for more focused generation
            "top_p": 0.8,           # Slightly lower for consistency
            "num_ctx": 3072,        # Smaller context for faster processing
            "repeat_penalty": 1.15, # Higher to avoid repetition
            "stop": ["Human:", "Assistant:", "Q:", "A:", "###", "---", "\n\n\n"],  # More stop tokens
            "num_thread": 4,        
            "mirostat": 2,          
            "mirostat_eta": 0.1,    
            "mirostat_tau": 4.0,    # Lower target perplexity for consistency
        }
    )
except Exception as e:
    print(f"Warning: Some Ollama parameters not supported. Using basic configuration: {e}")
    llm = Ollama(model="phi3.5:3.8b", temperature=0.2)

# Cache for processed documents to avoid reprocessing
document_cache = {}

def validate_num_sentences(num_sentences: Union[int, str]) -> int:
    """Validate and convert num_sentences to integer with proper error handling"""
    if isinstance(num_sentences, str):
        # Check if the string is actually meant to be text content
        if len(num_sentences) > 20 or not num_sentences.strip().isdigit():
            print(f"Warning: Expected number but got text content: {num_sentences[:50]}...")
            return 10  # Default fallback value
        try:
            return int(num_sentences.strip())
        except ValueError:
            print(f"Warning: Could not convert '{num_sentences}' to integer. Using default value 10.")
            return 10
    elif isinstance(num_sentences, (int, float)):
        return int(num_sentences)
    else:
        print(f"Warning: Unexpected type for num_sentences: {type(num_sentences)}. Using default value 10.")
        return 10

def get_file_hash(file_path: str) -> str:
    """Generate a hash for the file to use as cache key"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

@lru_cache(maxsize=32)
def get_text_splitter():
    """Cached text splitter - OPTIMIZED FOR PHI3.5"""
    return RecursiveCharacterTextSplitter(
        chunk_size=500,      # Smaller chunks for faster processing
        chunk_overlap=80,    # Reduced overlap for efficiency
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        keep_separator=True
    )

def extract_text_optimized(file_path: str) -> str:
    """Optimized text extraction with better PDF handling"""
    try:
        file_hash = get_file_hash(file_path)
        
        # Check cache first
        if file_hash in document_cache:
            return document_cache[file_hash]
        
        text = ""
        
        if file_path.endswith(".pdf"):
            if PYMUPDF_AVAILABLE:
                # Use PyMuPDF which is much faster than PyPDF2
                doc = fitz.open(file_path)
                
                # Process pages in batches for memory efficiency
                batch_size = 10
                for i in range(0, len(doc), batch_size):
                    batch_text = []
                    for page_num in range(i, min(i + batch_size, len(doc))):
                        page = doc[page_num]
                        page_text = page.get_text()
                        if page_text.strip():  # Only add non-empty pages
                            # Clean up the text
                            page_text = re.sub(r'\n+', '\n', page_text)  # Remove excessive newlines
                            batch_text.append(page_text)
                        page = None  # Explicit cleanup
                    
                    text += "\n".join(batch_text)
                    batch_text.clear()
                    
                    # Force garbage collection after each batch
                    if i % (batch_size * 2) == 0:
                        gc.collect()
                
                doc.close()
            else:
                # Fallback to PyPDF2
                reader = PdfReader(file_path)
                text_parts = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Clean up the text
                        page_text = re.sub(r'\n+', '\n', page_text)
                        text_parts.append(page_text)
                    
                    # Periodic cleanup for large documents
                    if i % 20 == 0 and i > 0:
                        gc.collect()
                
                text = "\n".join(text_parts)
            
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            # Process paragraphs in batches
            batch_size = 100
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            for i in range(0, len(paragraphs), batch_size):
                batch = paragraphs[i:i + batch_size]
                text += "\n".join(batch)
                
        elif file_path.endswith((".xlsx", ".xls")):
            # Optimize Excel reading
            try:
                df_dict = pd.read_excel(file_path, sheet_name=None, nrows=1000)  # Limit rows
                text_parts = []
                for sheet_name, df in df_dict.items():
                    # Only include non-empty sheets
                    if not df.empty:
                        text_parts.append(f"Sheet: {sheet_name}\n{df.head(100).to_string()}")
                text = "\n\n".join(text_parts)
            except Exception as e:
                return f"Error reading Excel file: {str(e)}"
        else:
            return "Unsupported file format"
        
        # Clean up the text before caching
        text = clean_extracted_text(text)
        
        # Cache the result
        document_cache[file_hash] = text
        return text
        
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def clean_extracted_text(text: str) -> str:
    """Clean extracted text for better processing"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove page numbers and headers/footers patterns
    text = re.sub(r'\n\d+\s*\n', '\n', text)
    text = re.sub(r'\nPage \d+.*?\n', '\n', text, flags=re.IGNORECASE)
    
    # Fix common OCR/extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    
    return text.strip()

def smart_chunk_text(text: str, max_chunks: int = 80) -> List[str]:
    """OPTIMIZED: Intelligent chunking for phi3.5 - reduced max chunks"""
    splitter = get_text_splitter()
    initial_chunks = splitter.split_text(text)
    
    # If we have too many chunks, increase chunk size dynamically
    if len(initial_chunks) > max_chunks:
        # Calculate new chunk size to get closer to max_chunks
        new_chunk_size = min(1000, int(len(text) / max_chunks * 1.1))
        
        dynamic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=new_chunk_size,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
            keep_separator=True
        )
        chunks = dynamic_splitter.split_text(text)
    else:
        chunks = initial_chunks
    
    # IMPROVED: Better chunk filtering and quality control
    quality_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        
        # Skip very short chunks
        if len(chunk) < 60:
            continue
            
        # Skip chunks that are mostly numbers or symbols
        if len(re.sub(r'[^a-zA-Z\s]', '', chunk)) < len(chunk) * 0.5:
            continue
            
        # Skip chunks that are mostly whitespace
        if len(chunk.replace(' ', '').replace('\n', '').replace('\t', '')) < 30:
            continue
            
        quality_chunks.append(chunk)
    
    return quality_chunks[:max_chunks]  # Ensure we don't exceed max

class ImprovedEmbeddings(FakeEmbeddings):
    """OPTIMIZED: Simplified embeddings for faster processing"""
    
    def __init__(self, size=256):  # Even smaller embedding size
        super().__init__(size=size)
        self._cache = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate simplified embedding for text with caching"""
        # Use first 150 chars for hashing for efficiency
        text_hash = hashlib.md5(text[:150].encode()).hexdigest()
        
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        # Simplified embedding generation
        embedding = np.zeros(self.size, dtype=np.float32)
        
        # Use text characteristics for embedding
        words = text.lower().split()[:20]  # Only first 20 words
        for i, word in enumerate(words):
            if i >= self.size // 4:
                break
            seed = sum(ord(c) for c in word) % (2**32)
            np.random.seed(seed)
            start_idx = i * 4
            end_idx = min(start_idx + 4, self.size)
            embedding[start_idx:end_idx] = np.random.rand(end_idx - start_idx)
        
        # Add text statistics
        if len(text) > 0:
            embedding[-4:] = [
                len(text) / 1000.0,
                len(words) / 50.0,
                text.count('.') / 10.0,
                sum(c.isupper() for c in text) / len(text)
            ]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Cache result (with size limit)
        if len(self._cache) < 200:
            self._cache[text_hash] = embedding
            
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed documents - simplified"""
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> np.ndarray:
        return self._get_embedding(text)

def create_vectorstore_optimized(chunks: List[str]) -> Tuple[FAISS, List[str]]:
    """OPTIMIZED: Create vectorstore with better chunk selection for phi3.5"""
    try:
        # Better chunk selection strategy - smaller limit for phi3.5
        if len(chunks) > 60:  # Reduced from 100
            # Score chunks by information density
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                score = calculate_chunk_importance(chunk, i, len(chunks))
                scored_chunks.append((score, chunk))
            
            # Sort by score and take top chunks
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            chunks = [chunk for _, chunk in scored_chunks[:60]]
        
        embedder = ImprovedEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedder)
        
        return vectorstore, chunks
        
    except Exception as e:
        raise Exception(f"Vectorstore creation failed: {str(e)}")

def calculate_chunk_importance(chunk: str, position: int, total_chunks: int) -> float:
    """Calculate importance score for a chunk"""
    score = 0.0
    
    # Length score (prefer substantial chunks)
    length_score = min(len(chunk) / 600.0, 1.0)
    score += length_score * 0.4
    
    # Position score (prefer chunks from beginning and end)
    if position < total_chunks * 0.25 or position > total_chunks * 0.75:
        score += 0.3
    
    # Content quality score
    word_count = len(chunk.split())
    if word_count > 30:
        score += 0.3
    
    return score

def build_improved_qa_chain(llm, vectorstore):
    """OPTIMIZED: Build QA system with strict length control"""
    
    # Simplified prompt template with strict length control
    prompt_template = """Context: {context}

Question: {question}

Instructions: Answer using ONLY the context above. Be concise and factual.

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,  # Retrieve even fewer documents for efficiency
        }
    )
    
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=PROMPT,
        verbose=False
    )
    
    # Create a custom QA function
    def qa_function(query):
        docs = retriever.get_relevant_documents(query)
        return qa_chain.run(input_documents=docs, question=query)
    
    return qa_function

def count_sentences_fast(text: str) -> int:
    """FAST: Quick sentence counting"""
    # Simple but effective sentence counting
    text = text.strip()
    if not text:
        return 0
    
    # Count sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    return len(valid_sentences)

def create_length_controlled_prompt(num_sentences: Union[int, str]) -> str:
    """Create prompt with strict length control"""
    
    # Validate and convert num_sentences
    num_sentences = validate_num_sentences(num_sentences)
    
    # Different strategies based on sentence count
    if num_sentences <= 3:
        instruction = "Write exactly 3 short, key sentences"
    elif num_sentences <= 7:
        instruction = f"Write exactly {num_sentences} sentences. Each sentence should be 15-25 words"
    else:
        instruction = f"Write exactly {num_sentences} sentences. Mix short (10-15 words) and longer (20-30 words) sentences"
    
    prompt = f"""Summarize the key information in exactly {num_sentences} sentences.

CRITICAL RULES:
1. Write EXACTLY {num_sentences} sentences - no more, no less
2. {instruction}
3. Each sentence must end with a period
4. Number your sentences: 1. 2. 3. etc.
5. Focus on the most important facts and details

Write your numbered {num_sentences} sentences:"""

    return prompt

def fast_summarize(qa_function, num_sentences: Union[int, str] = 10, include_header: bool = True) -> str:
    """FAST: Single-attempt summarization with strict length control"""
    
    # Validate and convert num_sentences
    num_sentences = validate_num_sentences(num_sentences)
    
    prompt = create_length_controlled_prompt(num_sentences)
    
    try:
        # Single generation attempt
        raw_summary = qa_function(prompt)
        
        # Extract numbered sentences
        processed_summary = extract_numbered_sentences(raw_summary, num_sentences)
        
        # Validate and adjust if needed
        sentence_count = count_sentences_fast(processed_summary)
        
        # If we're off by a lot, try a quick fix
        if abs(sentence_count - num_sentences) > 2:
            processed_summary = quick_length_fix(processed_summary, num_sentences)
        
        if include_header:
            header = f"## Summary ({num_sentences} Key Points)\n\n"
            return header + processed_summary
        
        return processed_summary
        
    except Exception as e:
        return f"Error in summarization: {str(e)}"

def extract_numbered_sentences(text: str, target_count: Union[int, str]) -> str:
    """Extract numbered sentences from the response"""
    
    # Validate and convert target_count
    target_count = validate_num_sentences(target_count)
    
    # Look for numbered sentences
    numbered_pattern = r'(\d+\.)\s*([^.!?]*[.!?])'
    matches = re.findall(numbered_pattern, text)
    
    if matches and len(matches) >= target_count * 0.7:  # If we found reasonable numbered sentences
        sentences = []
        for i, (num, sentence) in enumerate(matches[:target_count]):
            sentence = sentence.strip()
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            sentences.append(sentence)
        return ' '.join(sentences)
    
    # Fallback: extract regular sentences
    return extract_regular_sentences(text, target_count)

def extract_regular_sentences(text: str, target_count: Union[int, str]) -> str:
    """Extract regular sentences when numbered format fails"""
    
    # Validate and convert target_count
    target_count = validate_num_sentences(target_count)
    
    # Clean the text
    text = re.sub(r'^(Here is|This is|The following).*?summary.*?:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)  # Remove numbering
    
    # Split into sentences
    sentences = re.split(r'[.!?]+\s+', text.strip())
    
    # Filter and clean sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence.split()) > 3:
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            clean_sentences.append(sentence)
    
    # Take the target number of sentences
    return ' '.join(clean_sentences[:target_count])

def quick_length_fix(summary: str, target_sentences: Union[int, str]) -> str:
    """Quick fix for length issues"""
    
    # Validate and convert target_sentences
    target_sentences = validate_num_sentences(target_sentences)
    
    current_sentences = count_sentences_fast(summary)
    
    if current_sentences > target_sentences:
        # Truncate to target length
        sentences = re.split(r'[.!?]+\s+', summary.strip())
        return ' '.join(sentences[:target_sentences]) + '.'
    
    elif current_sentences < target_sentences and current_sentences > 0:
        # For now, just return what we have rather than trying to expand
        return summary
    
    return summary

# Direct summarization function - FAST VERSION
def direct_summarize_fast(text: str, num_sentences: Union[int, str] = 10, include_header: bool = True) -> str:
    """FAST: Direct summarization optimized for speed and length control"""
    
    # Validate and convert num_sentences
    num_sentences = validate_num_sentences(num_sentences)
    
    # Truncate very long texts more aggressively
    max_chars = 6000  # Reduced for faster processing
    if len(text) > max_chars:
        # Take just the beginning and a bit from the end
        text = text[:max_chars//2] + "\n\n" + text[-max_chars//4:]
    
    prompt = create_length_controlled_prompt(num_sentences)
    full_prompt = f"{prompt}\n\nContent to summarize:\n{text}\n\nYour numbered {num_sentences} sentences:"
    
    try:
        raw_summary = llm(full_prompt)
        processed_summary = extract_numbered_sentences(raw_summary, num_sentences)
        
        if include_header:
            header = f"## Summary ({num_sentences} Key Points)\n\n"
            return header + processed_summary
        return processed_summary
    except Exception as e:
        return f"Error in direct summarization: {str(e)}"

# Cleanup function to manage memory
def cleanup_resources():
    """Clean up resources and cache"""
    global document_cache
    if len(document_cache) > 2:  # Keep even fewer documents
        # Remove oldest entries
        keys_to_remove = list(document_cache.keys())[:-2]
        for key in keys_to_remove:
            del document_cache[key]
    
    gc.collect()

# Main functions with improved implementations
extract_text = extract_text_optimized
chunk_text = smart_chunk_text  
embed_chunks = create_vectorstore_optimized
build_rag_qa = build_improved_qa_chain
summarize = fast_summarize

# Simplified comparison function
def compare_summarization_methods(text: str, num_sentences: Union[int, str] = 10):
    """Compare different summarization approaches - simplified"""
    
    # Validate and convert num_sentences
    num_sentences = validate_num_sentences(num_sentences)
    
    print("Testing Direct Summarization...")
    direct_result = direct_summarize_fast(text, num_sentences, include_header=False)
    direct_count = count_sentences_fast(direct_result)
    
    print("Testing RAG Summarization...")
    try:
        chunks = chunk_text(text)
        vectorstore, selected_chunks = embed_chunks(chunks) 
        qa_function = build_rag_qa(llm, vectorstore)
        rag_result = summarize(qa_function, num_sentences, include_header=False)
        rag_count = count_sentences_fast(rag_result)
        
        print(f"\nDirect Method - Sentences: {direct_count}/{num_sentences}")
        print(f"RAG Method - Sentences: {rag_count}/{num_sentences}")
        print(f"\nDirect Result:\n{direct_result}")
        print(f"\nRAG Result:\n{rag_result}")
        
    except Exception as e:
        print(f"RAG method failed: {e}")
        print(f"\nDirect Result ({direct_count} sentences):\n{direct_result}")

# Additional utility for precise length control
def generate_exact_length_summary(text: str, num_sentences: Union[int, str], method: str = "direct") -> str:
    """Generate summary with exact sentence count"""
    
    # Validate and convert num_sentences
    num_sentences = validate_num_sentences(num_sentences)
    
    if method == "direct":
        result = direct_summarize_fast(text, num_sentences, include_header=False)
    else:
        try:
            chunks = chunk_text(text)
            vectorstore, _ = embed_chunks(chunks)
            qa_function = build_rag_qa(llm, vectorstore)
            result = summarize(qa_function, num_sentences, include_header=False)
        except Exception as e:
            # Fallback to direct method
            result = direct_summarize_fast(text, num_sentences, include_header=False)
    
    # Final length check and adjustment
    current_count = count_sentences_fast(result)
    if current_count != num_sentences:
        print(f"Warning: Generated {current_count} sentences instead of {num_sentences}")
    
    return result

# Safe wrapper functions for backward compatibility
def safe_summarize(qa_function, num_sentences=10, include_header=True):
    """Safe wrapper for summarize function"""
    try:
        return fast_summarize(qa_function, num_sentences, include_header)
    except Exception as e:
        print(f"Error in summarization: {e}")
        return f"Error: Could not generate summary. {str(e)}"

def safe_direct_summarize(text, num_sentences=10, include_header=True):
    """Safe wrapper for direct summarization"""
    try:
        return direct_summarize_fast(text, num_sentences, include_header)
    except Exception as e:
        print(f"Error in direct summarization: {e}")
        return f"Error: Could not generate summary. {str(e)}"