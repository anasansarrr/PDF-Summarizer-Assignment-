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
from typing import List, Tuple, Optional

# Initialize model with optimized settings for phi3.5:3.8b - OPTIMIZED FOR SMALLER MODEL
try:
    llm = Ollama(
        model="phi3.5:3.8b", 
        temperature=0.3,  # Slightly higher for phi3.5 to improve creativity
        **{
            "num_predict": 2048,     # Reduced for smaller model efficiency
            "top_k": 40,            # Increased for phi3.5 diversity
            "top_p": 0.85,          # Optimal for phi3.5
            "num_ctx": 4096,        # Reduced context for better performance
            "repeat_penalty": 1.1,  # Lower for phi3.5 to avoid over-penalization
            "stop": ["Human:", "Assistant:", "Q:", "A:", "###", "---"],  # Clear stop tokens for phi3.5
            "num_thread": 4,        # Reduced for smaller model
            "mirostat": 2,          # Enable mirostat for phi3.5 coherence
            "mirostat_eta": 0.1,    # Fine-tune mirostat
            "mirostat_tau": 5.0,    # Target perplexity
        }
    )
except Exception as e:
    print(f"Warning: Some Ollama parameters not supported. Using basic configuration: {e}")
    llm = Ollama(model="phi3.5:3.8b", temperature=0.3)

# Cache for processed documents to avoid reprocessing
document_cache = {}

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
        chunk_size=600,      # Smaller chunks for phi3.5 processing
        chunk_overlap=100,   # Reduced overlap for efficiency
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

def smart_chunk_text(text: str, max_chunks: int = 100) -> List[str]:
    """OPTIMIZED: Intelligent chunking for phi3.5 - smaller max chunks"""
    splitter = get_text_splitter()
    initial_chunks = splitter.split_text(text)
    
    # If we have too many chunks, increase chunk size dynamically
    if len(initial_chunks) > max_chunks:
        # Calculate new chunk size to get closer to max_chunks
        new_chunk_size = min(1200, int(len(text) / max_chunks * 1.1))  # Smaller for phi3.5
        
        dynamic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=new_chunk_size,
            chunk_overlap=150,  # Reduced overlap
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
        if len(chunk) < 80:  # Slightly smaller threshold for phi3.5
            continue
            
        # Skip chunks that are mostly numbers or symbols
        if len(re.sub(r'[^a-zA-Z\s]', '', chunk)) < len(chunk) * 0.5:
            continue
            
        # Skip chunks that are mostly whitespace
        if len(chunk.replace(' ', '').replace('\n', '').replace('\t', '')) < 40:  # Reduced threshold
            continue
            
        quality_chunks.append(chunk)
    
    return quality_chunks[:max_chunks]  # Ensure we don't exceed max

class ImprovedEmbeddings(FakeEmbeddings):
    """OPTIMIZED: Embeddings optimized for phi3.5 processing"""
    
    def __init__(self, size=384):  # Smaller embedding size for phi3.5
        super().__init__(size=size)
        self._cache = {}
        # Pre-compute some common word vectors for better consistency
        self._word_vectors = {}
    
    def _get_text_features(self, text: str) -> dict:
        """Extract features from text for better embeddings"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'paragraph_count': len(text.split('\n\n')),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
        }
        return features
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate more sophisticated embedding for text with caching"""
        # Use first 200 chars for hashing for phi3.5 efficiency
        text_hash = hashlib.md5(text[:200].encode()).hexdigest()
        
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        # Get text features
        features = self._get_text_features(text)
        
        # Create base embedding from multiple hash seeds
        embedding = np.zeros(self.size, dtype=np.float32)
        
        # Use different parts of text for different embedding dimensions
        text_parts = [
            text[:len(text)//3],      # Beginning
            text[len(text)//3:2*len(text)//3],  # Middle
            text[2*len(text)//3:],    # End
            ' '.join(text.split()[:8]),   # First 8 words (reduced for phi3.5)
            ' '.join(text.split()[-8:]),  # Last 8 words
        ]
        
        for i, part in enumerate(text_parts):
            if part:
                seed = sum(ord(c) for c in part) % (2**32)
                np.random.seed(seed)
                start_idx = i * (self.size // len(text_parts))
                end_idx = (i + 1) * (self.size // len(text_parts))
                embedding[start_idx:end_idx] = np.random.rand(end_idx - start_idx)
        
        # Incorporate text features into embedding
        feature_vector = np.array([
            features['length'] / 1000.0,
            features['word_count'] / 100.0,
            features['sentence_count'] / 10.0,
            features['avg_word_length'] / 10.0,
            features['uppercase_ratio'],
            features['digit_ratio']
        ])
        
        # Blend feature vector into embedding
        embedding[:len(feature_vector)] += feature_vector * 0.1
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Cache result (with size limit)
        if len(self._cache) < 300:  # Smaller cache for phi3.5
            self._cache[text_hash] = embedding
            
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed documents with threading for speed"""
        if len(texts) < 15:  # Lower threshold for phi3.5
            return [self._get_embedding(text) for text in texts]
        
        # For larger batches, use threading with fewer workers
        with ThreadPoolExecutor(max_workers=2) as executor:
            embeddings = list(executor.map(self._get_embedding, texts))
        
        return embeddings
    
    def embed_query(self, text: str) -> np.ndarray:
        return self._get_embedding(text)

def create_vectorstore_optimized(chunks: List[str]) -> Tuple[FAISS, List[str]]:
    """OPTIMIZED: Create vectorstore with better chunk selection for phi3.5"""
    try:
        # Better chunk selection strategy - smaller limit for phi3.5
        if len(chunks) > 100:
            # Score chunks by information density
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                score = calculate_chunk_importance(chunk, i, len(chunks))
                scored_chunks.append((score, chunk))
            
            # Sort by score and take top chunks
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            chunks = [chunk for _, chunk in scored_chunks[:100]]
        
        embedder = ImprovedEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedder)
        
        return vectorstore, chunks
        
    except Exception as e:
        raise Exception(f"Vectorstore creation failed: {str(e)}")

def calculate_chunk_importance(chunk: str, position: int, total_chunks: int) -> float:
    """Calculate importance score for a chunk"""
    score = 0.0
    
    # Length score (prefer substantial chunks)
    length_score = min(len(chunk) / 800.0, 1.0)  # Adjusted for smaller chunks
    score += length_score * 0.3
    
    # Position score (prefer chunks from beginning and end)
    if position < total_chunks * 0.2 or position > total_chunks * 0.8:
        score += 0.2
    
    # Content quality score
    word_count = len(chunk.split())
    if word_count > 40:  # Reduced threshold for phi3.5
        score += 0.3
    
    # Information density (prefer chunks with varied vocabulary)
    unique_words = len(set(chunk.lower().split()))
    if word_count > 0:
        density = unique_words / word_count
        score += density * 0.2
    
    return score

def build_improved_qa_chain(llm, vectorstore):
    """OPTIMIZED: Build QA system with phi3.5-optimized prompt template"""
    
    # Custom prompt template optimized for phi3.5
    prompt_template = """Use the context below to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer the question using only information from the context. Be specific and include important details like numbers, dates, and requirements. If the context doesn't have enough information, say so clearly.

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Simpler search for phi3.5
        search_kwargs={
            "k": 6,  # Retrieve fewer documents for phi3.5 efficiency
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

def count_sentences_improved(text: str) -> int:
    """IMPROVED: More accurate sentence counting"""
    # Clean the text first
    text = text.strip()
    
    # Handle common abbreviations by replacing them temporarily
    abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'i.e.', 'e.g.']
    temp_text = text
    for i, abbr in enumerate(abbreviations):
        temp_text = temp_text.replace(abbr, f'ABBREV{i}')
    
    # Split on sentence endings
    sentences = re.split(r'[.!?]+\s+', temp_text)
    
    # Restore abbreviations and filter valid sentences
    valid_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Restore abbreviations
        for i, abbr in enumerate(abbreviations):
            sentence = sentence.replace(f'ABBREV{i}', abbr)
        
        # Must have at least 8 characters and 2 words (more lenient for phi3.5)
        if len(sentence) >= 8 and len(sentence.split()) >= 2:
            valid_sentences.append(sentence)
    
    return len(valid_sentences)

def create_comprehensive_summary_prompt(num_sentences: int, doc_type: str = "document") -> str:
    """OPTIMIZED: Create phi3.5-friendly prompt based on requested sentence count"""
    
    if num_sentences <= 5:
        detail_level = "brief summary"
        instruction = "Focus on the most important points"
    elif num_sentences <= 15:
        detail_level = "comprehensive summary"
        instruction = "Include key details and important information"
    else:
        detail_level = "detailed summary"
        instruction = "Provide thorough analysis with specific details"
    
    prompt = f"""Create a {detail_level} of this {doc_type} using exactly {num_sentences} sentences.

Requirements:
- Write exactly {num_sentences} complete sentences
- Each sentence should be informative and substantial
- Include specific facts, numbers, dates, and requirements when available
- Use clear, professional language
- {instruction}

Write your {num_sentences}-sentence summary:"""

    return prompt

def progressive_summarize_improved(qa_function, text: str, num_sentences: int = 10, include_header: bool = True) -> str:
    """OPTIMIZED: Progressive summarization optimized for phi3.5"""
    
    # Strategy 1: Try direct approach first
    prompt = create_comprehensive_summary_prompt(num_sentences)
    
    max_attempts = 2  # Reduced attempts for phi3.5 efficiency
    best_summary = ""
    best_sentence_count = 0
    
    for attempt in range(max_attempts):
        try:
            # Generate summary
            raw_summary = qa_function(prompt)
            
            # Clean and process the summary
            processed_summary = clean_and_validate_summary(raw_summary, num_sentences)
            sentence_count = count_sentences_improved(processed_summary)
            
            # Check if this is our best attempt so far
            if abs(sentence_count - num_sentences) < abs(best_sentence_count - num_sentences):
                best_summary = processed_summary
                best_sentence_count = sentence_count
            
            # If we got close enough, use it - more lenient for phi3.5
            if abs(sentence_count - num_sentences) <= 3:
                break
                
            # Modify prompt for next attempt
            if sentence_count < num_sentences:
                prompt = f"""Write exactly {num_sentences} sentences. Count as you write.

{prompt}

Remember: Write {num_sentences} sentences, no more, no less."""
            else:
                prompt = f"""You wrote too many sentences. Write exactly {num_sentences} sentences.

{prompt}

Stop after {num_sentences} sentences."""
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    # If we still don't have enough sentences, try supplemental approach
    if best_sentence_count < num_sentences * 0.7:  # More lenient threshold
        best_summary = supplement_summary(qa_function, best_summary, num_sentences, text)
    
    # Format final output
    if include_header:
        header = f"## Summary ({num_sentences} Key Points)\n\n"
        return header + best_summary
    
    return best_summary

def clean_and_validate_summary(raw_summary: str, target_sentences: int) -> str:
    """OPTIMIZED: Cleaning and validation optimized for phi3.5 output"""
    
    # Remove common prefixes - simpler patterns for phi3.5
    prefixes_to_remove = [
        r'^(?:Here is|This is|The following is).*?summary.*?:?\s*',
        r'^(?:Summary|Overview).*?:?\s*',
    ]
    
    cleaned = raw_summary.strip()
    for pattern in prefixes_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Handle abbreviations by replacing them temporarily
    abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'i.e.', 'e.g.']
    temp_cleaned = cleaned
    for i, abbr in enumerate(abbreviations):
        temp_cleaned = temp_cleaned.replace(abbr, f'ABBREV{i}')
    
    # Split into sentences more carefully
    sentence_parts = re.split(r'([.!?])\s+', temp_cleaned)
    
    sentences = []
    i = 0
    while i < len(sentence_parts):
        if i + 1 < len(sentence_parts) and sentence_parts[i+1] in '.!?':
            # Combine sentence with its punctuation
            sentence = (sentence_parts[i] + sentence_parts[i+1]).strip()
            i += 2
        else:
            sentence = sentence_parts[i].strip()
            i += 1
        
        # Restore abbreviations
        for j, abbr in enumerate(abbreviations):
            sentence = sentence.replace(f'ABBREV{j}', abbr)
        
        # More lenient filtering for phi3.5
        if len(sentence) >= 10 and len(sentence.split()) >= 3:
            # Ensure proper capitalization
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]
            
            # Ensure proper ending punctuation
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
                
            sentences.append(sentence)
    
    # Join sentences with proper spacing
    return ' '.join(sentences)

def supplement_summary(qa_function, existing_summary: str, target_sentences: int, full_text: str) -> str:
    """Add additional sentences if the summary is too short"""
    
    current_count = count_sentences_improved(existing_summary)
    needed = target_sentences - current_count
    
    if needed <= 0:
        return existing_summary
    
    # Ask for additional specific information - simplified for phi3.5
    supplement_prompt = f"""Add {needed} more sentences to complete the summary.

Current summary: {existing_summary}

Write {needed} additional sentences with important details not mentioned above:"""
    
    try:
        additional_content = qa_function(supplement_prompt)
        additional_sentences = clean_and_validate_summary(additional_content, needed)
        
        if additional_sentences:
            return existing_summary + ' ' + additional_sentences
    except:
        pass
    
    return existing_summary

# Alternative direct summarization function - OPTIMIZED FOR PHI3.5
def direct_summarize_improved(text: str, num_sentences: int = 10, include_header: bool = True) -> str:
    """OPTIMIZED: Direct summarization for phi3.5"""
    
    # Truncate very long texts intelligently - smaller limit for phi3.5
    max_chars = 8000  # Reduced for phi3.5 context limits
    if len(text) > max_chars:
        # Take strategic portions: beginning and end
        half = len(text) // 2
        truncated = text[:max_chars//2] + "\n\n[...content continues...]\n\n" + text[-max_chars//2:]
        text = truncated
    
    prompt = create_comprehensive_summary_prompt(num_sentences, "content")
    full_prompt = f"{prompt}\n\nContent:\n{text}\n\nYour {num_sentences}-sentence summary:"
    
    try:
        raw_summary = llm(full_prompt)
        processed_summary = clean_and_validate_summary(raw_summary, num_sentences)
        
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
    if len(document_cache) > 3:  # Keep fewer documents for phi3.5
        # Remove oldest entries
        keys_to_remove = list(document_cache.keys())[:-3]
        for key in keys_to_remove:
            del document_cache[key]
    
    gc.collect()

# Main functions with improved implementations
extract_text = extract_text_optimized
chunk_text = smart_chunk_text  
embed_chunks = create_vectorstore_optimized
build_rag_qa = build_improved_qa_chain
summarize = progressive_summarize_improved

# Additional utility function for testing different approaches
def compare_summarization_methods(text: str, num_sentences: int = 10):
    """Compare different summarization approaches"""
    print("Testing Direct Summarization...")
    direct_result = direct_summarize_improved(text, num_sentences, include_header=False)
    direct_count = count_sentences_improved(direct_result)
    
    print("Testing RAG Summarization...")
    try:
        chunks = chunk_text(text)
        vectorstore, selected_chunks = embed_chunks(chunks) 
        qa_function = build_rag_qa(llm, vectorstore)
        rag_result = summarize(qa_function, text, num_sentences, include_header=False)
        rag_count = count_sentences_improved(rag_result)
        
        print(f"\nDirect Method - Sentences: {direct_count}/{num_sentences}")
        print(f"RAG Method - Sentences: {rag_count}/{num_sentences}")
        print(f"\nDirect Result:\n{direct_result}")
        print(f"\nRAG Result:\n{rag_result}")
        
    except Exception as e:
        print(f"RAG method failed: {e}")
        print(f"\nDirect Result ({direct_count} sentences):\n{direct_result}")