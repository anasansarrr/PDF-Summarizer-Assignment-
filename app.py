import streamlit as st
import os
import tempfile
import time
from summarizer_phi import (
    extract_text, chunk_text, embed_chunks, build_rag_qa, 
    summarize, llm, cleanup_resources
)

# Configure Streamlit for better performance
st.set_page_config(
    page_title="Optimized RAG Summarizer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
.stProgress .st-bo {
    background-color: #00cc88;
}
.summary-box {
    background-color: #000000;
    color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #00cc88;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Optimized RAG-based PDF Summarizer")
st.markdown("*Optimized for large documents (50+ pages) with improved speed and memory efficiency*")

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    
    # Advanced options
    with st.expander("Advanced Options"):
        max_chunks = st.slider("Max Chunks (for very large docs)", 50, 300, 150, 
                              help="Limit chunks for faster processing of very large documents")
        
        chunk_strategy = st.selectbox("Chunk Strategy", 
                                    ["Adaptive", "Fixed"], 
                                    help="Adaptive adjusts chunk size based on document length")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your PDF document", 
        type=['pdf', 'docx', 'xlsx'],
        help="Optimized for large PDF files (50+ pages)"
    )

with col2:
    summary_type = st.selectbox(
        "Summary Length", 
        ["Short (8-10 sentences)", "Medium (15-20 sentences)", "Long (25-30 sentences)", "Detailed (40-50 sentences)"],
        index=1
    )

# Information section
if not uploaded_file:
    st.info("👆 Upload a document to get started")
    
    with st.expander("🔧 Optimization Features"):
        st.markdown("""
        **Speed Optimizations:**
        - 🔄 PyMuPDF for faster PDF processing (3-5x faster than PyPDF2)
        - 🧠 Intelligent chunking that adapts to document size
        - 📊 Batch processing for memory efficiency
        - 💝 Document caching to avoid reprocessing
        - 🎯 Optimized embeddings with reduced dimensionality
        
        **Memory Management:**
        - 📦 Batch processing of large documents
        - 🧹 Automatic garbage collection
        - 💾 Smart caching system
        - 🎚️ Configurable chunk limits
        
        **Quality Improvements:**
        - 🎯 Progressive summarization for different document sizes
        - 🔍 Better chunk selection for large documents
        - 📝 Enhanced prompt engineering
        """)

# Main processing
if uploaded_file:
    # Display file info
    file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # Size in MB
    st.info(f"📄 File: {uploaded_file.name} | Size: {file_size:.1f} MB")
    
    # Processing button
    if st.button("🔄 Process Document", type="primary"):
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            file_path = temp_file.name
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Extract text
            status_text.text("📖 Extracting text from document...")
            progress_bar.progress(10)
            
            raw_text = extract_text(file_path)
            
            if not raw_text or raw_text.startswith("Error"):
                st.error(f"❌ Failed to extract text: {raw_text}")
            else:
                # Display document stats
                word_count = len(raw_text.split())
                char_count = len(raw_text)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", f"{char_count:,}")
                with col2:
                    st.metric("Words", f"{word_count:,}")
                
                progress_bar.progress(25)
                
                # Step 2: Chunk text
                status_text.text("✂️ Chunking document...")
                chunks = chunk_text(raw_text, max_chunks=max_chunks)
                st.success(f"✅ Created {len(chunks)} chunks")
                progress_bar.progress(40)
                
                # Step 3: Create embeddings
                status_text.text("🧠 Creating embeddings...")
                vectorstore, chunk_list = embed_chunks(chunks)
                progress_bar.progress(60)
                
                # Step 4: Build QA system
                status_text.text("🔧 Building QA system...")
                qa = build_rag_qa(llm, vectorstore)
                progress_bar.progress(80)
                
                # Step 5: Generate summary
                status_text.text("📝 Generating summary...")
                num_sentences = int(summary_type.split("(")[1].split("-")[0]) if "-" in summary_type else int(summary_type.split("(")[1].replace(")", "").split()[0])
                
                summary = summarize(qa, raw_text, num_sentences)
                progress_bar.progress(100)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Display results
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Processing completed in {processing_time:.1f} seconds!")
                
                # Summary display
                st.markdown("---")
                st.subheader("📋 Document Summary")
                
                # Create a nice box for the summary
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                
                # Performance metrics
                with st.expander("📊 Processing Details"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Processing Time", f"{processing_time:.1f}s")
                    with col2:
                        st.metric("Chunks Processed", len(chunk_list))
                    with col3:
                        st.metric("Speed", f"{char_count/processing_time:.0f} chars/sec")
                    with col4:
                        st.metric("Efficiency", f"{word_count/processing_time:.0f} words/sec")
                
                # Download summary option
                st.download_button(
                    label="💾 Download Summary",
                    data=summary,
                    file_name=f"summary_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            
            # Debug information
            with st.expander("🐛 Debug Information"):
                st.code(str(e))
                st.markdown("""
                **Troubleshooting:**
                1. Ensure you have installed: `pip install PyMuPDF streamlit langchain faiss-cpu`
                2. Check that Ollama is running with llama3 model
                3. Try with a smaller document first
                """)
        
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
            cleanup_resources()
    
    # Reset button
    if st.button("🔄 Reset"):
        st.rerun()
