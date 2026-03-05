from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS
import re

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, trying to respect sentence boundaries"""
    if not text.strip():
        return []
    
    # First split on double newlines (paragraphs), then finer split
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            
            # If para too long, split on sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= chunk_size:
                        current_chunk += sent + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Apply overlap by duplicating last overlap chars from previous
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            overlap_text = chunks[i-1][-overlap:]
            chunk = overlap_text + chunk
        overlapped_chunks.append(chunk)
    
    print(f"✅ Created {len(overlapped_chunks)} chunks")
    return overlapped_chunks

def build_vectorstore(text):
    """Build FAISS vectorstore with metadata"""
    print("🔍 Building FAISS vectorstore...")
    chunks = chunk_text(text)
    
    if not chunks:
        raise ValueError("No chunks created from text — check parsing")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Add metadata for traceability
    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": "paper_text"}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("✅ FAISS vectorstore ready!")
        return vectorstore
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        raise

def retrieve_relevant_chunks(vectorstore, queries):
    """Retrieve most relevant chunks for given queries + dedup"""
    all_docs = []
    seen = set()
    
    for query in queries:
        try:
            docs = vectorstore.similarity_search(query, k=TOP_K_CHUNKS)
            for doc in docs:
                content = doc.page_content
                if content not in seen:
                    seen.add(content)
                    all_docs.append(content)
        except Exception as e:
            print(f"Warning: retrieval for query '{query}' failed: {e}")
    
    combined = "\n\n".join(all_docs)
    print(f"✅ RAG retrieved {len(combined):,} relevant characters")
    return combined

def get_relevant_context(text):
    """Main function - builds RAG and retrieves relevant context"""
    if len(text) < 500:
        print("⚠️ Paper text too short — skipping RAG")
        return text
    
    vectorstore = build_vectorstore(text)
    
    # Core ML implementation queries (very good!)
    queries = [
        "model architecture layers neural network",
        "training procedure optimizer learning rate",
        "dataset preprocessing augmentation",
        "hyperparameters batch size epochs",
        "experimental results accuracy performance",
        "learning rate specific value number",
        "optimizer Adam SGD RMSprop momentum",
        "weight decay regularization dropout value",
    ]
    
    relevant = retrieve_relevant_chunks(vectorstore, queries)
    return relevant