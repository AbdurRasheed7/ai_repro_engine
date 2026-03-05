from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def build_vectorstore(text):
    """Build FAISS vectorstore from text"""
    print("🔍 Building FAISS vectorstore...")
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("✅ FAISS vectorstore ready!")
    return vectorstore

def retrieve_relevant_chunks(vectorstore, queries):
    """Retrieve most relevant chunks for given queries"""
    all_docs = []
    seen = set()
    for query in queries:
        docs = vectorstore.similarity_search(query, k=TOP_K_CHUNKS)
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc.page_content)
    return "\n\n".join(all_docs)

def get_relevant_context(text):
    """Main function - builds RAG and retrieves relevant context"""
    vectorstore = build_vectorstore(text)
    
    # Query for most important ML paper sections
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
    print(f"✅ RAG retrieved {len(relevant)} relevant characters")
    return relevant