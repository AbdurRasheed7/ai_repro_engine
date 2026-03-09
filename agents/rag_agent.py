from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS
import re

# ── Domain-aware retrieval queries ─────────────────────────
DOMAIN_QUERIES = {
    "ml": [
        "model architecture layers neural network convolution",
        "training procedure optimizer learning rate",
        "dataset preprocessing augmentation",
        "hyperparameters batch size epochs",
        "experimental results accuracy performance",
        "learning rate specific value number",
        "optimizer Adam SGD RMSprop momentum",
        "weight decay regularization dropout value",
    ],
    "nlp": [
        "model architecture transformer attention heads",
        "tokenizer vocabulary embedding dimension",
        "training optimizer learning rate warmup",
        "dataset text classification benchmark",
        "hyperparameters batch size epochs sequence length",
        "accuracy F1 score BLEU perplexity results",
        "dropout regularization weight decay",
        "feed forward hidden size layers",
    ],
    "recommendation": [
        "matrix factorization collaborative filtering",
        "embedding dimension latent factors",
        "training optimizer learning rate",
        "dataset MovieLens user item interactions",
        "hyperparameters batch size epochs",
        "RMSE MAE accuracy evaluation metric",
        "regularization weight decay dropout",
        "neural collaborative filtering architecture",
    ],
    "rl": [
        "reinforcement learning policy network",
        "reward discount factor gamma",
        "replay buffer batch size episodes",
        "epsilon greedy exploration",
        "Q-learning DQN actor critic",
        "environment state action space",
        "training steps learning rate optimizer",
        "performance reward convergence results",
    ],
    "graph": [
        "graph neural network node classification",
        "adjacency matrix node features",
        "graph convolution layer aggregation",
        "training optimizer learning rate",
        "dataset Cora Citeseer nodes edges",
        "hyperparameters epochs hidden size",
        "accuracy classification results benchmark",
        "dropout regularization weight decay",
    ],
    "algorithm": [
        "algorithm complexity time space",
        "implementation steps procedure",
        "dataset benchmark evaluation",
        "hyperparameters configuration settings",
        "results performance comparison",
        "accuracy precision recall F1",
        "training testing procedure",
        "optimization objective loss function",
    ],
}

# Default queries used when domain is unknown
DEFAULT_QUERIES = DOMAIN_QUERIES["ml"]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, respecting sentence boundaries"""
    if not text.strip():
        return []

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

    # Apply overlap
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
    """Retrieve most relevant chunks for given queries, deduplicated"""
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


def get_relevant_context(text, domain="ml"):
    """Main function — builds RAG and retrieves domain-aware context"""
    if len(text) < 500:
        print("⚠️ Paper text too short — skipping RAG")
        return text

    vectorstore = build_vectorstore(text)

    queries = DOMAIN_QUERIES.get(domain, DEFAULT_QUERIES)
    print(f"   Using {domain} domain queries ({len(queries)} queries)")

    relevant = retrieve_relevant_chunks(vectorstore, queries)
    return relevant