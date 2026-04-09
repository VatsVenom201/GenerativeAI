import os
import warnings
import re
warnings.filterwarnings("ignore")
import shutil

if os.path.exists("vector_db"):
    shutil.rmtree("vector_db")
# ----------------- CACHE LOCATION -----------------
CACHE_DIR = r"D:\huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
print(f"Using shared cache: {CACHE_DIR}")

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ----------------- IMPORTS -----------------
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from transformers import logging
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

logging.set_verbosity_info()

# ----------------- GROQ API -----------------
groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------- LOAD PDF -----------------
pdf_paths = [
    "D:/PyCharm/GenAI/Chatbot/RAG-pipeline/Fundamental of Soil Sci by ISSS-1_Searchable.pdf",
]

# Patterns to remove anywhere in text
REMOVE_PATTERNS = [
    r"^\d+$",                      # page numbers
    r"^Table\s+\d+",                # Table references
    r"^Figure\s+\d+",               # Figure references
    r"INTRODUCTION",
    r"WEATHERING AND SOIL FORMATION",
    r"SOIL WATER",
    r"SOIL AIR AND SOIL TEMPERATURE",
    r"TILLAGE",
    r"WATER MANAGEMENT",
    r"SOIL EROSION AND SOIL CONSERVATION",
    r"FUNDAMENTALS OF SOIL SCIENCE",
    r"SOIL COLLOIDS AND ION EXCHANGE IN SOIL",
    r"SOIL SURVEY AND MAPPING",
    r"SOIL ACIDITY",
    r"SOIL SALINITY AND ALKALINITY",
    r"MINERAL NUTRITION OF PLANTS",
    r"SOIL CLASSIFICATION",
    r"NITROGEN",
    r"PHOSPHORUS",
    r"POTASSIUM",
    r"SECONDARY NUTRIENTS",
    r"MICRONUTRIENTS",
    r"ANALYSIS OF SOIL, PLANT AND FERTILIZER FOR PLANT NU1RIENTS",
    r"SOIL FERTILITY EVALUATION",
    r"SOIL BIOLOGY AND BIOCHEMISTRY",
    r"SOIL ORGANIC MATTER",
    r"FERTILIZERS, MANURES AND BIOFERTILIZERS",
    r"SOIL FERTILITY MANAGEMENT",
    r"SOIL AND WATER QUALITY",
    r"SOIL POLLUTION AND ITS CONTROL",
    r"SOIL MANAGEMENT FOR SUSTAINABLE FARMING",
    r"CHEMICAL COMPOSITION OF SOILS",
    r"PHYSICAL PROPERTIES OF SOILS"
]

# Sections to skip entirely
SKIP_SECTIONS = [
    r"^PREFACE",
    r"^REFERENCES",
    r"^CONTRIBUTORS"
]

# ----------------- CLEANING FUNCTIONS -----------------
def remove_unwanted_lines(text, patterns=REMOVE_PATTERNS):
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if any(re.search(pat, line_clean, re.IGNORECASE) for pat in patterns):
            continue
        cleaned_lines.append(line_clean)
    return "\n".join(cleaned_lines)

def skip_sections(text, section_patterns=SKIP_SECTIONS):
    lines = text.split("\n")
    cleaned_lines = []
    skip_flag = False
    for line in lines:
        line_strip = line.strip()
        if any(re.search(pat, line_strip, re.IGNORECASE) for pat in section_patterns):
            skip_flag = True
        # Stop skipping if a new normal line appears (heuristic)
        if skip_flag and len(line_strip) > 20:
            skip_flag = False
        if not skip_flag:
            cleaned_lines.append(line_strip)
    return "\n".join(cleaned_lines)

def is_noisy_line(line):
    line = line.strip()
    if not line:
        return True
    tokens = line.split()
    # Too many special characters
    special_chars = re.findall(r"[^a-zA-Z0-9\s]", line)
    if len(special_chars) / max(len(line), 1) > 0.3:
        return True
    # Too many numbers
    numbers = re.findall(r"\b\d+\b", line)
    if len(numbers) / max(len(tokens), 1) > 0.5:
        return True
    # Very low word content
    words = re.findall(r"\b[a-zA-Z]{3,}\b", line)
    if len(words) == 0:
        return True
    # Too short
    if len(line) < 20:
        return True
    return False

def clean_page(text):
    # Skip preface, references, contributors
    text = skip_sections(text)
    # Remove unwanted patterns
    text = remove_unwanted_lines(text)
    # Remove noisy lines
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if not is_noisy_line(line)]
    return "\n".join(cleaned_lines)

def clean_docs(docs):
    cleaned = []
    for doc in docs:
        text = clean_page(doc.page_content)
        if text.strip():
            doc.page_content = text
            cleaned.append(doc)
    return cleaned

def deduplicate_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        fingerprint = hashlib.md5(text.encode()).hexdigest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(doc)
    return unique

def normalize_text(text):
    text = re.sub(r"\n+", "\n", text)  # join broken lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # fix mid-sentence line breaks
    return text

def normalize_docs(docs):
    for doc in docs:
        doc.page_content = normalize_text(doc.page_content)
    return docs

# ----------------- LOAD AND CLEAN PDF -----------------
docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

docs = clean_docs(docs)
docs = deduplicate_docs(docs)
docs = normalize_docs(docs)

# ----------------- TEXT SPLITTING -----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[
        "\n\n",  # paragraphs first
        ". ",    # then sentences
    ]
)

docs = text_splitter.split_documents(docs)

# ----------------- EMBEDDINGS -----------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------- VECTOR DATABASE -----------------
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="vector_db"
)

# ----------------- RETRIEVER -----------------
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 60}
)

# ----------------- RERANKER -----------------
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank_documents(query, docs, top_n=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_n]]

def remove_duplicate_docs(docs):
    unique_docs = []
    seen = set()
    for doc in docs:
        fingerprint = doc.page_content.strip()[:200]
        if fingerprint not in seen:
            unique_docs.append(doc)
            seen.add(fingerprint)
    return unique_docs

def semantic_dedup(docs, embeddings=embedding_model, threshold=0.9):
    if not docs:
        return []
    texts = [doc.page_content for doc in docs]
    all_embeddings = embeddings.embed_documents(texts)
    unique_docs = []
    unique_vectors = []
    for doc, emb in zip(docs, all_embeddings):
        is_duplicate = False
        for v in unique_vectors:
            if cosine_similarity([emb], [v])[0][0] > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_docs.append(doc)
            unique_vectors.append(emb)
    return unique_docs

# ----------------- MAIN RAG FUNCTION -----------------
def response_llm(user_query, chat_history, image_context=None):
    candidate_docs = retriever.invoke(user_query)
    for i, doc in enumerate(candidate_docs):
        print(f"\nRetrieved {i}:\n{doc.page_content[:200]}")
    print("--"*50)
    print(f"\nRetrieved chunks (before dedup): {len(candidate_docs)}")

    unique_docs = remove_duplicate_docs(candidate_docs)
    unique_docs = semantic_dedup(unique_docs)
    print(f"Unique chunks after dedup: {len(unique_docs)}")

    retrieved_docs = rerank_documents(user_query, unique_docs, top_n=5)
    print(f"Chunks sent to reranker: {len(unique_docs)}")

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"Final Context to LLM:{context}")

    # -------- BUILD CHAT HISTORY --------
    history = [
        {
            "role": "system",
            "content": ("""You are a helpful assistant.Answer ONLY using the provided context.
If direct answer is not present, infer from related information in the context.
Do NOT say "Not found" if partial relevant information exists.
Only say "Not found" if absolutely no relevant information is present."""
                # "You are a helpful assistant. "
                # "Answer ONLY using the provided context. "
                # "If the answer is not present, say: Not found in context."
            )
        }
    ]

    MAX_HISTORY = 5
    recent_history = chat_history[-MAX_HISTORY:]

    for msg in recent_history:
        if msg.type == "human":
            history.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            history.append({"role": "assistant", "content": msg.content})

    prompt_text = f"Context:\n{context}\n\n"
    if image_context:
        prompt_text += f"Image Context:\n{image_context}\n\n"
    prompt_text += f"Question:{user_query}"

    history.append({"role": "user", "content": prompt_text})

    response = groq_client.chat.completions.create(
        messages=history,
        #model="llama-3.3-70b-versatile",
        model = "llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=500
    )

    output = response.choices[0].message.content

    sources = []
    for doc in retrieved_docs:
        sources.append({"content": doc.page_content, "metadata": doc.metadata})

    return output, sources
