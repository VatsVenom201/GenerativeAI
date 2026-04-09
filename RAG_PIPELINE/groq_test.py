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
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# ----------------- IMPORTS -----------------
from groq import Groq
#import pytesseract
#from PIL import Image
import io

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from sentence_transformers import CrossEncoder

from transformers import logging
logging.set_verbosity_info() # enables verbose of hf models cache usage/download
# ----------------- GROQ API -----------------

groq_client = Groq(
    api_key=GROQ_API_KEY
)

# ----------------- LOAD PDF -----------------

pdf_paths = [
    "D:/PyCharm/GenAI/Chatbot/RAG-pipeline/Fundamental of Soil Sci by ISSS-1_Searchable.pdf",
]
# Patterns to remove anywhere in text
REMOVE_PATTERNS = [
    r"^\d+$",                      # lines with only numbers (page numbers)
    r"^Table\s+\d+",                # Table references, e.g., Table 1
    r"^Figure\s+\d+",               # Figure references, e.g., Figure 7
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
    r"FERTILIZERS, MANURES AND BIOFERTILIZERS ",
    r"SOIL FERTILITY MANAGEMENT",
    r"SOIL AND WATER QUALITY",
    r"SOIL POLLUTION AND ITS CONTROL",
    r"SOIL MANAGEMENT FOR SUSTAINABLE FARMING",
    r"CHEMICAL COMPOSITION OF SOILS",
    r"PHYSICAL PROPERTIES OF SOILS"
]

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

# noise_removal from chunks
def is_noisy_line(line):
    line = line.strip()

    if not line:
        return True

    tokens = line.split()

    # 1. Too many special characters
    special_chars = re.findall(r"[^a-zA-Z0-9\s]", line)
    if len(special_chars) / len(line) > 0.3:
        return True

    # 2. Too many numbers (graph/table)
    numbers = re.findall(r"\b\d+\b", line)
    if len(numbers) / max(len(tokens), 1) > 0.5:
        return True

    # 3. Very low word content
    words = re.findall(r"\b[a-zA-Z]{3,}\b", line)
    if len(words) == 0:
        return True

        # too short
    if len(line) < 20:
            return True

    return False
# clean docs
def clean_page(text):

    text = remove_unwanted_lines(text) # remove explicit unwanted patterns anywhere
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        if not is_noisy_line(line):
            cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)
def clean_docs(docs):
    cleaned = []

    for doc in docs:
        text = clean_page(doc.page_content)

        if text.strip():  # keep if anything meaningful remains
            doc.page_content = text
            cleaned.append(doc)

    return cleaned
import hashlib

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
    # join broken lines
    text = re.sub(r"\n+", "\n", text)

    # fix mid-sentence line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text

def normalize_docs(docs):
    for doc in docs:
        doc.page_content = normalize_text(doc.page_content)
    return docs
docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    #loader = PyMuPDFLoader(path)
    docs.extend(loader.load())

docs = clean_docs(docs)
docs = deduplicate_docs(docs)
docs = normalize_docs(docs)

# ----------------- TEXT SPLITTING -----------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[
        "\n\n",   # paragraphs FIRST
        ". ",     # then lines



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

# retriever = vector_db.as_retriever(
#     search_type='similarity',
#     search_kwargs={"k": 30}
# )
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


# ----------------- REMOVE DUPLICATES -----------------

def remove_duplicate_docs(docs):

    unique_docs = []
    seen = set()

    for doc in docs:

        fingerprint = doc.page_content.strip()[:200]

        if fingerprint not in seen:

            unique_docs.append(doc)
            seen.add(fingerprint)

    return unique_docs

from sklearn.metrics.pairwise import cosine_similarity

# def semantic_dedup(docs, embeddings=embedding_model, threshold=0.9):
#     unique_docs = []
#     vectors = []
#
#     for doc in docs:
#         emb = embeddings.embed_documents(doc.page_content)[0]
#
#         is_duplicate = False
#         for v in vectors:
#             if cosine_similarity([emb], [v])[0][0] > threshold:
#                 is_duplicate = True
#                 break
#
#         if not is_duplicate:
#             vectors.append(emb)
#             unique_docs.append(doc)
#
#     return unique_docs
def semantic_dedup(docs, embeddings=embedding_model, threshold=0.9):
    if not docs:
        return []

    # 1. Extract text
    texts = [doc.page_content for doc in docs]

    # 2. Batch embedding (ONE API call)
    all_embeddings = embeddings.embed_documents(texts)

    unique_docs = []
    unique_vectors = []

    # 3. Compare
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
# # ----------------- IMAGE OCR -----------------
# def extract_text_from_image(image_bytes):
#     try:
#         # Note: If tesseract is not in your PATH on Windows, uncomment and set the path below:
#         pytesseract.pytesseract.tesseract_cmd = r'D:\PyCharm\GenAI\tesseract\tesseract.exe'
#         img = Image.open(io.BytesIO(image_bytes))
#         text = pytesseract.image_to_string(img, lang="guj")
#         return text.strip()
#     except Exception as e:
#         print(f"OCR Error: {e}")
#         return ""
#




# ----------------- MAIN RAG FUNCTION -----------------

def response_llm(user_query, chat_history, image_context=None):

    # -------- RETRIEVAL --------

    candidate_docs = retriever.invoke(user_query)
    for i, doc in enumerate(candidate_docs):
        print(f"\nRetrieved {i}:\n{doc.page_content[:200]}")
    print("--"*50)
    print(f"\nRetrieved chunks (before dedup): {len(candidate_docs)}")

    # unique_docs = remove_duplicate_docs(cleaned_docs)
    #
    # print(f"Unique chunks after dedup: {len(unique_docs)}")

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
            "content": (
                "You are a helpful assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not present, say: Not found in context."
            )
        }
    ]


    MAX_HISTORY = 5
    recent_history = chat_history[-MAX_HISTORY:]


    for msg in recent_history:

        if msg.type == "human":

            history.append({
                "role": "user",
                "content": msg.content
            })

        elif msg.type == "ai":

            history.append({
                "role": "assistant",
                "content": msg.content
            })


    prompt_text = f"Context:\n{context}\n\n"
    if image_context:
        prompt_text += f"Image Context (extracted via OCR from user's uploaded image):\n{image_context}\n\n"
    prompt_text += f"Question:{user_query}"

    history.append({
        "role": "user",
        "content": prompt_text
    })


    # -------- GROQ LLM INFERENCE --------

    response = groq_client.chat.completions.create(

        messages=history,

        model="llama-3.3-70b-versatile",

        temperature=0.7,

        max_tokens=500

    )


    output = response.choices[0].message.content


    # -------- RETURN SOURCES --------

    sources = []

    for doc in retrieved_docs:

        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })


    return output, sources
