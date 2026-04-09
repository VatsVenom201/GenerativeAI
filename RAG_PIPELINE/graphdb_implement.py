import os
import warnings
import re
warnings.filterwarnings("ignore")

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

# # ----------------- GRAPH IMPLEMENTATION -----------------
from neo4j import GraphDatabase

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12112003vats@neo.101"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
def is_worthy_for_graph(text):
    keywords = [
        "cause", "effect", "affect", "increase", "decrease",
        "lead", "result", "impact", "relation"
    ]
    return any(k in text.lower() for k in keywords)

g_docs = [doc for doc in docs if is_worthy_for_graph(doc.page_content)]

def extract_triplets_batch(text_batch):
    prompt = f"""
    Extract relationships as triplets in this format:

    (CHUNK_ID, entity1, relation, entity2)

    Rules:
    - CHUNK_ID must be like CHUNK_0, CHUNK_1 etc
    - Keep entities short
    - Normalize names (no duplicates)
    - Only meaningful scientific relations
    - DO NOT mix relations between chunks

    Text:
    {text_batch}
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
def normalize_entity(e):
    e = e.lower().strip()
    e = re.sub(r"[^a-z0-9\s]", "", e)
    return e
# llm's relation data parser
def parse_triplets_with_chunk(output):
    triplets = []

    matches = re.findall(r"\((.*?)\)", output)

    for m in matches:
        parts = [p.strip().lower() for p in m.split(",")]

        if len(parts) == 4:
            chunk_id = parts[0]   # chunk_0
            e1 = normalize_entity(parts[1])
            rel = normalize_entity(parts[2])
            e2 = normalize_entity(parts[3])
            triplets.append((chunk_id, e1, rel, e2))

    return triplets
# storing the parsed output
def add_triplet(tx, e1, rel, e2, chunk_text):
    tx.run("""
        MERGE (a:Entity {name: $e1})
        MERGE (b:Entity {name: $e2})
        MERGE (a)-[r:RELATION {type: $rel}]->(b)
        SET r.context = coalesce(r.context, '') + '\n' + $chunk
    """, e1=e1, e2=e2, rel=rel, chunk=chunk_text)

def store_triplets_batch(triplets_with_chunk, chunk_map):
    with driver.session() as session:
        for chunk_id, e1, rel, e2 in triplets_with_chunk:
            if chunk_id in chunk_map:
                session.execute_write(
                    add_triplet,
                    e1,
                    rel,
                    e2,
                    chunk_map[chunk_id]
                )
# graph build loop...

BATCH_SIZE = 5
batch = []

for doc in g_docs:

    text = doc.page_content

    batch.append(text)

    if len(batch) == BATCH_SIZE:

        # Step 1: Tag chunks
        combined_text = ""
        for i, chunk in enumerate(batch):
            combined_text += f"[CHUNK_{i}]\n{chunk}\n\n"

        # Step 2: LLM extraction
        raw_output = extract_triplets_batch(combined_text)

        # Step 3: Parse
        triplets = parse_triplets_with_chunk(raw_output)

        # Step 4: Map correctly
        chunk_map = {f"chunk_{i}": batch[i] for i in range(len(batch))}

        store_triplets_batch(triplets, chunk_map)

        batch = []

    # leftover batch
    if batch:
        combined_text = ""
        for i, chunk in enumerate(batch):
            combined_text += f"[CHUNK_{i}]\n{chunk}\n\n"

        raw_output = extract_triplets_batch(combined_text)
        triplets = parse_triplets_with_chunk(raw_output)

        chunk_map = {f"chunk_{i}": batch[i] for i in range(len(batch))}

        store_triplets_batch(triplets, chunk_map)



# getting relations according to user_query
def get_graph_context(query):

    query = query.lower()

    with driver.session() as session:

        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS $q OR toLower(b.name) CONTAINS $q
            RETURN a.name, r.type, b.name, r.context
            LIMIT 10
        """, q=query)

        contexts = []

        for record in result:
            contexts.append(record["r.context"])

    return contexts
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
    graph_contexts = get_graph_context(user_query)

    graph_text = "\n\n".join(graph_contexts)

    context = context + "\n\nThe following is graphDB's relational data" + graph_text
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
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=500
    )

    output = response.choices[0].message.content

    sources = []
    for doc in retrieved_docs:
        sources.append({"content": doc.page_content, "metadata": doc.metadata})

    return output, sources