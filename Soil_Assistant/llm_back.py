import os
import warnings

warnings.filterwarnings("ignore")

# SHARED CACHE LOCATION (use this in ALL your scripts)
CACHE_DIR = r"D:\huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR

print(f"Using shared cache: {CACHE_DIR}")

# ----------------- 1️⃣ IMPORTS -----------------
import streamlit as st

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# pdf loader

from langchain_community.document_loaders import PyPDFLoader
pdf_paths = [
    "D:/PyCharm/GenAI/Chatbot/RAG-pipeline/Fundamental of Soil Sci by ISSS-1_Searchable.pdf",

]

docs = []

for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

docs= text_splitter.split_documents(docs)

# embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vector database

from langchain_community.vectorstores import Chroma

vector_db = Chroma.from_documents( # defining a vector database
    documents=docs,                # documents to put in
    embedding=embedding_model,     # emb model to be used
    persist_directory="vector_db"  # db storage directory
)

# RAG - Retrival
retriever = vector_db.as_retriever(search_type='similarity',search_kwargs={"k":15}) # returns top 15 chunks

# RERANKER - Reranks the retrieved chunks

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank_documents(query, docs, top_n=3):
    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_n]]


# ----------------- 2️⃣ LOAD LOCAL MODEL (4GB VRAM Optimized) -----------------
@st.cache_resource  # Cache the model so it doesn't reload on every interaction
def load_local_llm():
    """Load SmolLM2-1.7B model locally with FP16"""

    print(f"Loading model from cache: {CACHE_DIR}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in FP16 (fits in 4GB VRAM)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        dtype=torch.float16,
        device_map="cuda:0",
    )
    model.generation_config.max_length = None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        return_full_text=False
    )

    return pipe, tokenizer


pipe, tokenizer = load_local_llm()

# # -------- OPTIONAL : RELEVANT HISTORY RETRIEVAL FUNCTION --------
# from sklearn.metrics.pairwise import cosine_similarity
# def retrieve_relevant_history(user_query, chat_history, embedding_model, threshold=0.75):
#
#     query_embedding = embedding_model.embed_query(user_query)
#
#     relevant_messages = []
#
#     for msg in chat_history:
#
#         msg_embedding = embedding_model.embed_query(msg.content)
#
#         similarity = cosine_similarity(
#             [query_embedding],
#             [msg_embedding]
#         )[0][0]
#
#         if similarity > threshold:
#             relevant_messages.append(msg)
#
#     return relevant_messages

from sentence_transformers import CrossEncoder


# removing duplicates from retrieved chunks---
def remove_duplicate_docs(docs):
    unique_docs = []
    seen = set()

    for doc in docs:
        # use first 200 chars as fingerprint
        fingerprint = doc.page_content.strip()[:200]

        if fingerprint not in seen:
            unique_docs.append(doc)
            seen.add(fingerprint)

    return unique_docs

def response_llm(user_query, chat_history):

    # Retrieve relevant chunks

       #retrieved_docs = retriever.invoke(user_query)  #----without reranker*----

    # Step 1: retrieve candidates
    candidate_docs = retriever.invoke(user_query)

    print(f"\nRetrieved chunks (before dedup): {len(candidate_docs)}")

    # Step 2: remove duplicates
    unique_docs = remove_duplicate_docs(candidate_docs)

    print(f"Unique chunks after dedup: {len(unique_docs)}")

    # Step 3: rerank unique docs
    retrieved_docs = rerank_documents(user_query, unique_docs, top_n=5)

    print(f"Chunks sent to reranker: {len(unique_docs)}")
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    history = [
        {
            "role": "system",
            "content": "Use the context to answer the question. Extract the answer clearly. If the context does not contain the answer, say: Not found in context."
        }
    ]

    # *****full chat history code ******
    # for msg in chat_history:
    #     if msg.type == "human":
    #         history.append({"role": "user", "content": msg.content})
    #     elif msg.type == "ai":
    #         history.append({"role": "assistant", "content": msg.content})


    # ***** limiting upto last 5 conversations in history ******
    MAX_HISTORY = 5  # sliding context window*
    recent_history = chat_history[-MAX_HISTORY:]

    for msg in recent_history:
        if msg.type == "human":
            history.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            history.append({"role": "assistant", "content": msg.content})

    # -------- OPTIONAL: RETRIEVE RELEVANT HISTORY (COMMENTED) --------
    # relevant_history = retrieve_relevant_history(user_query, chat_history, embedding_model)

    # for msg in relevant_history:
    #     if msg.type == "human":
    #         history.append({"role": "user", "content": msg.content})
    #     elif msg.type == "ai":
    #         history.append({"role": "assistant", "content": msg.content})


    history.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:{user_query}"
    })

    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )


    output = pipe(prompt)[0]["generated_text"]
    sources = []
    for i, doc in enumerate(retrieved_docs):
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    return output, sources

