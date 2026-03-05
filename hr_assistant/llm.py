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

loader = PyPDFLoader('D:/PyCharm/GenAI/Chatbot/hr_assistant/the_nestle_hr_policy_pdf_2012.pdf')

docs = loader.load()

# text splitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

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
retriever = vector_db.as_retriever(search_kwargs={"k":3}) # returns top3 chunks


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


def response_llm(user_query, chat_history):

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(user_query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    history = [{"role": "system", "content": "Answer only using the provided context."}]

    for msg in chat_history:
        if msg.type == "human":
            history.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            history.append({"role": "assistant", "content": msg.content})

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

