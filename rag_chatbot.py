import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

@st.cache_data
def load_data():
    df = pd.read_csv("Training_Dataset.csv")
    df.fillna('', inplace=True)
    corpus = df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()
    return corpus

text_corpus = load_data()

@st.cache_resource
def setup_index(corpus):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return model, index

embed_model, faiss_index = setup_index(text_corpus)

def retrieve_top_k(query, k=3):
    query_vec = embed_model.encode([query])
    _, indices = faiss_index.search(query_vec, k)
    return [text_corpus[i] for i in indices[0]]

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", tokenizer="tiiuae/falcon-7b-instruct")

generator = load_generator()

def generate_answer(query):
    context = " ".join(retrieve_top_k(query))
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=300, do_sample=True)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()

st.set_page_config(page_title="Loan Dataset RAG Chatbot")
st.title("📊 RAG Q&A Chatbot – Loan Approval Dataset")
st.markdown("Ask any question about the loan dataset and get AI-powered answers.")

user_query = st.text_input("🔍 Enter your question:")

if user_query:
    with st.spinner("🔎 Searching and thinking..."):
        result = generate_answer(user_query)
        st.success(result)
