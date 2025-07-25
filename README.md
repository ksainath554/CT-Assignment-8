#  RAG Q&A Chatbot – Loan Approval Dataset (All-in-One File)

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using a **single Python script** that can:
- Embed and search the Loan Approval dataset
- Retrieve relevant rows using FAISS
- Generate answers using Hugging Face language models
- Provide an interactive chatbot UI using Streamlit

##  Dataset

**Source:** [Kaggle Dataset – Loan Approval Prediction](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)

➡Download the file `Training_Dataset.csv`  
➡Place it in the **same directory** as your `.py` file

##  How to Run

### 1 Install Requirements
```bash
pip install -r requirements.txt
```

### 2️ Run the Chatbot (Streamlit UI)
```bash
streamlit run rag_chatbot.py
```

➡ Open browser at `http://localhost:8501`

## 💬 Sample Questions
- How many loans were approved?
- What is the average loan amount?
- Are married applicants more likely to get a loan?
- How many applicants are self-employed?
