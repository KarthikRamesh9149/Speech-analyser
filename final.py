# Import necessary libraries
import streamlit as st
import speech_recognition as sr
from st_audiorec import st_audiorec  # Audio recording for Streamlit
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import requests
import os

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize Session State
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Persistent Directory for ChromaDB
CHROMA_DB_DIR = "./chroma_db"

# Speech Recognizer
recognizer = sr.Recognizer()

# Tavily API Key
TAVILY_API_KEY = "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"

# Helper Functions
def transcribe_audio(audio_bytes, speaker):
    """Capture and transcribe recorded audio."""
    try:
        temp_audio_path = f"temp_{speaker}.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            st.success(f"{speaker} said: {transcription}")
            return transcription
    except sr.UnknownValueError:
        st.error(f"Could not understand {speaker}'s audio.")
        return None
    except sr.RequestError:
        st.error("Speech Recognition service is unavailable.")
        return None

def identify_question(text):
    """Identify embedded questions in text."""
    prompt = f"Identify the question embedded in the following statement:\n{text}"
    response = llm.invoke(prompt)
    return response.content.strip()

def decide_action(question):
    """Decide whether to use Web Search or LLM."""
    decision_prompt = f"""You are tasked with deciding the action for answering the following question:\n
    Question: {question}\n
    Decide whether to use 'Web Search' or 'LLM' to answer.\nProvide one of these answers: 'Web Search', 'LLM'."""
    response = llm.invoke(decision_prompt)
    return response.content.strip()

def web_search(question):
    """Perform web search using Tavily API."""
    response = requests.post(
        'https://api.tavily.com/search',
        json={"query": question, "num_results": 3},
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"}
    )
    if response.status_code == 200:
        results = response.json().get("results", [])
        summaries = [result['snippet'] for result in results]
        return " ".join(summaries[:3])[:100]  # Limit to 100 words
    return "Unable to perform web search at the moment."

def trim_answer(answer, word_limit=100):
    """Trim answer to a specified word limit."""
    words = answer.split()
    return " ".join(words[:word_limit])

# Streamlit UI
st.title("AI Assistant with Agentic RAG Framework")
st.write("Supports RAG, Web Search, and direct LLM responses.")

# PDF Upload for RAG
uploaded_pdf = st.file_uploader("Upload a PDF for RAG (optional)", type=["pdf"])
if uploaded_pdf and not st.session_state.pdf_uploaded:
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_pdf.read())
    st.success("PDF uploaded successfully! Processing...")
    loader = PyPDFLoader("uploaded_document.pdf")
    documents = loader.load()

    # Initialize Chroma vector store with a persistent directory
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)

    st.session_state.vector_store = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    st.session_state.vector_store.persist()
    st.session_state.pdf_uploaded = True

vector_store = st.session_state.vector_store

# Conversation Input Loop
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎤 Speaker 1")
    audio_bytes = st_audiorec()
    if audio_bytes:
        text = transcribe_audio(audio_bytes, "Speaker 1")
        if text:
            st.session_state.conversation.append(("Speaker 1", text))

with col2:
    st.subheader("🎤 Speaker 2")
    audio_bytes = st_audiorec()
    if audio_bytes:
        text = transcribe_audio(audio_bytes, "Speaker 2")
        if text:
            st.session_state.conversation.append(("Speaker 2", text))

# Display Conversation
if st.session_state.conversation:
    st.subheader("**Conversation History:**")
    for speaker, line in st.session_state.conversation:
        st.write(f"{speaker}: {line}")

# Processing and Answer Generation
if st.button("Help Me!"):
    st.subheader("Generating Answer...")

    full_conversation = " ".join([line for _, line in st.session_state.conversation])
    st.write("**Combined Transcription:**")
    st.code(full_conversation)

    question = identify_question(full_conversation)
    st.write("**Question Identified:**")
    st.info(question)

    final_source = None  # Track the source of the answer

    # Check RAG
    if vector_store:
        st.write("Checking PDF context (RAG)...")
        retriever = vector_store.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
        answer = qa_chain.run({"question": question, "chat_history": []})
        if answer and "I don't know" not in answer:
            final_source = "RAG"

    # If RAG fails, decide between Web Search and LLM
    if not final_source:
        decision = decide_action(question)
        st.write("**LLM Decision:**")
        st.warning(decision)

        if decision == "Web Search":
            answer = web_search(question)
            final_source = "Web Search"
        else:
            response = llm.invoke(question)
            answer = response.content
            final_source = "LLM"

    # Display the final answer and source
    trimmed_answer = trim_answer(answer, word_limit=100)
    st.subheader("**Answer:**")
    st.success(trimmed_answer)

    st.write(f"**Source of Answer:** {final_source}")

# Clear History
if st.button("Clear Conversation History"):
    st.session_state.conversation = []
    st.session_state.pdf_uploaded = False
    st.session_state.vector_store = None
    st.success("Conversation history cleared.")
