import os
import re
import streamlit as st
from learning_assistant_functions import load_data, split_text, generate_questions
from learning_assistant_prompts import prompts
import dotenv

# Load environment variables at the very top
dotenv.load_dotenv()

st.set_page_config(page_title="📘 Personal Learning Assistant", layout="wide")
st.title("📘 Personal Learning Assistant")

# --------------------------
# Upload PDF
# --------------------------
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file is not None:
    # --------------------------
    # Load & Split PDF (cached)
    # --------------------------
    if "chunks" not in st.session_state:
        with st.spinner("📖 Reading and processing your PDF..."):
            documents = load_data(pdf_file)
            st.session_state.chunks = split_text(documents)

    chunks = st.session_state.chunks

    # --------------------------
    # Generate Questions (cached)
    # --------------------------
    if "questions" not in st.session_state:
        with st.spinner("✍️ Generating practice questions..."):
            questions_text = generate_questions(chunks, prompts)

            # Split questions reliably using regex
            questions = re.split(r"\n\d+\.\s+", questions_text)
            questions = [q.strip() for q in questions if q.strip()]

            st.session_state.questions = questions

    questions = st.session_state.questions

    # --------------------------
    # Display Questions & Answer Boxes
    # --------------------------
    st.subheader("📝 Practice Questions")
    selected = st.multiselect(
        "Select the questions you want to answer:",
        options=questions
    )

    if selected:
        st.subheader("✏️ Your Answers")
        for i, q in enumerate(selected, 1):
            st.text_area(f"Q{i}: {q}", key=f"answer_{i}", height=120)
