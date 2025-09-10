import os
import re
import streamlit as st
from learning_assistant_functions import load_data, split_text, generate_questions, generate_answers, create_vector_store
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

            # Split into lines and strip the leading number and space from each line
            lines = questions_text.split('\n')
            questions = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if line.strip()]

            st.session_state.questions = questions

    questions = st.session_state.questions

    # --------------------------
    # Create Vector Store (cached)
    # --------------------------
    if "vectorstore" not in st.session_state:
        with st.spinner("🧠 Building knowledge base..."):
            st.session_state.vectorstore = create_vector_store(chunks)

    # --------------------------
    # Display Questions & Answer Boxes
    # --------------------------
    st.subheader("📝 Practice Questions")

    # Display the full list of questions before the multiselect
    for i, q in enumerate(questions, 1):
        st.write(f"{i}. {q}")

    selected = st.multiselect(
        "Select the questions you want to answer:",
        options=questions
    )

    # --------------------------
    # Generate Answers from LLM
    # --------------------------
    if st.button("Generate answers") and selected:
        with st.spinner("🤖 Generating answers..."):
            generated_answers = generate_answers(
                selected,
                st.session_state.vectorstore,
                prompts
            )
            st.session_state.generated_answers = generated_answers

    # Display generated answers if they exist
    if "generated_answers" in st.session_state and st.session_state.generated_answers:
        st.subheader("💡 Suggested Answers")
        for question, answer in st.session_state.generated_answers.items():
            st.info(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
