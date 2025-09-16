import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
import tempfile
import dotenv
from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Load environment variables
dotenv.load_dotenv()

# --- Prompts ---
prompts = {
    "question": """
    You are a helpful teaching assistant. Based on the following study material:

    {text}

    Generate 15 high-quality practice questions that cover key concepts.
    Do not include answers, only the questions.
    """,
    "answer": """
    You are a helpful teaching assistant. Based on the following study material, provide a
    comprehensive answer to the question. If the information is not available in the provided
    material, state that explicitly.

    Study Material:
    {context}

    Question: {question}

    Answer:
    """
}

# --- LangGraph State and Nodes ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    pdf_content: Any  # Store the content as bytes
    chunks: List[Document]
    questions: List[str]
    selected_questions: List[str]
    vectorstore: Any
    generated_answers: Dict[str, str]
    user_question: str
    custom_answer: str

def load_and_split_node(state: GraphState):
    """Loads and splits the PDF into chunks from its raw content."""
    st.info("📖 Reading and processing your PDF...")
    pdf_content = state["pdf_content"]
    
    # Create a temporary file to save the content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_content)
        tmp_path = tmp_file.name
    
    try:
        # Pass the temporary file path to PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
    finally:
        # Ensure the temporary file is deleted
        os.remove(tmp_path)
        
    return {"chunks": chunks}

def generate_questions_node(state: GraphState):
    """Generates practice questions from chunks using an LLM."""
    st.info("✍️ Generating practice questions...")
    chunks = state["chunks"]
    text = "\n\n".join([chunk.page_content for chunk in chunks])
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        temperature=0.5,
    )
    question_prompt = PromptTemplate(input_variables=["text"], template=prompts["question"])
    chain = LLMChain(llm=llm, prompt=question_prompt)
    questions_text = chain.run({"text": text})
    lines = questions_text.split('\n')
    questions = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if line.strip()]
    return {"questions": questions}

def create_vector_store_node(state: GraphState):
    """Creates a Chroma vector store from document chunks."""
    st.info("🧠 Building knowledge base...")
    chunks = state["chunks"]
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version="2023-05-15",
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return {"vectorstore": vectorstore}

def generate_answers_node(state: GraphState):
    """Generates answers for selected questions."""
    st.info("🤖 Generating answers...")
    selected_questions = state["selected_questions"]
    vectorstore = state["vectorstore"]
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        temperature=0.3,
    )
    answer_prompt = PromptTemplate(input_variables=["context", "question"], template=prompts["answer"])
    chain = LLMChain(llm=llm, prompt=answer_prompt)
    answers = {}
    for question in selected_questions:
        docs = vectorstore.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = chain.run({"question": question, "context": context})
        answers[question] = response
    return {"generated_answers": answers}

def generate_custom_answer_node(state: GraphState):
    """Generates an answer to a user's custom question."""
    st.info("🤖 Generating answer to your question...")
    user_question = state["user_question"]
    vectorstore = state["vectorstore"]
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        temperature=0.3,
    )
    answer_prompt = PromptTemplate(input_variables=["context", "question"], template=prompts["answer"])
    chain = LLMChain(llm=llm, prompt=answer_prompt)
    docs = vectorstore.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    custom_answer = chain.run({"question": user_question, "context": context})
    return {"custom_answer": custom_answer}

# --- LangGraph Workflow ---
def create_graph():
    builder = StateGraph(GraphState)
    builder.add_node("load_and_split", load_and_split_node)
    builder.add_node("generate_questions", generate_questions_node)
    builder.add_node("create_vector_store", create_vector_store_node)
    builder.add_node("generate_answers", generate_answers_node)
    builder.add_node("generate_custom_answer", generate_custom_answer_node)

    # Define the initial workflow
    builder.set_entry_point("load_and_split")
    builder.add_edge("load_and_split", "generate_questions")
    builder.add_edge("generate_questions", "create_vector_store")

    # We need a state-based router to handle user input
    def router(state):
        if state["selected_questions"]:
            return "generate_answers"
        if state["user_question"]:
            return "generate_custom_answer"
        return "end"

    builder.add_conditional_edges(
        "create_vector_store",
        router,
        {
            "generate_answers": "generate_answers",
            "generate_custom_answer": "generate_custom_answer",
            "end": END
        }
    )

    builder.add_edge("generate_answers", END)
    builder.add_edge("generate_custom_answer", END)

    return builder.compile()

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="📘 Personal Learning Assistant", layout="wide")
    st.title("📘 Personal Learning Assistant")

    # Initialize session state for the graph
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = {
            "pdf_content": None,  # Use a bytes object to store content
            "chunks": None,
            "questions": None,
            "selected_questions": [],
            "vectorstore": None,
            "generated_answers": {},
            "user_question": "",
            "custom_answer": ""
        }
    
    if "graph_app" not in st.session_state:
        st.session_state.graph_app = create_graph()

    # --- Upload PDF ---
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

    # The new fix: Check if a new file has been uploaded and read its content
    if pdf_file is not None and st.session_state.graph_state["pdf_content"] != pdf_file.getvalue():
        st.session_state.graph_state = {
            "pdf_content": pdf_file.getvalue(),  # Store the file's content as bytes
            "chunks": None,
            "questions": None,
            "selected_questions": [],
            "vectorstore": None,
            "generated_answers": {},
            "user_question": "",
            "custom_answer": ""
        }
        
        with st.spinner("Processing PDF..."):
            initial_state = st.session_state.graph_app.invoke(st.session_state.graph_state, {"recursion_limit": 5})
            st.session_state.graph_state = initial_state
    
    # --- Display Questions & Answer Boxes (only if questions exist) ---
    if st.session_state.graph_state["questions"]:
        st.subheader("📝 Practice Questions")
        for i, q in enumerate(st.session_state.graph_state["questions"], 1):
            st.write(f"{i}. {q}")
        
        selected = st.multiselect(
            "Select the questions you want the answers for:",
            options=st.session_state.graph_state["questions"]
        )
        
        if st.button("Generate answers"):
            st.session_state.graph_state["selected_questions"] = selected
            st.session_state.graph_state["user_question"] = ""
            with st.spinner("🤖 Generating answers..."):
                final_state = st.session_state.graph_app.invoke(st.session_state.graph_state, {"recursion_limit": 5})
                st.session_state.graph_state = final_state

        if st.session_state.graph_state["generated_answers"]:
            st.subheader("💡 Suggested Answers")
            for question, answer in st.session_state.graph_state["generated_answers"].items():
                st.info(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")

        # --- User Custom Question Input ---
        st.subheader("💬 Ask Your Own Question")
        user_question = st.text_input("Type your question about the uploaded PDF:")
        if st.button("Get Answer to My Question") and user_question:
            st.session_state.graph_state["user_question"] = user_question
            st.session_state.graph_state["selected_questions"] = []
            st.session_state.graph_state["generated_answers"] = {}
            with st.spinner("🤖 Generating answer to your question..."):
                final_state = st.session_state.graph_app.invoke(st.session_state.graph_state, {"recursion_limit": 5})
                st.session_state.graph_state = final_state

        if st.session_state.graph_state["custom_answer"]:
            st.info(f"**Your Question:** {st.session_state.graph_state['user_question']}")
            st.markdown(f"**Answer:** {st.session_state.graph_state['custom_answer']}")

if __name__ == "__main__":
    main()