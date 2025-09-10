import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
import tempfile
import dotenv

dotenv.load_dotenv()

# --------------------------
# Load PDF
# --------------------------
def load_data(pdf_file):
    """Saves an uploaded PDF to a temporary file and loads it."""
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    # Load PDF using the temp file path
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    return documents


# --------------------------
# Split into Chunks
# --------------------------
def split_text(documents, chunk_size=1000, chunk_overlap=200):
    """Splits a list of documents into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# --------------------------
# Create Embeddings + Vector Store
# --------------------------
def create_vector_store(chunks):
    """Creates a Chroma vector store from document chunks for efficient retrieval."""
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version="2023-05-15",
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore


# --------------------------
# Generate Questions
# --------------------------
def generate_questions(chunks, prompts):
    """Generates practice questions based on document chunks using an LLM."""
    # Combine all chunks into one text
    text = "\n\n".join([chunk.page_content for chunk in chunks])

    # Use Azure OpenAI GPT
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        temperature=0.5,
    )

    # Build the question prompt
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template=prompts["question"]
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=question_prompt)

    # Run and return questions
    questions = chain.run({"text": text})
    return questions


# --------------------------
# Generate Answers
# --------------------------
def generate_answers(selected_questions, vectorstore, prompts):
    """Generates answers to selected questions using an LLM and a vector store."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        temperature=0.3,
    )

    # Build the answer prompt
    answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompts["answer"]
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=answer_prompt)

    answers = {}
    for question in selected_questions:
        # Retrieve relevant chunks from the vector store
        docs = vectorstore.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate the answer
        response = chain.run(
            {"question": question, "context": context}
        )
        answers[question] = response

    return answers
