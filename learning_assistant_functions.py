import dotenv

dotenv.load_dotenv()
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
import tempfile

# --------------------------
# Load PDF
# --------------------------
def load_data(pdf_file):
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# --------------------------
# Create Embeddings + Vector Store
# --------------------------
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version="2023-05-15",
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore


# --------------------------
# Generate Questions (Azure Chat Compatible)
# --------------------------
def generate_questions(chunks, prompts):
    """
    Generate practice questions from a list of document chunks using Azure OpenAI Chat model.
    """
    # Combine all chunks into one text
    text = "\n\n".join([chunk.page_content for chunk in chunks])

    # Initialize Azure Chat model correctly
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # your deployment name in Azure
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),      # e.g., https://<your-resource>.openai.azure.com/
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        temperature=0.5,
    )

    # Build the question prompt
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template=prompts["question"]
    )

    # Create the chain
    chain = LLMChain(llm=llm, prompt=question_prompt)

    # Generate questions
    questions_text = chain.run({"text": text})

    # Return as plain text
    return questions_text
