# Personal Learning Assistant

A Streamlit-based app that helps users study more effectively by generating practice questions and answers from uploaded PDF study materials using Azure OpenAI and vector search.

## Features
- Upload a PDF and automatically split it into study chunks
- Generate 15 high-quality practice questions from your material
- Select questions and get AI-generated answers based on your PDF
- Uses Azure OpenAI for LLM and embeddings
- Fast, context-aware retrieval with Chroma vector store

## How It Works
1. **Upload PDF**: The app reads and splits your PDF into manageable chunks.
2. **Generate Questions**: An LLM creates practice questions covering key concepts.
3. **Select Questions**: Pick questions you want answered.
4. **Get Answers**: The app retrieves relevant context and generates answers using the LLM.

## Installation
1. Clone this repository:
   ```powershell
   git clone https://github.com/snehalathag-infomagnus/personal-learning-assistant.git
   cd personal-learning-assistant
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Set up your Azure OpenAI credentials in a .env file:
   ```
   AZURE_OPENAI_API_KEY=your-key-here
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=your-deployment-name
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment
   ```

## Usage
Run the Streamlit app:
```powershell
streamlit run learning_assistant.py
```
Follow the UI to upload a PDF, generate questions, and get answers.

## File Structure
- learning_assistant.py: Main Streamlit app
- learning_assistant_functions.py: Core logic for PDF loading, chunking, embeddings, LLM calls
- learning_assistant_prompts.py: Prompt templates for question and answer generation
- requirements.txt: Python dependencies
- .env: Azure OpenAI credentials (not tracked in git)

## Dependencies
- streamlit
- langchain
- langchain-community
- openai
- pypdf
- chromadb
- langchain-openai
- python-dotenv