# RAG-based Chatbot

This chatbot aims to answer user's questions on the content of PDF files the user uploads. It utilises Pinecone for vector storage and OpenAI gpt-4o model for generating responses. The UI is built on Streamlit.

## Setup instructions
### 1. Clone the repository
```
git clone [git repo]
cd [local repo path]
```
### 2. Create a .env file with your API keys based on the .env.sample provided
```
OPENAI_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
- `OPENAI_KEY`: You can obtain an API key from [OpenAI's platform](https://platform.openai.com/docs/).
- `PINECONE_API_KEY`: You can get this from [Pinecone](https://www.pinecone.io/) after creating an account.


### 3. Build and Run the Application in Docker
```
docker build -t streamlit-app:latest .
docker run -p 8501:8501 streamlit-app:latest
```
### 4. Access the Application
Once the container is running, open your browser and navigate to:
```
http://localhost:8501
```
You will be able to upload PDF files and ask questions about their content directly via the UI.

## Prerequisites
- Docker (ensure you have [Docker](https://docs.docker.com/engine/install/) installed)
- API keys for OpenAI and Pinecone
