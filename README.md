# Government Scheme Eligibility Navigator

A multilingual RAG-based application that helps users understand their eligibility for government schemes using official government documents.

Users provide basic information such as their state, occupation, land ownership, income-related details, and other eligibility factors. The system retrieves relevant information from government scheme documents and uses an LLM to provide an eligibility assessment, reasoning, and next steps.

## Features

- Government scheme eligibility checking
- Retrieval-Augmented Generation (RAG)
- Uses official government documents as the knowledge source
- Hindi and English language support
- Step-by-step user questionnaire
- Eligibility reasoning based on retrieved documents
- Source information from retrieved documents
- Next-step guidance

## Tech Stack

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend
- Python
- FastAPI
- LangChain
- Hugging Face Embeddings
- FAISS
- Groq LLM

### Data
- Government scheme PDFs
- Processed scheme documents
- FAISS vector index


## How It Works

User Information
       |
       v
Eligibility Questions
       |
       v
RAG Retrieval
       |
       v
Government Scheme Documents
       |
       v
Relevant Eligibility Rules
       |
       v
LLM Analysis
       |
       v
Eligibility Result
       |
       v
Reasoning and Next Steps

## Currently Supported Scheme

- PM-KISAN

Additional government schemes can be added by processing their official documents and creating the corresponding embeddings.

## Running Locally

### Backend

Navigate to the backend:

cd backend

Activate the virtual environment:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GROQ_API_KEY=your_groq_api_key

Start the FastAPI server:

python -m uvicorn main:app --reload

The backend will run at:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

### Frontend

Open another terminal:

cd frontend

Install dependencies:

npm install

Start the development server:

npm run dev

The frontend will run at:

http://localhost:3000

## Environment Variables

Backend:

GROQ_API_KEY=your_groq_api_key

Frontend:

NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

Do not commit API keys or .env files to GitHub.

## Disclaimer

This project provides an informational eligibility assessment based on available government documents and the information provided by the user. It is not an official government service and does not guarantee eligibility or approval for any government scheme. Final eligibility is determined by the relevant government authority.
