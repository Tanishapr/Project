from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.eligibility import determine_eligibility


app = FastAPI(
    title="Government Scheme Eligibility Navigator"
)


# Allow the Next.js frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://govt-scheme-navigator-2.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EligibilityRequest(BaseModel):
    scheme: str
    language: str = "en"
    user_profile: dict


@app.get("/")
def root():
    return {
        "message": "Government Scheme Navigator API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/eligibility")
def eligibility(request: EligibilityRequest):
    result = determine_eligibility(
    user_profile=request.user_profile,
    scheme=request.scheme,
    language=request.language,
)

    return result
