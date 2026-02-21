from fastapi import FastAPI
from pydantic import BaseModel
from app.retrieval import RetrievalPipeline
from app.generation import LLMGenerator
from app.utils import setup_logger
import time

logger = setup_logger()

app = FastAPI(title="Multimodal RAG API")

# Initialize pipeline once at startup
retrieval = RetrievalPipeline()
llm = LLMGenerator()


class AddDocumentRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Multimodal RAG API is running."}


@app.post("/add-document")
def add_document(request: AddDocumentRequest):
    retrieval.add_text_document(request.text)
    return {"status": "Document added successfully."}


@app.post("/query")
def query(request: QueryRequest):
    start_time = time.time()

    docs = retrieval.retrieve(request.question)
    context = "\n".join(docs)

    response = llm.generate(context=context, question=request.question)

    latency = round(time.time() - start_time, 3)

    return {
        "question": request.question,
        "retrieved_documents": docs,
        "answer": response,
        "latency_seconds": latency
    }