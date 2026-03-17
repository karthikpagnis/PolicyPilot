from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from utils.pdf_utils import get_page_count
from utils.ollama_client import get_ollama_status

load_dotenv()

from workflow import claim_workflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup: Validate Ollama and phi3 model before accepting requests
    status = get_ollama_status(model="phi3:latest")

    if not status["ok"]:
        error_msg = status.get('error', 'Unknown error')
        print(f"Startup Error: {error_msg}")
        print("To fix: 1) Install Ollama from https://ollama.ai  2) Run: ollama pull phi3")
        raise RuntimeError(f"Ollama validation failed: {error_msg}")

    print(f"Ollama ready at {status['host']} with model '{status['model']}'")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="Claim Processing Pipeline",
    description="Processes medical claim PDFs using AI to extract structured data",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Claim Processing API is live",
        "usage": "POST /api/process with claim_id (string) and file (PDF)"
    }


@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(..., description="Unique identifier for this claim"),
    file: UploadFile = File(..., description="PDF file of the medical claim")
):
    """
    Main endpoint - receives a claim ID and PDF, runs the full
    LangGraph pipeline, and returns extracted data as JSON.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    page_count = get_page_count(pdf_bytes)
    print(f"Processing claim '{claim_id}' with {page_count} pages")

    initial_state = {
        "claim_id": claim_id,
        "pdf_bytes": pdf_bytes,
        "page_images": {},
        "page_classifications": {},
        "routing": {},
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "final_result": {}
    }

    try:
        result = await claim_workflow.ainvoke(initial_state)
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    final_result = result.get("final_result", {})
    print(f"Claim '{claim_id}' processed successfully")
    return JSONResponse(content=final_result)
