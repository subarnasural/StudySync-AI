"""
backend/main.py
===============
FastAPI application entry-point.
Supports Smart Learning Platform features: Quiz, Dashboard, Modes, Bilingual, and Attention Monitoring.
"""

import logging
import os
import shutil
import time
import threading
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.utils.ocr_engine import extract_text_from_image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join("uploaded_data", "files")
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
SUPPORTED_IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="StudySync AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# Attention State
attention_active = False
attention_results = {}
attention_thread = None

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup_summary():
    logger.info("=" * 60)
    logger.info("AI Teaching Assistant v2.3.0 starting")
    logger.info("  Upload dir   : %s", os.path.abspath(UPLOAD_DIR))
    logger.info("=" * 60)

@app.on_event("shutdown")
async def _shutdown_cleanup():
    logger.info("Server shutting down. Cleaning up temporary files...")
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    logger.info(f"Deleted: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    
    # Global stop for attention
    global attention_active
    attention_active = False

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    mode: Optional[str] = "default"
    language: Optional[str] = "english"
    session_id: Optional[str] = None  # For conversation memory (Agentic RAG)
    fast_mode: Optional[bool] = True  # Fast Direct RAG (1 LLM call)

class QuizRequest(BaseModel):
    topic: str
    quiz_type: Optional[str] = "mcq"
    num_questions: Optional[int] = 3
    context_text: Optional[str] = None
    language: Optional[str] = "english"

class QuizScoreRequest(BaseModel):
    topic: str
    score: int
    total: int
    quiz_type: str

class IndexTextRequest(BaseModel):
    text: str
    filename: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_extension(filename: str, allowed: set, label: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"{label} requires one of {sorted(allowed)}. Got: '{ext}'",
        )

def run_attention_tracker():

    global attention_active
    global attention_results

    from backend.attention.attention_tracker import AttentionTracker

    tracker = AttentionTracker()

    try:

        def update_results(results):
            global attention_results
            attention_results = results

        tracker.calculate_attention(
            is_active_callback=lambda: attention_active,
            update_callback=update_results
        )

    except Exception as e:

        logger.error(f"Attention tracker failed: {e}")

    finally:

        attention_active = False

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    from backend.utils.indexer import get_index_stats
    stats = get_index_stats()
    return {"status": "ok", **stats}


@app.get("/files/")
def list_files():
    from backend.utils.indexer import get_indexed_files
    files = get_indexed_files()
    return {"files": files, "total": len(files)}


@app.post("/upload_pdf/")
async def upload_pdf(files: list[UploadFile] = File(...)):
    from backend.utils.indexer import index_file
    from backend.analytics.progress_tracker import record_document_indexed

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    results = []

    for file in files:
        _assert_extension(file.filename, SUPPORTED_UPLOAD_EXTENSIONS, "Upload")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        
        try:
            result = index_file(file_path)
            results.append(result)
            record_document_indexed()
        except Exception as exc:
            logger.error(f"Indexing failed for {file.filename}: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    return {"message": f"Successfully indexed {len(files)} file(s).", "details": results}


@app.post("/build_db/")
async def build_db():
    from backend.utils.indexer import index_directory
    try:
        results = index_directory(UPLOAD_DIR)
        return {"message": "Knowledge base built!", "details": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reset/")
async def reset_app():
    """
    Complete reset of the application state:
    1. Removes all indexed document chunks from ChromaDB.
    2. Deletes all uploaded source files in uploaded_data/files.
    3. Resets student study analytics & quiz scores.
    4. Resets the hybrid retriever and BM25 singleton.
    5. Clears conversation memory sessions.
    """
    try:
        from backend.utils.indexer import clear_index
        from backend.analytics.progress_tracker import reset_analytics
        from backend.agents.memory import get_memory
        import backend.agents.hybrid_retriever as hr

        # 1. Clear ChromaDB vector database
        clear_index()

        # 2. Invalidate retriever & BM25 cache
        hr._retriever_singleton = None

        # 3. Delete all uploaded documents
        deleted_count = 0
        if os.path.exists(UPLOAD_DIR):
            for fname in os.listdir(UPLOAD_DIR):
                fpath = os.path.join(UPLOAD_DIR, fname)
                try:
                    if os.path.isfile(fpath) or os.path.islink(fpath):
                        os.unlink(fpath)
                        deleted_count += 1
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath)
                        deleted_count += 1
                except Exception as del_err:
                    logger.warning("Could not delete %s: %s", fpath, del_err)

        # 4. Reset student analytics & dashboard
        reset_analytics()

        # 5. Clear conversation memory
        try:
            get_memory().clear()
        except Exception:
            pass

        logger.info("Application reset complete. Deleted %d file(s) and cleared ChromaDB.", deleted_count)
        return {
            "status": "success",
            "message": f"Successfully reset application: cleared ChromaDB, deleted {deleted_count} file(s), and reset analytics.",
            "deleted_files": deleted_count
        }
    except Exception as exc:
        logger.error("Reset failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(exc)}")



@app.post("/index_text/")
async def index_text(request: IndexTextRequest):
    try:
        from backend.utils.indexer import _get_chroma, _calculate_chunk_ids
        from backend.utils.document_processor import _make_documents, _subject_from_filename
        from datetime import datetime, timezone
        from backend.analytics.progress_tracker import record_document_indexed

        subject = _subject_from_filename(request.filename)
        upload_ts = datetime.now(timezone.utc).isoformat()

        base_meta = {
            "source": f"ocr://{request.filename}",
            "filename": request.filename,
            "file_type": "text",
            "page": 1,
            "ocr_method": "ocr_manual",
            "subject": subject,
            "upload_timestamp": upload_ts,
        }

        # Create chunks using the processor's _make_documents helper
        chunks = _make_documents(request.text, base_meta)
        if not chunks:
            return {"status": "skipped", "message": "No content to index.", "chunks_added": 0}

        chunks = _calculate_chunk_ids(chunks)

        db = _get_chroma()
        existing = db.get(include=[])
        existing_ids = set(existing["ids"])

        new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
        skipped = len(chunks) - len(new_chunks)

        indexed_at = datetime.now(timezone.utc).isoformat()
        for c in new_chunks:
            c.metadata["indexed_at"] = indexed_at

        if new_chunks:
            new_ids = [c.metadata["id"] for c in new_chunks]
            db.add_documents(new_chunks, ids=new_ids)
            record_document_indexed()
            logger.info("Directly indexed raw OCR text for '%s': %d chunks added.", request.filename, len(new_chunks))
        else:
            logger.info("Raw OCR text for '%s' already indexed – skipped.", request.filename)

        return {
            "status": "success",
            "message": f"Successfully indexed raw OCR text for '{request.filename}'.",
            "chunks_added": len(new_chunks),
            "chunks_skipped": skipped,
            "total_chunks": len(chunks)
        }
    except Exception as exc:
        logger.error(f"Failed to index raw text: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/")
async def chat(request: QueryRequest):
    try:
        from scripts.query_data import query_rag
        result = query_rag(
            request.question,
            mode=request.mode,
            language=request.language,
            session_id=request.session_id,
            fast_mode=request.fast_mode,
        )
        return result
    except Exception as exc:
        logger.error(f"Chat failed: {exc}")
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/ocr/")
async def ocr(image: UploadFile = File(...)):
    _assert_extension(image.filename, SUPPORTED_IMAGE_EXTENSIONS, "OCR")
    try:
        image_bytes = await image.read()
        ocr_result = extract_text_from_image(image_bytes)
        return ocr_result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/quiz/")
async def quiz(request: QuizRequest):
    try:
        from backend.utils.quiz_generator import generate_quiz
        context = request.context_text
        if not context:
            from scripts.query_data import query_rag
            res = query_rag(request.topic)
            context = res.get("context_text", "")

        quiz_data = generate_quiz(
            context=context,
            topic=request.topic,
            quiz_type=request.quiz_type,
            num_questions=request.num_questions,
            language=request.language
        )
        return quiz_data
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/quiz/score/")
async def quiz_score(request: QuizScoreRequest):
    try:
        from backend.analytics.progress_tracker import record_quiz_result
        record_quiz_result(request.topic, request.score, request.total, request.quiz_type)
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dashboard/")
def dashboard():
    try:
        from backend.analytics.progress_tracker import get_dashboard_data
        return get_dashboard_data()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# --- Attention Endpoints ---

@app.post("/attention/start")
def start_attention():

    global attention_active

    if attention_active:
        return {"status": "already_active"}

    attention_active = True

    thread = threading.Thread(
        target=run_attention_tracker,
        daemon=True
    )

    thread.start()

    return {"status": "started"}
@app.post("/attention/stop")
def stop_attention():
    global attention_active
    attention_active = False
    return {"status": "stopped", "last_results": attention_results}

@app.get("/attention/status")
def get_attention_status():
    global attention_active, attention_results
    return {
        "active": attention_active,
        "results": attention_results
    }


@app.get("/attention/video_feed")
def video_feed():
    from backend.attention.attention_tracker import gen_frames
    return StreamingResponse(
        gen_frames(lambda: attention_active),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
