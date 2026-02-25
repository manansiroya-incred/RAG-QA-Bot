import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
USER_UPLOADS_DIR = BASE_DIR / "data" / "user_uploads"
SESSION_CHROMA_ROOT = BASE_DIR / "data" / "chroma_db" / "sessions"
SQLITE_INDEX_PATH = BASE_DIR / "data" / "index.sqlite"

# Model configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # Legacy (kept for future use if needed)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Groq Llama API key

EMBED_MODEL = "intfloat/e5-large-v2"  

#LLM_MODEL = "gemini-2.5-flash"
#LLM_MODEL = "gemini-3-flash-preview"
#LLM_MODEL = "gemini-3.1-pro-preview"
LLM_MODEL = "llama-3.3-70b-versatile"  

# Tesseract configuration (legacy OCR - not used)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)
if TESSERACT_CMD:
    os.environ["TESSDATA_PREFIX"] = str(Path(TESSERACT_CMD).parent)

# Vector DB configuration
USE_QDRANT = bool(os.getenv("QDRANT_URL"))
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Chunking defaults
CHUNK_SIZE = 900  
CHUNK_OVERLAP = 250  
TABLE_MAX_COLS = 20
TABLE_MAX_ROWS = 200

# Retrieval defaults
TOP_K = 5
RERANK_TOP_K = 15

# Reranker (cross-encoder)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def ensure_dirs() -> None:
    """Ensure expected directories exist."""
    for path in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        CHROMA_DB_DIR,
        USER_UPLOADS_DIR,
        SESSION_CHROMA_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)