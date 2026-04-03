from pathlib import Path

MODEL_NAME = "gemini-3.1-pro-preview"
SEGMENT_LENGTH_S = 30
TOP_K_VIDEOS = 20
# Over-select candidates to handle unavailable videos
TOP_K_CANDIDATES = 40

API_CALL_DELAY_S = 1.0

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
BUILDERS_CACHE_DIR = CACHE_DIR / "builders"
ANSWERS_CACHE_DIR = CACHE_DIR / "answers"
RESULTS_DIR = PROJECT_ROOT / "results"

MINERVA_URL = "https://storage.googleapis.com/neptunedata/minerva.json"
MINERVA_LOCAL = DATA_DIR / "minerva.json"
DURATIONS_CACHE = CACHE_DIR / "durations.json"

# Three-tier mixed routing thresholds
SPEECH_DENSE_WORD_THRESHOLD = 30
VISUALLY_ACTIVE_WORD_THRESHOLD = 50

# Low-FPS policy
LOW_FPS_RATE = 0.2

# Low-resolution policy (~64 tokens per frame)
LOW_RES_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_LOW"
