"""项目配置。所有可调参数集中在这里。"""
import os
from pathlib import Path


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"F:\hf-cache")

def _load_dotenv(path: Path) -> None:
    """读取项目根目录的 .env；真实环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(PROJECT_ROOT / ".env")

VAULT = Path(os.environ.get("OBSIDIAN_VAULT", PROJECT_ROOT / "notes"))

COURSE_DIRS = [
    d.strip()
    for d in os.environ.get("COURSE_DIRS", "电子电路基础").split(",")
    if d.strip()
]

EXCLUDE_DIR_PREFIXES = ("lec", "_")

INDEX_DIR = Path(os.environ.get("INDEX_DIR", PROJECT_ROOT / "index"))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 100
CHUNK_MIN_CHARS = 50

RECALL_K = 20
RRF_K = 60
TOP_K = 5