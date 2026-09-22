"""Shared paths, credential readers and JSONL helpers. Credentials are read at call time, never stored."""
import json, re
from pathlib import Path
import psycopg2, psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "qwen_discovery" / "sample.jsonl"
GOLD_DIR = ROOT / "gold"
LABELS_DIR = ROOT / "labels"
RESULTS_DIR = ROOT / "results"
_CRED = Path("<credentials-file>")
_AI_ENV = Path("<ai-services-env>")
_JEV_ENV = Path("<bench-repo>/.env")


def _env_value(path: Path, key: str) -> str:
    for line in path.read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"{key} missing in {path}")


def pcai_key() -> str:
    return _env_value(_AI_ENV, "PCAI_API_KEY")


def jev_key() -> str:
    return _env_value(_JEV_ENV, "JEV_API_KEY")


def db():
    row = next(l for l in _CRED.read_text().splitlines() if "<db-row>" in l)
    pw = re.search(r"Admin: \*\*<db-user> / ([^*]+)\*\*", row).group(1).strip()
    conn = psycopg2.connect(host="<db-host>", port=5432, dbname="<db-name>",
                            user="<db-user>", password=pw, connect_timeout=8)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in read_jsonl(path) if "id" in r}
