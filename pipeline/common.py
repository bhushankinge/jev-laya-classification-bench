"""Shared paths, credential readers and JSONL helpers. Credentials are read at call time, never stored."""
import json, re
from itertools import zip_longest
from pathlib import Path

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
    import psycopg2  # lazy: bench venvs (e.g. Laya) lack this driver and never call db()/cursor()
    row = next(l for l in _CRED.read_text().splitlines() if "<db-row>" in l)
    pw = re.search(r"Admin: \*\*<db-user> / ([^*]+)\*\*", row).group(1).strip()
    conn = psycopg2.connect(host="<db-host>", port=5432, dbname="<db-name>",
                            user="<db-user>", password=pw, connect_timeout=8)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def cursor(conn):
    import psycopg2.extras
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
    """Ids considered complete for resume purposes. Error rows are excluded so they are retried
    on the next run (a later success line for the same id then sits alongside the old error line)."""
    return {r["id"] for r in read_jsonl(path) if "id" in r and "error" not in r}


def select_rows(rows: list[dict], limit=None, done=(), ids_from=None, balanced=False) -> list[dict]:
    """The rows a labeler should do next.

    ids_from: a set of ids or a jsonl path (e.g. a gold file) to restrict to.
    balanced: interleave vehicles round-robin, so a --limit prefix is not one vehicle.
    The limit is applied BEFORE the done-filter, so resuming keeps the same prefix instead of
    walking further down the sample on every run."""
    if ids_from is not None:
        keep = ids_from if isinstance(ids_from, (set, frozenset, dict)) else {r["id"] for r in read_jsonl(ids_from)}
        rows = [r for r in rows if r["id"] in keep]
    if balanced:
        by_vehicle = {}
        for r in rows:
            by_vehicle.setdefault(r.get("vehicle"), []).append(r)
        rows = [r for group in zip_longest(*(by_vehicle[v] for v in sorted(by_vehicle, key=str)))
                for r in group if r is not None]
    if limit:
        rows = rows[:limit]
    return [r for r in rows if r["id"] not in done]


def latest_by_id(path: Path) -> dict:
    """The last row per id (later lines win), so a retried success supersedes an earlier error row
    for the same id without needing the file to be rewritten in place."""
    out = {}
    for r in read_jsonl(path):
        if "id" in r:
            out[r["id"]] = r
    return out
