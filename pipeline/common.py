"""Shared paths, credential readers and JSONL helpers.

Secrets come from environment variables (JEV_API_KEY, PCAI_API_KEY, QWEN_ENDPOINT, CRM_DB_DSN,
LAYA_MODEL_DIR). Missing ones are loaded from the file named by LJB_ENV_FILE, default
~/.config/laya-jev-bench.env, which lives outside the repo. See .env.example.
"""
import json, os
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "qwen_discovery" / "sample.jsonl"
GOLD_DIR = ROOT / "gold"
LABELS_DIR = ROOT / "labels"
RESULTS_DIR = ROOT / "results"
ENV_FILE = Path(os.environ.get("LJB_ENV_FILE", Path.home() / ".config" / "laya-jev-bench.env"))


def _load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env_file()


def env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"{key} not set: export it or add it to {ENV_FILE} (see .env.example)")
    return v


def pcai_key() -> str:
    return env("PCAI_API_KEY")


def jev_key() -> str:
    return env("JEV_API_KEY")


def db():
    """Read-only connection to the CRM database from CRM_DB_DSN (postgresql://user:pw@host:5432/db)."""
    import psycopg2  # lazy: bench venvs (e.g. Laya) lack this driver and never call db()/cursor()
    conn = psycopg2.connect(env("CRM_DB_DSN"), connect_timeout=8)
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
