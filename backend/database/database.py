import os
import sys
import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# ------------------------------------------------------------------------------
# DATABASE_URL — required, no fallback
# ------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║              FATAL: DATABASE_URL is not set                 ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  DocuLingua requires PostgreSQL. Set DATABASE_URL before     ║\n"
        "║  starting the application.                                   ║\n"
        "║                                                              ║\n"
        "║  Example (Docker Compose):                                   ║\n"
        "║    DATABASE_URL=postgresql://doculingua:secret@db:5432/doculingua\n"
        "║                                                              ║\n"
        "║  Example (local PostgreSQL):                                 ║\n"
        "║    DATABASE_URL=postgresql://user:password@localhost:5432/doculingua\n"
        "║                                                              ║\n"
        "║  Steps to fix:                                               ║\n"
        "║    1. cp .env.example .env                                   ║\n"
        "║    2. Edit .env and set POSTGRES_PASSWORD and DATABASE_URL   ║\n"
        "║    3. docker compose up  (or export DATABASE_URL=... locally)║\n"
        "╚══════════════════════════════════════════════════════════════╝\n",
        file=sys.stderr,
    )
    sys.exit(1)

if DATABASE_URL.startswith("sqlite"):
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║            ERROR: SQLite is not supported                   ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  DATABASE_URL points to SQLite, but DocuLingua requires     ║\n"
        "║  PostgreSQL for production use.                              ║\n"
        "║                                                              ║\n"
        "║  Set DATABASE_URL to a PostgreSQL connection string:         ║\n"
        "║    postgresql://user:password@host:5432/doculingua           ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n",
        file=sys.stderr,
    )
    sys.exit(1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # discard stale connections before handing them out
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ------------------------------------------------------------------------------
# Connection verification with exponential-backoff retry
# ------------------------------------------------------------------------------

def verify_connection(max_attempts: int = 3) -> None:
    """
    Execute a trivial query to confirm the database is reachable.

    Retries up to *max_attempts* times with exponential backoff:
        attempt 1 — immediate
        attempt 2 — wait 1 s
        attempt 3 — wait 2 s

    Raises RuntimeError after all attempts are exhausted so callers can
    decide whether to abort (startup) or surface a 503 (health-check).
    """
    delay = 1  # seconds before the *next* attempt
    last_exc: Exception = RuntimeError("No connection attempt was made.")

    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return  # success — exit immediately
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                print(
                    f"  [DB] Connection attempt {attempt}/{max_attempts} failed "
                    f"({exc!s}). Retrying in {delay}s…",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay *= 2  # 1 s → 2 s → 4 s …

    raise RuntimeError(
        f"Cannot connect to the database after {max_attempts} attempt(s).\n"
        f"  Last error : {last_exc}\n"
        f"  DATABASE_URL: {DATABASE_URL[:DATABASE_URL.find('@') + 1]}…  "
        f"(credentials redacted)\n"
        "  Check that PostgreSQL is running and the credentials are correct.\n"
        "  If using Docker Compose:  docker compose up -d db"
    ) from last_exc


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
