"""Where runs live.

FileStore     reads the tracked files: data/transcripts, data/runs/<backend>.json (accepted runs)
              and evals/results/<backend>.json (accepted eval reports). No database needed.
PostgresStore raw SQL + JSONB. On startup it seeds itself from the files, and `fieldcoach eval`
              records every run in it when DATABASE_URL is set.
Repository    what the API uses: Postgres when DATABASE_URL is set, otherwise files.
"""

from __future__ import annotations

import json
from functools import cached_property
from typing import Any

from fieldcoach.config import Settings
from fieldcoach.schemas import CoachingResult, MeetingResult, RunSnapshot, Transcript
from fieldcoach.transcripts import load_transcripts

BACKEND_PREFERENCE = ("claude", "heuristic")  # the dashboard shows Claude's run when there is one


class FileStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def _transcripts(self) -> dict[str, Transcript]:
        return {t.id: t for t in load_transcripts(self.settings.transcripts_dir)}

    def transcripts(self) -> dict[str, Transcript]:
        return self._transcripts

    def accepted_runs(self) -> list[RunSnapshot]:
        runs = []
        for backend in BACKEND_PREFERENCE:
            path = self.settings.runs_dir / f"{backend}.json"
            if path.is_file():
                runs.append(RunSnapshot.model_validate_json(path.read_text(encoding="utf-8")))
        return runs

    def latest_run(self) -> RunSnapshot | None:
        runs = self.accepted_runs()
        return runs[0] if runs else None

    def eval_report(self, backend: str) -> dict[str, Any] | None:
        path = self.settings.results_dir / f"{backend}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id            TEXT PRIMARY KEY,
    rep_id        TEXT NOT NULL,
    rep_name      TEXT NOT NULL,
    meeting_date  DATE NOT NULL,
    synthetic     BOOLEAN NOT NULL CHECK (synthetic),
    doc           JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    backend          TEXT NOT NULL,
    model            TEXT,
    prompt_versions  JSONB NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL,
    synthetic        BOOLEAN NOT NULL DEFAULT TRUE CHECK (synthetic),
    eval_report      JSONB
);
CREATE TABLE IF NOT EXISTS meeting_results (
    run_id         TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    transcript_id  TEXT NOT NULL REFERENCES transcripts(id),
    rep_id         TEXT NOT NULL,
    score          INTEGER NOT NULL,
    result         JSONB NOT NULL,
    coaching       JSONB,
    PRIMARY KEY (run_id, transcript_id)
);
CREATE INDEX IF NOT EXISTS meeting_results_rep_idx ON meeting_results (run_id, rep_id);
"""


class PostgresStore:
    def __init__(self, url: str) -> None:
        self.url = url

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.url, row_factory=dict_row)

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def bootstrap(self, files: FileStore) -> None:
        """Create tables, load transcripts, and load any accepted run that isn't stored yet."""
        self.init_schema()
        self.upsert_transcripts(list(files.transcripts().values()))
        for run in files.accepted_runs():
            if not self.has_run(run.run_id):
                self.save_run(run, files.eval_report(run.backend))

    def upsert_transcripts(self, transcripts: list[Transcript]) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO transcripts (id, rep_id, rep_name, meeting_date, synthetic, doc)
                VALUES (%s, %s, %s, %s, TRUE, %s)
                ON CONFLICT (id) DO UPDATE
                   SET rep_id = EXCLUDED.rep_id, rep_name = EXCLUDED.rep_name,
                       meeting_date = EXCLUDED.meeting_date, doc = EXCLUDED.doc
                """,
                [(t.id, t.rep_id, t.rep_name, t.meeting_date, Jsonb(t.model_dump(mode="json")))
                 for t in transcripts],
            )

    def has_run(self, run_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM runs WHERE id = %s", (run_id,)).fetchone() is not None

    def save_run(self, run: RunSnapshot, eval_report: dict | None = None) -> None:
        from psycopg.types.json import Jsonb

        coaching = {c.transcript_id: c for c in run.coaching}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (id, backend, model, prompt_versions, created_at, eval_report)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET eval_report = EXCLUDED.eval_report
                """,
                (run.run_id, run.backend, run.model, Jsonb(run.prompt_versions), run.created_at,
                 Jsonb(eval_report) if eval_report else None),
            )
            cur.executemany(
                """
                INSERT INTO meeting_results (run_id, transcript_id, rep_id, score, result, coaching)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, transcript_id) DO UPDATE
                   SET score = EXCLUDED.score, result = EXCLUDED.result, coaching = EXCLUDED.coaching
                """,
                [
                    (run.run_id, m.transcript_id, m.rep_id, m.score, Jsonb(m.model_dump(mode="json")),
                     Jsonb(coaching[m.transcript_id].model_dump(mode="json"))
                     if m.transcript_id in coaching else None)
                    for m in run.meetings
                ],
            )

    def transcripts(self) -> dict[str, Transcript]:
        with self._connect() as conn:
            rows = conn.execute("SELECT doc FROM transcripts ORDER BY id").fetchall()
        return {r["doc"]["id"]: Transcript.model_validate(r["doc"]) for r in rows}

    def latest_run(self) -> RunSnapshot | None:
        with self._connect() as conn:
            run = conn.execute(
                """
                SELECT id, backend, model, prompt_versions, created_at FROM runs
                ORDER BY (backend = 'claude') DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if run is None:
                return None
            rows = conn.execute(
                "SELECT result, coaching FROM meeting_results WHERE run_id = %s ORDER BY transcript_id",
                (run["id"],),
            ).fetchall()
        return RunSnapshot(
            run_id=run["id"],
            backend=run["backend"],
            model=run["model"],
            prompt_versions=run["prompt_versions"],
            created_at=run["created_at"],
            meetings=[MeetingResult.model_validate(r["result"]) for r in rows],
            coaching=[CoachingResult.model_validate(r["coaching"]) for r in rows if r["coaching"]],
        )

    def eval_report(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT eval_report FROM runs WHERE id = %s", (run_id,)).fetchone()
        return row["eval_report"] if row else None

    def rep_scores(self, run_id: str) -> list[dict[str, Any]]:
        """Average score per rep for one run, aggregated in SQL."""
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT t.rep_id, t.rep_name, COUNT(*) AS meetings, ROUND(AVG(m.score), 1) AS avg_score
                  FROM meeting_results m JOIN transcripts t ON t.id = m.transcript_id
                 WHERE m.run_id = %s
                 GROUP BY t.rep_id, t.rep_name
                 ORDER BY avg_score
                """,
                (run_id,),
            ).fetchall()


class Repository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.files = FileStore(settings)
        self.db = PostgresStore(settings.database_url) if settings.database_url else None
        if self.db:
            self.db.bootstrap(self.files)

    @property
    def storage(self) -> str:
        return "postgres" if self.db else "files"

    def transcripts(self) -> dict[str, Transcript]:
        return self.db.transcripts() if self.db else self.files.transcripts()

    def latest_run(self) -> RunSnapshot | None:
        return self.db.latest_run() if self.db else self.files.latest_run()

    def eval_report(self, run: RunSnapshot) -> dict[str, Any] | None:
        report = self.db.eval_report(run.run_id) if self.db else None
        return report or self.files.eval_report(run.backend)
