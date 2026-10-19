"""Postgres round trip. Runs only when FIELDCOACH_TEST_DATABASE_URL points at a throwaway database."""

import os

import pytest

from fieldcoach.store import FileStore, PostgresStore

URL = os.environ.get("FIELDCOACH_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set FIELDCOACH_TEST_DATABASE_URL to a throwaway database")


def test_bootstrap_and_read_back(settings):
    files = FileStore(settings)
    db = PostgresStore(URL)
    db.bootstrap(files)
    db.bootstrap(files)  # idempotent

    assert len(db.transcripts()) == 20
    run = db.latest_run()
    expected = files.latest_run()
    assert run.run_id == expected.run_id
    assert [m.score for m in run.meetings] == [m.score for m in expected.meetings]
    assert len(run.coaching) == len(expected.coaching)
    assert db.eval_report(run.run_id)["synthetic"] is True

    per_rep = db.rep_scores(run.run_id)
    assert len(per_rep) == 5 and sum(r["meetings"] for r in per_rep) == 20
