"""Command line: fieldcoach eval | accept | analyze | load | serve | readme."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from fieldcoach.config import Settings, get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fieldcoach",
        description="FieldCoach: AI coaching for in-person field sales (synthetic data demo).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("eval", help="rerun the pipeline and every eval, then print before/after vs the accepted baseline")
    p.add_argument("--backend", choices=["auto", "claude", "heuristic"], default="auto",
                   help="auto = claude when ANTHROPIC_API_KEY is set, otherwise the keyword baseline")
    p.add_argument("--samples", type=int, default=3, help="runs per transcript for the consistency check (default 3)")
    p.add_argument("--limit", type=int, help="only the first N transcripts, for a quick smoke test")
    p.add_argument("--fail-on-regression", action="store_true", help="exit with status 2 if any metric got worse")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("accept", help="promote the latest eval run to baseline and refresh the README table")
    p.add_argument("--backend", choices=["claude", "heuristic"], help="default: the backend that ran most recently")
    p.add_argument("--force", action="store_true", help="accept a partial run (made with --limit)")
    p.set_defaults(func=cmd_accept)

    p = sub.add_parser("analyze", help="analyze and coach one transcript and print a readable report")
    p.add_argument("transcript", help="a transcript id such as T13, or a path to a transcript file")
    p.add_argument("--backend", choices=["auto", "claude", "heuristic"], default="auto")
    p.add_argument("--json", action="store_true", help="print the raw JSON instead")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("load", help="create tables, then load transcripts and accepted runs into Postgres")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("serve", help="run the dashboard and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("readme", help="rewrite the README eval table from the accepted baselines")
    p.set_defaults(func=cmd_readme)

    args = parser.parse_args(argv)
    return args.func(args, get_settings()) or 0


# ---------------------------------------------------------------- eval / accept


def _latest_paths(settings: Settings, backend: str) -> tuple[Path, Path]:
    return (settings.results_dir / f"latest-{backend}.json", settings.runs_dir / f"latest-{backend}.json")


def _accepted_paths(settings: Settings, backend: str) -> tuple[Path, Path]:
    return (settings.results_dir / f"{backend}.json", settings.runs_dir / f"{backend}.json")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def cmd_eval(args, settings: Settings) -> int:
    from fieldcoach.backends import make_backend
    from fieldcoach.evals import REGRESSION_METRICS, compare_table, evaluate, load_golden, metric
    from fieldcoach.pipeline import run_pipeline
    from fieldcoach.transcripts import load_transcripts

    backend = make_backend(args.backend, settings)
    if args.backend == "auto" and backend.name == "heuristic":
        print("No ANTHROPIC_API_KEY found, so this run uses the offline keyword baseline.\n"
              "Set the key (or put it in .env) to evaluate Claude.\n", file=sys.stderr)
    transcripts = load_transcripts(settings.transcripts_dir)
    total = len(transcripts)
    if args.limit:
        transcripts = transcripts[: args.limit]
    golden = load_golden(settings.golden_path)

    from fieldcoach.llm import LLMError

    try:
        output = run_pipeline(backend, transcripts, samples=max(1, args.samples), concurrency=settings.concurrency)
    except LLMError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print("Check ANTHROPIC_API_KEY and FIELDCOACH_MODEL (and FIELDCOACH_EFFORT, if set).", file=sys.stderr)
        return 1
    report = evaluate(output, transcripts, golden)
    snapshot = output.snapshot()

    results_path, run_path = _latest_paths(settings, backend.name)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run_path.write_text(snapshot.model_dump_json(indent=1) + "\n", encoding="utf-8")

    baseline = _read_json(_accepted_paths(settings, backend.name)[0])
    print()
    print(compare_table(baseline, report))
    print()
    if len(transcripts) < total:
        print(f"Partial run: {len(transcripts)} of {total} transcripts, so before/after is not like for like.")
    usage = report["usage"]
    if usage:
        print(f"Claude calls: {usage['calls']} ({usage['cache_hits']} served from cache), "
              f"tokens in/out: {usage['input_tokens']:,}/{usage['output_tokens']:,}")
    print(f"Took {report['seconds']}s. Errors: {len(report['errors'])}")
    for err in report["errors"][:10]:
        print(f"  - {err}")
    print(f"Wrote {results_path.relative_to(settings.root)} and {run_path.relative_to(settings.root)}")

    if settings.database_url:
        try:
            from fieldcoach.store import FileStore, PostgresStore

            db = PostgresStore(settings.database_url)
            db.bootstrap(FileStore(settings))
            db.save_run(snapshot, report)
            print(f"Recorded run {snapshot.run_id} in Postgres")
        except Exception as exc:  # the eval itself succeeded; storage is a bonus here
            print(f"Could not record the run in Postgres: {exc}", file=sys.stderr)

    print(f"\nIf the numbers look right: fieldcoach accept --backend {backend.name}")

    if not output.results.get(0):
        return 1
    if args.fail_on_regression and baseline:
        worse = []
        for key, name, better, _ in REGRESSION_METRICS:
            b, a = metric(baseline, key), metric(report, key)
            if a is not None and b is not None and ((a < b) if better == "higher" else (a > b)):
                worse.append(name)
        if worse:
            print("Regressed: " + ", ".join(worse), file=sys.stderr)
            return 2
    return 0


def cmd_accept(args, settings: Settings) -> int:
    backend = args.backend
    if backend is None:
        candidates = [b for b in ("claude", "heuristic") if _latest_paths(settings, b)[0].is_file()]
        if not candidates:
            print("Nothing to accept yet. Run `fieldcoach eval` first.", file=sys.stderr)
            return 1
        backend = max(candidates, key=lambda b: _latest_paths(settings, b)[0].stat().st_mtime)
    latest_results, latest_run = _latest_paths(settings, backend)
    if not latest_results.is_file() or not latest_run.is_file():
        print(f"No latest {backend} run found. Run `fieldcoach eval --backend {backend}` first.", file=sys.stderr)
        return 1
    report = _read_json(latest_results)
    total = len(list(settings.transcripts_dir.glob("*.txt")))
    if report and report.get("transcripts", 0) < total and not args.force:
        print(f"The latest {backend} run covered {report['transcripts']} of {total} transcripts. "
              "Rerun without --limit, or pass --force.", file=sys.stderr)
        return 1
    if report and report.get("errors") and not args.force:
        print(f"The latest {backend} run had {len(report['errors'])} failed calls, for example: "
              f"{report['errors'][0]}\nRerun `fieldcoach eval` (cached answers are reused), or pass --force.",
              file=sys.stderr)
        return 1
    accepted_results, accepted_run = _accepted_paths(settings, backend)
    shutil.copyfile(latest_results, accepted_results)
    shutil.copyfile(latest_run, accepted_run)
    print(f"Accepted the latest {backend} run as the new baseline.")
    cmd_readme(args, settings)
    return 0


def cmd_readme(args, settings: Settings) -> int:
    from fieldcoach.evals import readme_table, update_readme

    claude = _read_json(_accepted_paths(settings, "claude")[0])
    heuristic = _read_json(_accepted_paths(settings, "heuristic")[0])
    if update_readme(settings.readme_path, readme_table(claude, heuristic)):
        print("README eval table updated.")
        return 0
    print("README has no eval-table markers; nothing changed.", file=sys.stderr)
    return 1


# ---------------------------------------------------------------- analyze one transcript


def cmd_analyze(args, settings: Settings) -> int:
    from fieldcoach.backends import make_backend
    from fieldcoach.pipeline import build_meeting_result, coach_meeting
    from fieldcoach.scoring import coaching_benchmarks
    from fieldcoach.store import FileStore
    from fieldcoach.transcripts import parse_transcript

    path = Path(args.transcript)
    if not path.is_file():
        path = settings.transcripts_dir / f"{args.transcript}.txt"
    if not path.is_file():
        print(f"No transcript {args.transcript!r} (looked for a file and for an id in data/transcripts).",
              file=sys.stderr)
        return 1
    transcript = parse_transcript(path.read_text(encoding="utf-8"), source=path.name)
    backend = make_backend(args.backend, settings)
    accepted = FileStore(settings).latest_run()
    benchmarks = coaching_benchmarks(accepted.meetings) if accepted else "No team benchmarks yet."

    meeting = build_meeting_result(transcript, backend.analyze(transcript))
    coaching = coach_meeting(backend, transcript, meeting, benchmarks)
    if args.json:
        print(json.dumps({"analysis": meeting.model_dump(mode="json"),
                          "coaching": coaching.model_dump(mode="json")}, indent=2))
        return 0

    t, m, a = transcript, meeting, meeting.analysis
    ok = sum(c.ok for c in m.checks)
    print(f"{t.id}  {t.rep_name} at {t.account} (synthetic)   score {m.score}/100   backend: {backend.name}")
    print(f"Talk share {m.talk.rep_share:.0%} (target 35-55%), longest rep turn {m.talk.longest_rep_turn_words} words")
    print(f"\nDiscovery questions ({len(m.discovery_turns)} grounded turns):")
    for q in a.discovery_questions:
        print(f"  [{q.turn}] {q.quote}  ({q.kind})")
    print(f"\nObjections ({len(m.objections)} grounded):")
    for o in m.objections:
        print(f"  [{o.turn}] {o.category}, handled {o.handling}: {o.quote}")
        if o.response_quote:
            print(f"        rep [{o.response_turn}]: {o.response_quote}")
    stated = f' [{a.next_step.turn}] "{a.next_step.quote}"' if a.next_step.quote else ""
    print(f"\nNext step: {a.next_step.quality}{stated}")
    print(f"Analysis quotes found in the transcript: {ok}/{len(m.checks)}")

    note = coaching.note
    first_ok = sum(c.ok for c in coaching.first_pass_checks)
    print(f"\nCoaching note (quotes grounded first pass {first_ok}/{len(coaching.first_pass_checks)}"
          f"{', repaired once' if coaching.repaired else ''}{f', {coaching.dropped} dropped' if coaching.dropped else ''}):")
    print(f"  {note.summary}")
    if note.keep_doing:
        print(f"  Keep doing: {note.keep_doing.title} [{note.keep_doing.evidence_turn}] \"{note.keep_doing.evidence_quote}\"")
    for i, s in enumerate(note.suggestions, start=1):
        print(f"  {i}. {s.title}")
        print(f"     {s.what_happened}")
        print(f"     evidence [{s.evidence_turn}]: \"{s.evidence_quote}\"")
        print(f"     try: {s.try_instead}")
    return 0


# ---------------------------------------------------------------- storage / server


def cmd_load(args, settings: Settings) -> int:
    if not settings.database_url:
        print("Set DATABASE_URL first (see .env.example).", file=sys.stderr)
        return 1
    from fieldcoach.store import FileStore, PostgresStore

    db = PostgresStore(settings.database_url)
    db.bootstrap(FileStore(settings))
    run = db.latest_run()
    print(f"Loaded transcripts and accepted runs. Dashboard run: {run.run_id if run else 'none'}")
    return 0


def cmd_serve(args, settings: Settings) -> int:
    import uvicorn

    uvicorn.run("fieldcoach.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
