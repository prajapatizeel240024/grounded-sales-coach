.PHONY: install test eval accept serve up readme

install:  ## install the package and test deps in editable mode
	pip install -e ".[dev]"

test:     ## unit tests (no API key or database needed)
	pytest

eval:     ## rerun the whole pipeline + evals and print before/after vs the accepted baseline
	fieldcoach eval

accept:   ## promote the latest eval run to baseline, refresh the README table and dashboard data
	fieldcoach accept

serve:    ## run the API and dashboard locally on :8000
	fieldcoach serve --port 8000

up:       ## app + Postgres in Docker, dashboard on :8000
	docker compose up --build

readme:   ## rewrite the README eval table from the accepted baselines
	fieldcoach readme
