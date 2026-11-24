# water-gte-eval — pinned, reproducible image.
# Build:  docker build -t water-gte-eval .
# Quickstart inside the container (mounts keep outputs on the host):
#   docker run --rm -v "$PWD/out:/app/out" water-gte-eval make-example-processed --out out/processed_demo
#   ... or run the 5-command quickstart from README.md with `docker run --rm
#   -v "$PWD:/app/out" -w /app/out water-gte-eval ...` (see README.md "Docker").
# Dependencies are fully pinned in requirements-lock.txt (pip freeze of the
# tested environment); the package itself is installed --no-deps -e so the lock
# is the single source of versions.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

COPY configs/ configs/
COPY tests/ tests/
COPY docs/ docs/
COPY README.md ./
COPY data/raw/README.md data/raw/README.md

# sanity: the CLI imports and the test suite collects
RUN python -m water_gte_eval --help >/dev/null && python -m pytest --collect-only -q >/dev/null

ENTRYPOINT ["python", "-m", "water_gte_eval"]
CMD ["--help"]
