# Serving image for the ragpipe API: BM25 retrieval + grounded answers + citation
# verification over HTTP.
#
# ## Why this installs so little
#
# The project's full dependency set is 973 MB installed, 516 MB of which is torch. The
# served path uses none of it: `ragpipe serve` runs BM25 (pure Python plus scipy) and
# the citation verifier (pure Python), and calls generation over HTTP. Dense retrieval,
# cross-encoder reranking and the Qdrant client are in extras and stay out of the image.
#
# That is not a packaging preference, it is the same finding as reports/ann_recall.md:
# at 13,423 chunks the parts of this pipeline that need a GPU-shaped dependency tree are
# the parts that measurably did not pay for themselves.
#
# ## Why the corpus is a volume, not a layer
#
# `data/` is 290 MB of third-party PDFs and their derivatives, is gitignored, and is not
# this project's to redistribute. The chunk file is mounted at runtime. An image that
# baked it in could not be built from a clean clone, which would make the Dockerfile a
# thing that works only on this laptop.
FROM python:3.11-slim

# Unbuffered so container logs appear in order under `docker compose logs`; no .pyc
# because the layer is read-only at runtime anyway.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Paths in ragpipe/__init__.py are derived from the source tree by default, which is
    # wrong once the package lives in site-packages. This is where the corpus is mounted.
    RAGPIPE_ROOT=/app

WORKDIR /app

# Dependencies before source, so editing a module does not reinstall the dependency tree.
COPY pyproject.toml README.md ./
COPY src/ragpipe/__init__.py ./src/ragpipe/__init__.py
RUN pip install --no-cache-dir .

COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Non-root, and it owns nothing it does not need to write. The corpus mount is read-only.
RUN useradd --create-home --uid 10001 ragpipe
USER ragpipe

EXPOSE 8000

# `/health` distinguishes "still building the index" from "broken", which is exactly what
# a healthcheck needs: index construction over 13,423 chunks takes seconds, and an
# orchestrator that killed the container during it would never let it start.
HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=6 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if __import__('json').load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['ready'] else 1)"

# Generation stays enabled: without GEMINI_API_KEY the service comes up retrieval-only
# and says so on /health, rather than refusing to start over a missing third-party key.
CMD ["ragpipe", "serve", "--host", "0.0.0.0", "--port", "8000"]
