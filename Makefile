.PHONY: setup manifest fetch stats corpus extract chunk evalset eval ablation sweep gen-eval gen-eval-full golden judge revalidate ann bench curate failures serve up down logs install-agentrag clean lint test

# Dense model and reranker used by the `eval` / `ablation` targets.
EMBED_MODEL ?= bge-small
RERANKER    ?= bge-reranker-base
CANDIDATE_K ?= 50

# Number of documents to sample from each source. Override: make corpus FDA_N=200
FDA_N    ?= 120
CTGOV_N  ?= 40
SEED     ?= 20260813

setup:
	uv sync --extra dev --extra embed --extra vectorstore --extra bedrock

# Stage 1: query both source indexes, sample deterministically, write the manifest.
# Network calls hit only the two JSON/REST indexes — no PDFs downloaded yet.
manifest:
	uv run ragpipe manifest --fda-n $(FDA_N) --ctgov-n $(CTGOV_N) --seed $(SEED)

# Stage 2: download every PDF, pin/verify sha256, classify text layer.
# Idempotent and resumable: re-running skips files whose hash already matches the pin.
fetch:
	uv run ragpipe fetch

# Stage 3: build the Phase 0 checkpoint report from the manifest + fetch results.
stats:
	uv run ragpipe stats

corpus: manifest fetch stats

# Phase 1 stage 1: PDF -> normalised text, section structure, exact identifiers.
# Incremental by content hash — only documents whose sha256 pin changed are redone.
extract:
	uv run ragpipe extract

# Phase 1 stage 2: chunk with every strategy, then cluster near-duplicates.
# `semantic` embeds each sentence, so this is minutes rather than seconds; the other
# two are effectively free. Re-running one strategy is safe — the report re-derives
# the others from chunks on disk rather than dropping them from the comparison.
chunk:
	uv run ragpipe chunk

# Phase 2: the golden retrieval set. Ground truth is character spans, not chunk ids,
# so it stays valid across re-chunking and the strategy comparison is honest.
evalset:
	uv run ragpipe evalset

# Phase 2 baseline: sparse only, no models, seconds to run.
eval:
	uv run ragpipe eval --with-ablations

# Phase 3: the full ablation table — dense, three fusion methods across a weight
# sweep, and cross-encoder reranking. Embeddings are cached by exact embedded text,
# so only the first run pays for them; reranking is the slow part on every run.
ablation:
	uv run ragpipe eval --with-ablations \
	  --dense-models $(EMBED_MODEL) \
	  --fusion-weights 0.1 0.3 0.5 0.7 \
	  --rerank-model $(RERANKER) --candidate-k $(CANDIDATE_K)

# Phase 4: chunk-size sweep. Chunks in memory at each target size and scores with the
# same golden set, so size is the only variable. Embeddings persist via the vector
# cache, so a re-run is free; the first run embeds ~66k chunks and takes ~25 min.
#
# Reports the denominator trap explicitly: `recall@k` rises with chunk size because
# larger chunks mean fewer chunks per section, while `hit@k` is flat. Reading recall
# alone concludes "bigger is better" and ships a worse default — which is what this
# project originally did.
sweep:
	uv run ragpipe sweep --strategy structural --target-chars 512 1024 2048 4096

# Phase 5: grounded generation with verified citations. Needs GEMINI_API_KEY in
# .env.gemini (free key at https://aistudio.google.com/apikey).
#
# BM25 alone as the first stage, deliberately: it has the best refusal separability
# Phase 3 measured (best-threshold accuracy 0.983 on structural, vs 0.667 for min-max
# fusion and 0.942 for rank fusion + rerank, all on structural), so a score gate
# calibrates most cleanly against it. Min-max's 0.725 is `semantic`, not structural.
#
# The binding constraint is the free tier's **20 requests per day, per model** — not
# per minute, which is what the 429's own "retry in 4.4s" text implies. A rejected
# request still spends a quota unit, so retrying into an exhausted daily quota is pure
# loss: the first run of this target burned one model's entire daily budget and scored
# 3 of 80. `--per-slice 4` is 16 queries, which is what fits with retry headroom.
#
# Quota is per model, so `--generator` is how you get another budget. Requests are
# also paced ~3.2s apart for the unpublished per-minute cap.
gen-eval:
	uv run ragpipe gen-eval --per-slice 4 --k 5

# Both tiers in one run: generation + citation location + "does the quote support the
# answer". The judge is a different model from the generator so nothing grades its own
# homework, and it is calibrated against controls first — faithfulness is omitted from
# the report entirely if calibration fails, rather than printed with a caveat.
#
# Costs more requests than `gen-eval`: one per judgment batch on top of one per query.
gen-eval-full:
	uv run ragpipe gen-eval --per-slice 4 --k 5 --judge gemini-3.1-flash-lite

# Phase 6: draft + validate golden Q&A pairs. Needs GEMINI_API_KEY in .env.gemini.
#
# One request per document (deviation #15: ~10 pairs per call, because the free tier
# counts requests not tokens — 20/day/model). Every pair is validated locally with the
# Phase 5 tier-1 verifier before anything is written, so a hallucinated reference answer
# never reaches the curation pass; rejects are persisted with their reason.
#
# Re-validating an existing set costs nothing, so prompt and threshold changes can be
# tested against pairs already on disk without spending quota.
golden:
	uv run ragpipe golden --docs 10 --pairs-per-doc 10

# Phase 6: calibrate the tier-2 support judge before trusting any of its verdicts.
#
# Runs against controls whose answer is known: golden pairs (presumed supported) and
# *constructed negatives* — a real answer paired with another document's evidence, which
# cannot be supported under any reading. A judge that answers "supported" to everything
# scores perfectly on positives and zero on negatives, and a positives-only calibration
# could not see it.
#
# The judge model deliberately differs from the drafting/answering default so nothing
# grades its own homework. Exits non-zero if calibration fails.
judge:
	uv run ragpipe judge --show

# Phase 6: re-run every golden-set check against the pairs already on disk. No API calls,
# dry by default. This exists because three of those checks — the redundancy gate, the
# factual triage, and the review sampler — had zero call sites outside their own module
# until the Phase 6 code-review gate said so, and because "re-validation is free to re-run"
# was true of a throwaway script rather than of any shipped command.
revalidate:
	uv run ragpipe revalidate

# Phase 7: what approximate search costs against the exact baseline this project ships.
#
# Needs a Qdrant **server** — `make up` starts one. Local mode brute-forces every query
# and would report index recall 1.0000 at every `ef`, which is what the first attempt at
# this measurement did; `sweep_recall` now refuses to run against a local client.
#
# Five builds, not one, and that is the whole point of the target. HNSW graph construction
# is randomised: the first version of this measurement built once, reported recall and
# top-k agreement of exactly 1.0000 at Qdrant's default, and concluded that approximation
# costs nothing. Rebuilding did not reproduce either figure. Two sweeps over one built
# index are bit-identical, so the variance is the graph — the report gives a median over
# builds with the observed range, and re-checks the within-build determinism every run.
#
# Overwrites reports/ann_recall.{md,json}. A run with fewer builds is a weaker artifact
# that looks the same, so the build count is recorded in both.
ann:
	uv run ragpipe ann --builds 5

# Phase 8: the failure analysis, assembled from the artifacts rather than from memory.
#
# Reads only committed artifacts plus the chunk file, spends nothing, and omits any
# finding whose artifact is missing instead of estimating it. A mutation test asserts that
# no number in the rendered prose survives replacing the inputs with sentinels — which is
# how writing it caught a corpus figure that three live source files had been quoting
# since before the post-quarantine regeneration.
failures:
	uv run ragpipe failures

# Phase 8: triage the golden set for the defect classes lexical coverage cannot see.
#
# Costs nothing and touches no network. The five validation checks behind `make golden`
# are a bag of words, and Phase 6 measured what that misses: "60 days" against "30 days",
# "may" against "shall", and "shall not submit" against "shall submit" all score coverage
# 1.000. Numbers are already handled; this covers obligation level and polarity.
#
# Writes a worksheet, and never edits the golden set. `ragpipe curate --apply` is the
# separate step that applies verdicts a *human* recorded in corpus/curation_verdicts.json
# — the one artifact here that cannot be regenerated, only redone.
curate:
	uv run ragpipe curate

# Phase 8: per-stage serving latency, as a report rather than a line in a chat log.
#
# Phase 7 measured these on /stats and quoted three single observations (retrieve 0.44 ms,
# generate 6257 ms, verify 0.012 ms) that lived in no reports/*.json — the exact defect
# that phase names twice, and the one that cost the ANN report its headline.
#
# The three stages cannot be sampled the same way, which is why `n` is a column:
# retrieval is local and free (180 queries x 3 passes); verification is local and free but
# needs citations, so it replays the real ones stored in generation_eval.json; generation
# is a third-party call against a 20-per-day free tier, so it is read from that same stored
# run and the `source` column says so. `--live-generate N` measures it fresh, capped at 5.
bench:
	uv run ragpipe bench

# Phase 7: serve retrieval + grounded answers + citation verification on :8000.
#
# Loads the index before binding the port, so a slow start is visibly a slow start rather
# than a fast start that serves 30-second requests. Generation is optional: without
# GEMINI_API_KEY the service comes up retrieval-only and /health says which it is.
#
# /stats reports per-stage p50/p95 over requests actually served (plan deviation #8).
# Retrieval and verification are sub-millisecond; generation is seconds of somebody
# else's network, which is why the stages are timed separately.
demo:
	@# Renders docs/demo.{mp4,srt} from the live service: headless-Chrome frames,
	@# `say` narration, subtitles generated from the same strings. Spends ONE
	@# generation request on a probe, deliberately -- `generator_available: true`
	@# only means a key is configured, and a spent daily quota renders five frames
	@# of retrieval-only output under narration that describes answering.
	bin/build-demo $(ARGS)

serve:
	uv run ragpipe serve

# Phase 7: the container pair. `api` serves; `qdrant` exists so `make ann` can reproduce
# the measurement that concluded the API should not use it at this corpus size.
#
# The API mounts data/chunks read-only, so run `make chunk` first — data/ is gitignored
# and a fresh clone has to generate it. GEMINI_API_KEY is passed from the environment,
# never baked into a layer.
up:
	docker compose up -d --build

# `bin/agentrag` wraps this target plus the colima start and the health wait that have to
# happen around it. Run once to put it on PATH; it echoes what it runs, so these targets
# stay the primary interface.
install-agentrag:
	mkdir -p $(HOME)/.local/bin
	ln -sf "$(CURDIR)/bin/agentrag" $(HOME)/.local/bin/agentrag
	@echo "linked: $(HOME)/.local/bin/agentrag -> $(CURDIR)/bin/agentrag"
	@command -v agentrag >/dev/null || echo "NOTE: $(HOME)/.local/bin is not on your PATH"

down:
	docker compose down

logs:
	docker compose logs -f --tail=50


lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

clean:
	rm -rf data/raw data/fetched.jsonl

test:
	uv run pytest tests -q
