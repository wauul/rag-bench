# Guided optimization verification — 7 October 2026

## Implementation

The authenticated API and dashboard support dataset/baseline selection, seeded bounded
search, frozen plan review, explicit approval, progress and HTTP-attempt accounting,
cancellation/resume, recorded request-budget increases, tuning/held-out comparison,
JSON report export and saving a new observed configuration. Existing benchmark
execution, LangChain retrieval, LangGraph row graphs, Chroma, Ragas, profiling and
authoritative storage remain in the path. See [workflow semantics](guided-optimization.md).

## Checks

- `python -m scripts.checks`: **137 passed, 11 intentionally skipped**, three existing
  upstream Ragas embedding-wrapper deprecation warnings. Coverage **76.54%**, above
  the repository's recorded threshold. Includes dependency lock/installed compatibility,
  Ruff formatting/lint, mypy and credential-free regression tests.
- Additional final dashboard checks: **2 passed** after the selector/copy/chart changes.
- `bandit -r backend scripts -ll`: no medium/high findings. Low-severity token-counter
  and existing tokenizer-padding false positives are not security vulnerabilities.
- `git diff --check` and final Ruff checks passed.

New tests cover seeded valid candidates/invalid combinations/duplicates; disjoint
stable splits and exploratory acknowledgement; frozen input compatibility; reference
exclusion from generation; actual HTTP sends and Groq SDK retry enforcement; consumed
budget and saved-answer retention across cancellation/resume; fresh-process death
after request reservation; stable trial IDs; incomplete metrics; quality/latency
constraints; baseline ties/no improvement; a deliberately disappointing held-out
winner that does not reopen selection; auth/admission/amendment/save/export routes;
and review/form/dashboard behavior. Ordinary benchmarks pass their existing tests.

## Real-provider smoke

An isolated local store at `.tools/optimization-engineering-data` used the repository's
current pinned full runtime in `.tools/engineering-venv`, real local MiniLM embeddings,
Chroma, LangChain, LangGraph, real Groq generation and real Ragas judging. No scoring
fixtures were used for this experiment. Input was the committed Harbor handbook and
**one** committed reference question. Exploratory tuning was explicitly acknowledged;
there was no independent held-out split. This cannot establish generalization.

Experiment: `647f73f81951470396b1667d51ac0382`. Baseline: 192-token chunks/32 overlap;
candidate: 128-token chunks/16 overlap. Both used MiniLM, candidate_k=3, context_k=1,
no reranking, and the same frozen generation/judge conditions. Ceiling: four trials
and **30 HTTP attempts**. Two tuning trials completed with two answers and all eight
required metric values. **16 actual HTTP attempts**, **6,744 provider-reported tokens**.

Observed equal-weight quality: baseline **0.9825**, candidate **0.9716**. Selection
retained baseline and reported **No improvement found**. These single-question scores
are smoke evidence only. No global optimum, significant difference or performance
gain is claimed. Request accounting matched the trial profiles. The exported report
contains complete coverage and immutable provenance.

Earlier isolated attempts did not reach Groq: the minimal development environment
lacked Transformers, and an older verification environment had incompatible Chroma
cache support. Those records were preserved. A new plan was created in the current
pinned full environment; models/dependencies were not silently changed within a plan.

## Browser verification

Local API: `127.0.0.1:8017`; dashboard: `127.0.0.1:8517`. The Codex browser checked
setup, exact plan review, explicit start, progress, completed comparison table and
quality/latency chart, the no-improvement outcome, exploratory label, JSON download,
and saving an observed configuration. Downloaded JSON was read back: completed status,
two tuning results, baseline selection, matching usage, and no raw documents in the
public plan. Saving created a separate configuration. Desktop and 390px mobile views
were inspected, including the mobile navigation drawer. Final console error list
was empty. Browser cancellation/resume were verified by automated API tests, not
another real-provider run.

Local screenshot evidence: `.tools/optimization-results-desktop.png` and
`.tools/optimization-results-mobile.png`. Local private full report:
`.tools/optimization-real-report.json`. Downloaded public report:
`ragbench-optimization-647f73f81951470396b1667d51ac0382.json` in Downloads.

## Limits and deployment

Optimization initially requires a LangChain + LangGraph baseline. HTTP attempts are
enforced; optional token/wall-time ceilings are not implemented. Cross-trial index
reuse is deliberately conservative: only the same fingerprint-checked trial reuses
its index. Completed selection is frozen; incomplete candidates remain excluded and
visible. Retuning after a completed selection requires a new reviewed experiment.
The workspace retains its existing shared-access authorization model.

The new experiment is **locally implemented and verified, not deployed**. Existing
backend/dashboard manifests and pinned dependencies need no changes or paid services.
No production database, deployment, branch or default configuration was modified.
Git changes are uncommitted, with no unrelated initial changes in this checkout.
