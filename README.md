# triage0: Evidence Checker for AI-Written Bug Reports

**triage0** is an open, terminal-native verification harness and GitHub Action designed for open-source maintainers. Rather than guessing whether a security report "sounds like AI" (which risks discarding real AI-assisted vulnerability findings), triage0 extracts every concrete claim from a bug report and tests it deterministically against the project's source code and history at the exact reported version inside an isolated ephemeral sandbox.

Maintainers receive a one-screen evidence summary where every claim is marked **`CONFIRMED`**, **`CONTRADICTED`**, or **`CANT_CHECK`**, linked directly to the exact git commit, file line, or command output—**never an automatic rejection or style verdict**.

---

## Why This Exists

In 2026, low-effort, hallucinated bug reports went from an annoyance to a crisis that forced major open-source security programs to pause or shut down:
* **`curl`:** Dropped its bug bounty on January 31, 2026, and closed its vulnerability inbox for a month in July 2026 after ~20% of submissions turned out to be fabricated slop handled by a 7-person volunteer team.
* **Linux Kernel:** In May 2026, Linus Torvalds described the private kernel security list as almost unmanageable due to a flood of duplicate and low-quality AI reports (5–10/day vs. 2–3/week two years prior).
* **Program Suspensions:** Nextcloud suspended paid bounties in April 2026 waiting for reliable filtering, and Google paused product submissions to its open-source bounty program on October 1, 2026.
* **Platform Volume:** HackerOne submissions rose 76% YoY (valid rate stuck at 25%), Bugcrowd's triage queue spiked 334% in three weeks of March 2026, and Elastic received 1,390+ reports in H1 2026 (more than 2024 and 2025 combined).

**The Core Principle (Zero False-Flag Guarantee):** Missing a real vulnerability costs far more than reading one more fake. In triage0, a claim can **only** be marked `CONTRADICTED` on hard, deterministic evidence from `git`, AST parsers, or compiler tools. Model judgment alone can at best output `CANT_CHECK`.

---

## Quick Start (CLI & GitHub Action)

### 1. Local CLI Usage
Clone the repository, point `triage0` at a bug report (Markdown, HackerOne JSON export, or GitHub Advisory) and a target git repository:

```bash
# Verify a single report against a local or remote repository
triage0 verify \
  --report ./datasets/curl_slop_49/h1_report_01.md \
  --repo [https://github.com/curl/curl.git](https://github.com/curl/curl.git)

# Run the full benchmark evaluation harness
triage0 eval --suite curl --include-baseline

```

## Codebase Layout

The codebase is modularized into distinct packages around a shared `core/` contract, allowing parallel development across Extraction/Interface, Sandbox/Checks, and Data/Evaluation.

```text
triage0/
├── pyproject.toml                   # Package metadata, dependencies, CLI entrypoint (`triage0`), pytest config
├── Dockerfile.sandbox               # Air-gapped container image (universal-ctags, tree-sitter, clang, ASan, git)
├── action.yml                       # GitHub Action definition for automated advisory/PR evidence comments
├── README.md                        # Project documentation and auto-generated benchmark accuracy tables
│
├── src/
│   └── triage0/
│       ├── __init__.py
│       │
│       ├── core/                    # SHARED DOMAIN CONTRACTS (Locked Week 1)
│       │   ├── models.py            # NormalizedReport, VerificationPlan, and typed Claim variants
│       │   ├── evidence.py          # EvidenceResult & DeterministicProof (enforces zero-LLM-contradiction rule)
│       │   ├── interfaces.py        # Abstract Base Classes: IntakeAdapter, SandboxProvider, BaseClaimChecker
│       │   └── config.py            # Environment config, model settings, sandbox resource limits
│       │
│       ├── cli/                     # TERMINAL INTERFACE & OUTPUT RENDERERS
│       │   ├── main.py              # CLI command definitions (`verify`, `eval`, `index`)
│       │   ├── tui.py               # Live Terminal UI streaming sandbox PTY commands & claim status spinners
│       │   └── renderers/
│       │       ├── terminal.py      # Renders one-screen ANSI summary with clickable file/line links
│       │       ├── markdown.py      # Formats portable Markdown reports for GitHub Action / H1 comments
│       │       └── json_export.py   # Serializes structured runs for evaluation scoring & tracing
│       │
│       ├── intake/                  # REPORT INGESTION ADAPTERS
│       │   ├── markdown_adapter.py  # Normalizes raw pasted Markdown or security mailing list emails
│       │   ├── hackerone_adapter.py # Extracts report text, metadata, and versions from HackerOne exports
│       │   └── github_adapter.py    # Parses GitHub Private Vulnerability Reporting API payloads
│       │
│       ├── extraction/              # CLAIM EXTRACTION & NORMALIZATION
│       │   ├── decomposer.py        # Agent A: Parses unstructured prose into atomic Pydantic Claim objects
│       │   ├── skeptic.py           # Agent B: Audits claims for testability; flags vague prose as CANT_CHECK
│       │   └── prompts/             # System prompts and structured output schemas
│       │
│       ├── sandbox/                 # EPHEMERAL COMPUTE & ISOLATION LAYER
│       │   ├── manager.py           # Orchestrates container boot, git worktree mount, network cutoff, & teardown
│       │   ├── providers/
│       │   │   ├── docker_local.py  # Local Docker/Podman/gVisor container backend for laptop/CI runs
│       │   │   └── microvm_cloud.py # Ephemeral cloud sandbox backend for parallel benchmark execution
│       │   └── daemon/              # Runs INSIDE the isolated container
│       │       ├── server.py        # Internal JSON-RPC/stdio bridge between host orchestrator and container
│       │       ├── pty_session.py   # Stateful bash PTY manager for OpenCode-style terminal investigation
│       │       └── proof_gate.py    # Deterministic verification gate that re-runs commands to seal proofs
│       │
│       ├── checkers/                # STATIC, HISTORICAL & DYNAMIC VERIFICATION
│       │   ├── router.py            # Dispatches extracted claims concurrently across verification tracks
│       │   ├── tools/               # Deterministic CLI wrappers inside the sandbox
│       │   │   ├── git_tools.py     # Fuzzy tag resolution, worktree checkout, `git log -S/-L`, `git blame`
│       │   │   ├── ast_tools.py     # `tree-sitter` & `universal-ctags` symbol lookup and exact line readers
│       │   │   ├── callgraph.py     # `clang` tooling / AST static call-path reachability verifier
│       │   │   └── build_tools.py   # CMake/Autoconf parser for default build flags & ASan compilation
│       │   └── agents/              # Bounded multi-agent investigators
│       │       ├── code_agent.py    # Investigates missing symbols, fuzzy paths, and git function renames
│       │       ├── history_agent.py # Verifies post-version fixes and adjudicates semantic duplicate candidates
│       │       └── poc_agent.py     # [Stretch] Configures ASan builds and runs PoC reproduction steps
│       │
│       ├── storage/                 # PERSISTENCE, REPO CACHE & VECTOR INDEX
│       │   ├── db.py                # Database abstraction (SQLite + sqlite-vec locally; Postgres + pgvector)
│       │   ├── repo_cache.py        # Manages local bare git repositories for fast worktree spawning
│       │   └── indexers/
│       │       ├── osv_sync.py      # Syncs project advisories (e.g., curl `vuln.json` OSV format)
│       │       └── vector_index.py  # Generates and queries embeddings of past reports & commit logs
│       │
│       └── eval/                    # BENCHMARK HARNESS & METRICS
│           ├── harness.py           # Replays labelled datasets through the pipeline and records traces
│           ├── baseline.py          # Frontier LLM-only baseline (reads report + repo with no tools)
│           ├── metrics.py           # Computes flag rate, false-flag rate (target 0%), cost, and time saved
│           └── table_gen.py         # Writes benchmark comparison tables directly into README.md
│
├── datasets/                        # GROUND TRUTH EVALUATION DATASETS
│   ├── curl_slop_49/                # 49 confirmed fake curl reports from public HackerOne links
│   ├── curl_cves_215/               # 215 confirmed curl CVEs (`vuln.json`) and disclosed original reports
│   └── held_out_project/            # Disclosed reports from a second OSS project (e.g., Nextcloud / Node.js)
│
└── tests/                           # AUTOMATED TEST SUITE
    ├── unit/                        # Schema validation, adapter parsing, and proof-gate invariant tests
    └── integration/                 # End-to-end sandbox replay tests on sample fake and real reports
```