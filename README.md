# agentrecap

Generate a local, metadata-only HTML report from your Codex and Claude Code sessions.

```bash
uvx agentrecap
# or
pipx run agentrecap
```

If you do not have `uvx` or `pipx` installed, clone the repository and run it with Python:

```bash
git clone https://github.com/bansalarnav/agentrecap.git
cd agentrecap
python3 -m venv .venv && .venv/bin/pip install .
.venv/bin/python -m agentrecap.cli
```

The local virtual environment keeps `agentrecap` and its dependencies separate from your existing Python installation.

## Overview

`agentrecap` only reads your session data and writes its report to a separate output directory. It will not modify your existing environment or any existing session data.

By default, `agentrecap` reads active and archived sessions from `~/.codex` and Claude Code sessions from `~/.claude/projects`, then writes the report to `~/.agentrecap/reports/<timestamp>/index.html`. When it finishes, it asks whether you want to open the report in your browser. Use `--open` to open it immediately without the prompt.

```bash
agentrecap \
  --codex-input /path/to/codex/home \
  --claude-input /path/to/claude/projects \
  --output-dir /path/to/report \
  --title "My agent usage report"
```

The report includes:

- Headline recent, month-to-date, and all-time estimated API costs alongside usage metrics.
- Codex and Claude comparisons.
- Model usage, cache ratios, reasoning-token metrics, and monthly estimated API costs.
- Run-duration, response-gap, thread-length, token, cache, and tool-call charts.
- Human-readable, metadata-only CSV files under the report's `data/` directory.

The generated report does not include transcript contents. Thread, run, event, agent, and tool-call identifiers are hashed before they are written.

The report's `data/threads.csv` is the complete metadata-only event export used to generate every analysis table. It retains per-request token categories, hashed message/request identities, explicit speed and service-tier metadata, Claude's nullable reported cost, and a `thread_speed_status` of `standard`, `fast`, `mixed`, or `unknown`. The one-row-per-thread aggregate is written separately as `data/thread_summary.csv`; deduplicated per-call pricing inputs and estimates are in `data/model_calls.csv`.

## Cost estimates

For cost estimation, `agentrecap` uses the pricing catalog from [models.dev](https://models.dev).

Every request is priced before model and month totals are aggregated. Estimates distinguish uncached input, cache reads, five-minute cache writes, one-hour cache writes, unclassified cache creation, output, and separately priced reasoning when available. Generic models.dev context tiers are selected from each request's complete input footprint, so model-specific 200K, 256K, 272K, and future thresholds can be honored. A bounded fallback supplies the former Anthropic long-context rates for logged Sonnet 4 and 4.5 calls over 200K before Anthropic retired that 1M-context beta on April 30, 2026.

Explicit historical `fast`/priority metadata uses the provider's fast-mode price from models.dev; `default` is treated as standard. When a historical record has no speed metadata, it remains `unknown` and is estimated at the standard API rate as a clearly marked fallback—today's local configuration is never applied retroactively. Claude's raw top-level `costUSD`, when present, is retained only as `reported_cost_usd` provenance. All report cards, tables, monthly totals, and displayed exports use the independently calculated `estimated_cost_usd`.

These are API-equivalent token estimates, not ChatGPT, Codex, Claude, or Claude Code subscription spend or credit consumption. They do not include negotiated discounts, batch pricing, or provider/platform charges not represented in the logs. Unknown model IDs and fast modes without an explicit catalog price remain unpriced rather than receiving a guessed rate.


## Development

The project requires Python 3.10 or newer and uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync
uv run agentrecap --help
```

To generate a report from local session data while developing:

```bash
uv run agentrecap --output-dir ./report
```


Build the source distribution and wheel with:

```bash
uv build
```
