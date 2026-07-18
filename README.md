# agentrecap

Generate a local, metadata-only HTML report from your Codex and Claude Code sessions.

```bash
uvx agentrecap
# or
pipx run agentrecap
```

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

The report's `data/threads.csv` is the complete metadata-only event export used to generate every analysis table. It is retained unchanged so it can be shared as a single portable input; the one-row-per-thread aggregate is written separately as `data/thread_summary.csv`.

## Cost estimates

For cost estimation, `agentrecap` uses the pricing catalog from [models.dev](https://models.dev)

Costs use standard API rates per million tokens for uncached input, cache reads, five-minute and one-hour cache writes, output, and separately priced reasoning when available. They do not include subscriptions, discounts, batch pricing, or long-context tiers. Unknown model IDs remain unmatched instead of receiving a guessed price.


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
