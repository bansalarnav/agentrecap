# agentrecap

Generate a local, metadata-only HTML report from your Codex and Claude Code sessions.

```bash
uv run agentrecap
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

During report generation, `agentrecap` downloads the current pricing catalog from [models.dev](https://models.dev) and caches it under the report's `data/` directory. If the refresh fails, it uses the cached catalog. If no catalog is available, the report still builds with costs marked unavailable.

Costs use standard API rates per million tokens for uncached input, cache reads, five-minute and one-hour cache writes, output, and separately priced reasoning when available. They do not include subscriptions, discounts, batch pricing, or long-context tiers. Unknown model IDs remain unmatched instead of receiving a guessed price.

When comparing with ccusage, use its source-specific `ccusage claude monthly` and `ccusage codex monthly` reports. The unqualified `ccusage monthly` report can include other detected agents that are outside agentrecap's Claude-and-Codex scope.

## Metric notes

A run starts at a provider's model-run prompt and ends at its corresponding terminal event. Codex exposes explicit completion and abort events. Claude has no equivalent terminal event in these logs, so its end is inferred from the last assistant event before the next prompt. Incomplete runs remain visible.

Imported or resumed history can contain blocks of events with nearly identical timestamps. Runs with model or tool activity and a timestamp-derived duration of 100 ms or less are marked `collapsed_timestamps` and excluded from duration summaries.

Delegated and subagent prompts count as runs and are marked with `is_sidechain`. Tool-call counts include request-side events only. Codex token snapshots are converted from cumulative values into per-call increments, with repeated zero-delta snapshots discarded.

Claude Code can repeat one API request's usage across multiple transcript records. Records with the same request ID are counted once using the final usage values.

Reasoning tokens are a subset of output tokens. Separate reasoning counts are currently available for Codex but not Claude. End-to-end output tokens per second include reasoning, tool execution, and orchestration time; they are not a measure of raw model decode throughput.
