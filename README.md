# agentrecap

Generate a local, metadata-only HTML report from your Codex and Claude Code sessions.

```bash
uv run agentrecap
```

By default, `agentrecap` reads `~/.codex/sessions` and `~/.claude/projects`, then writes the report to `~/.agentrecap/reports/<timestamp>/index.html`. When it finishes, it asks whether you want to open the report in your browser. Use `--open` to open it immediately without the prompt.

```bash
agentrecap \
  --codex-input /path/to/codex/sessions \
  --claude-input /path/to/claude/projects \
  --output-dir /path/to/report \
  --title "My agent usage report"
```

The report includes:

- Headline thread, run, event, duration, tool-call, and token metrics.
- Codex and Claude comparisons.
- Model usage, cache ratios, reasoning-token metrics, and estimated API costs.
- Tool usage and failure ratios.
- Run-duration, response-gap, thread-length, token, cache, and tool-call charts.
- Human-readable, metadata-only CSV files under the report's `data/` directory.

The generated report does not include transcript contents. Thread, run, event, agent, and tool-call identifiers are hashed before they are written.

## Cost estimates

During report generation, `agentrecap` downloads the current pricing catalog from [models.dev](https://models.dev) and caches it under the report's `data/` directory. If the refresh fails, it uses the cached catalog. If no catalog is available, the report still builds with costs marked unavailable.

Costs use standard API rates per million tokens for uncached input, cache reads, cache writes, output, and separately priced reasoning when available. They do not include subscriptions, discounts, batch pricing, or long-context tiers. Unknown model IDs remain unmatched instead of receiving a guessed price.

## Metric notes

A run starts at a provider's model-run prompt and ends at its corresponding terminal event. Codex exposes explicit completion and abort events. Claude has no equivalent terminal event in these logs, so its end is inferred from the last assistant event before the next prompt. Incomplete runs remain visible.

Imported or resumed history can contain blocks of events with nearly identical timestamps. Runs with model or tool activity and a timestamp-derived duration of 100 ms or less are marked `collapsed_timestamps` and excluded from duration summaries.

Delegated and subagent prompts count as runs and are marked with `is_sidechain`. Tool-call counts include request-side events only. Codex token snapshots are converted from cumulative values into per-call increments, with repeated zero-delta snapshots discarded.

Reasoning tokens are a subset of output tokens. Separate reasoning counts are currently available for Codex but not Claude. End-to-end output tokens per second include reasoning, tool execution, and orchestration time; they are not a measure of raw model decode throughput.
