"""Chart generation for agent usage reports."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]

# Every chart make_plots can produce: filename -> (report heading, description).
# The report renders whichever of these exist after a run, in this order.
CHARTS = {
    "run_duration_hist.png": ("Run duration", "Distribution of end-to-end run duration."),
    "run_duration_ecdf.png": ("Run duration percentiles", "Cumulative view of run duration."),
    "response_gap_hist.png": ("Response gaps", "Idle time between an answer and the next prompt."),
    "response_gap_ecdf.png": ("Response gap percentiles", "Cumulative view of user think time."),
    "tool_calls_per_run_hist.png": ("Tool calls per run", "How tool-heavy individual runs are."),
    "tool_calls_per_run_ecdf.png": ("Tool-call percentiles", "Cumulative view of tool calls per run."),
    "user_messages_per_thread_hist.png": ("Messages per thread", "Conversation length by user messages."),
    "user_messages_per_thread_ecdf.png": ("Thread-length percentiles", "Cumulative conversation length."),
    "cache_ratios.png": ("Cache usage", "Cache-read ratios at run and thread level."),
    "run_load_vs_duration.png": ("Load and duration", "Input size and tool calls compared with duration."),
    "tokens_per_run.png": ("Tokens per run", "Input and output token distributions."),
    "reasoning_vs_output_tokens.png": ("Reasoning tokens", "Reasoning volume and share of model output."),
    "top_tools.png": ("Most-used tools", "The most frequent tools for each source."),
}


def assign_source_colors(sources) -> dict:
    """Stable color per source, assigned alphabetically so every chart in a
    report uses the same mapping regardless of which sources are present."""
    return {source: PALETTE[i % len(PALETTE)] for i, source in enumerate(sorted(sources))}


def plot_ecdf(frame: pd.DataFrame, column: str, title: str, xlabel: str, path: Path, colors: dict, log_x=False):
    """Empirical CDF per source. Complements the histograms: the histogram shows
    distribution shape, the ECDF makes percentiles and cross-source stochastic
    dominance easy to read off directly.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [("all", frame), *list(frame.groupby("source"))]
    for source, group in groups:
        values = group[column].replace([np.inf, -np.inf], np.nan).dropna()
        if log_x:
            values = values[values.gt(0)]
        values = np.sort(values.to_numpy())
        if len(values):
            style = (
                {"color": "black", "linestyle": "--", "linewidth": 1.5}
                if source == "all"
                else {"color": colors.get(source)}
            )
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=source, **style)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of observations ≤ x")
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_hist(ax, frame, column, bins, colors, clip_upper=None, log_x=False):
    """Overlay a per-source histogram normalized to each source's own fraction.

    Sources differ hugely in sample size (e.g. codex has ~10x claude's runs), so
    raw counts bury the smaller source. Normalizing each source to a fraction of
    its own observations makes the shapes directly comparable. A single value
    is shown as a rug marker because normalizing it would create a misleading
    full-height bar.
    """
    for source in sorted(frame["source"].dropna().unique()):
        values = frame.loc[frame["source"].eq(source), column]
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if log_x:
            values = values[values.gt(0)]
        if clip_upper is not None:
            values = values.clip(upper=clip_upper)
        if values.empty:
            continue
        if len(values) == 1:
            ax.plot(
                values,
                [0.02],
                marker="|",
                markersize=10,
                linestyle="none",
                alpha=0.8,
                label=source,
                color=colors.get(source),
                transform=ax.get_xaxis_transform(),
            )
            continue
        weights = np.ones(len(values)) / len(values)
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            alpha=0.55,
            label=source,
            color=colors.get(source),
        )


def plot_count_hist(
    frame: pd.DataFrame, column: str, title: str, xlabel: str, path: Path, colors: dict
):
    """Histogram for small-integer count distributions, grouped by source.

    A CDF hides the shape of these (mode, skew, spikes at 0/1); a frequency
    histogram over integer bins makes it legible.
    """
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    upper = int(np.ceil(values.quantile(0.99))) if len(values) > 1 else int(values.max())
    upper = max(upper, 1)
    bins = np.arange(0, upper + 2) - 0.5  # one integer per bin, centered on the tick

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_hist(ax, frame, column, bins, colors, clip_upper=upper)
    ax.set_title(title)
    ax.set_xlabel(f"{xlabel} (≥{upper} grouped into the last bin)")
    ax.set_ylabel("Fraction of that source")
    if upper <= 30:
        ax.set_xticks(np.arange(0, upper + 1))
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_log_hist(
    frame: pd.DataFrame, column: str, title: str, xlabel: str, path: Path, colors: dict
):
    """Histogram for a positive, heavy-tailed continuous metric (e.g. duration).

    Log-spaced bins on a log x-axis so the bulk and the long tail are both
    visible; normalized per source for cross-source comparison.
    """
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values.gt(0)]
    if values.empty:
        return
    lower = values.min()
    upper = values.max()
    if lower == upper:
        lower /= np.sqrt(10)
        upper *= np.sqrt(10)
    bins = np.logspace(np.log10(lower), np.log10(upper), 40)

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_hist(ax, frame, column, bins, colors, log_x=True)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of that source")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cache_ratios(runs: pd.DataFrame, threads: pd.DataFrame, path: Path, colors: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (frame, level) in zip(axes, ((runs, "Run"), (threads, "Thread")), strict=True):
        _draw_hist(ax, frame, "cache_read_ratio", np.linspace(0, 1, 31), colors)
        ax.set(
            title=f"{level} cache-read ratio",
            xlabel="Cached / served input",
            ylabel="Fraction of that source",
        )
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_load_vs_duration(runs: pd.DataFrame, path: Path, colors: dict):
    """Scatter served input tokens and tool calls against run duration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(7)
    if len(runs):
        sample_index = rng.choice(len(runs), size=min(len(runs), 5000), replace=False)
        sample = runs.iloc[sample_index]
    else:
        sample = runs
    panels = (
        ("served_input_tokens", "Served input tokens", "Input load vs. end-to-end duration"),
        ("tool_calls", "Tool calls", "Tool calls vs. end-to-end duration"),
    )
    for ax, (column, xlabel, title) in zip(axes, panels, strict=True):
        for source, group in sample.groupby("source"):
            valid = group[group[column].gt(0) & group["duration_seconds"].gt(0)]
            ax.scatter(
                valid[column],
                valid["duration_seconds"],
                s=12,
                alpha=0.35,
                label=source,
                color=colors.get(source),
            )
        has_data = (sample[column].gt(0) & sample["duration_seconds"].gt(0)).any()
        ax.set(
            title=title,
            xlabel=f"{xlabel} (log)" if has_data else xlabel,
            ylabel="Duration seconds (log)" if has_data else "Duration seconds",
        )
        if has_data:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tokens_per_run(runs: pd.DataFrame, path: Path, colors: dict):
    """Box plots of input and output tokens per run, grouped by source."""
    token_plot = runs.melt(
        id_vars="source",
        value_vars=["served_input_tokens", "output_tokens"],
        var_name="token_type",
        value_name="tokens",
    )
    token_plot = token_plot[token_plot["tokens"].gt(0)].copy()
    labels = []
    values = []
    for (source, token_type), group in token_plot.groupby(["source", "token_type"], sort=True):
        labels.append(f"{source}\n{token_type.replace('_tokens', '').replace('_', ' ')}")
        values.append(group["tokens"].to_numpy())
    if not values:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    boxes = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for box, label in zip(boxes["boxes"], labels, strict=True):
        box.set_facecolor(colors.get(label.split("\n", 1)[0], "white"))
        box.set_alpha(0.55)
    ax.set_yscale("log")
    ax.set(title="Tokens per run", ylabel="Tokens (log scale)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reasoning_tokens(runs: pd.DataFrame, model_calls: pd.DataFrame, path: Path):
    """Reasoning volume against total output, and reasoning share distributions."""
    reasoning_calls = model_calls[
        model_calls["reasoning_tokens_available"] & model_calls["output_tokens"].gt(0)
    ].copy()
    if reasoning_calls.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    positive = reasoning_calls[reasoning_calls["reasoning_output_tokens"].gt(0)]
    if not positive.empty:
        density = axes[0].hexbin(
            positive["output_tokens"],
            positive["reasoning_output_tokens"],
            xscale="log",
            yscale="log",
            gridsize=40,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        lower = min(positive["output_tokens"].min(), positive["reasoning_output_tokens"].min())
        upper = max(positive["output_tokens"].max(), positive["reasoning_output_tokens"].max())
        axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        fig.colorbar(density, ax=axes[0], label="Model-call count (log color scale)")
    axes[0].set(
        title="Reasoning tokens vs. total output per model call",
        xlabel="Total output tokens (log)",
        ylabel="Reasoning output tokens (log)",
    )
    axes[0].grid(alpha=0.2)

    axes[1].hist(
        reasoning_calls["reasoning_share_of_output"].dropna(),
        bins=np.linspace(0, 1, 31),
        alpha=0.6,
        density=True,
        label="model calls",
    )
    reasoning_runs = runs[
        runs["reasoning_tokens_available"] & runs["reasoning_share_of_output"].notna()
    ]
    axes[1].hist(
        reasoning_runs["reasoning_share_of_output"],
        bins=np.linspace(0, 1, 31),
        alpha=0.6,
        density=True,
        label="runs",
    )
    axes[1].set(
        title="Reasoning share of total output",
        xlabel="Reasoning tokens / total output tokens",
        ylabel="Density",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_top_tools(tool_calls: pd.DataFrame, path: Path, colors: dict):
    source_names = sorted(tool_calls["source"].dropna().unique())
    if not source_names:
        return
    fig, axes = plt.subplots(1, len(source_names), figsize=(7 * len(source_names), 6), squeeze=False)
    for ax, source in zip(axes[0], source_names):
        top_tools = tool_calls[tool_calls["source"].eq(source)]["tool_name"].value_counts().head(12)
        top_tools.sort_values().plot.barh(ax=ax, color=colors.get(source))
        ax.set(title=f"Most-used tools: {source}", xlabel="Calls", ylabel="Tool")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_plots(
    runs: pd.DataFrame,
    threads: pd.DataFrame,
    model_calls: pd.DataFrame,
    tool_calls: pd.DataFrame,
    response_gaps: pd.DataFrame,
    output_dir: Path,
) -> None:
    sources = set()
    for frame in (runs, threads, model_calls, tool_calls, response_gaps):
        if "source" in frame.columns and len(frame):
            sources.update(frame["source"].dropna().unique())
    colors = assign_source_colors(sources)

    plot_log_hist(
        runs,
        "duration_seconds",
        "Agent run duration",
        "Duration (seconds, log scale)",
        output_dir / "run_duration_hist.png",
        colors,
    )
    plot_ecdf(
        runs,
        "duration_seconds",
        "Agent run duration",
        "Duration (seconds, log scale)",
        output_dir / "run_duration_ecdf.png",
        colors,
        log_x=True,
    )
    plot_log_hist(
        response_gaps,
        "gap_seconds",
        "Idle time from last model token to next user message",
        "Seconds (log scale)",
        output_dir / "response_gap_hist.png",
        colors,
    )
    plot_ecdf(
        response_gaps,
        "gap_seconds",
        "Idle time from last model token to next user message",
        "Seconds (log scale)",
        output_dir / "response_gap_ecdf.png",
        colors,
        log_x=True,
    )
    plot_count_hist(
        runs,
        "tool_calls",
        "Tool calls per run",
        "Tool calls",
        output_dir / "tool_calls_per_run_hist.png",
        colors,
    )
    plot_ecdf(
        runs,
        "tool_calls",
        "Tool calls per run",
        "Tool calls",
        output_dir / "tool_calls_per_run_ecdf.png",
        colors,
    )
    plot_count_hist(
        threads,
        "user_messages",
        "User-role prompts per thread",
        "User-role prompts",
        output_dir / "user_messages_per_thread_hist.png",
        colors,
    )
    plot_ecdf(
        threads,
        "user_messages",
        "User-role prompts per thread",
        "User-role prompts",
        output_dir / "user_messages_per_thread_ecdf.png",
        colors,
    )

    plot_cache_ratios(runs, threads, output_dir / "cache_ratios.png", colors)
    plot_load_vs_duration(runs, output_dir / "run_load_vs_duration.png", colors)
    plot_tokens_per_run(runs, output_dir / "tokens_per_run.png", colors)
    plot_reasoning_tokens(runs, model_calls, output_dir / "reasoning_vs_output_tokens.png")
    plot_top_tools(tool_calls, output_dir / "top_tools.png", colors)
