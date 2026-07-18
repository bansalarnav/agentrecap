"""Chart generation for agent usage reports."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_COLORS = {"codex": "tab:blue", "claude": "tab:orange"}


def plot_ecdf(frame: pd.DataFrame, column: str, title: str, xlabel: str, path: Path, log_x=False):
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
                else {"color": SOURCE_COLORS.get(source)}
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


def _draw_hist(ax, frame, column, bins, clip_upper=None, log_x=False):
    """Overlay a per-source histogram normalized to each source's own fraction.

    Sources differ hugely in sample size (e.g. codex has ~10x claude's runs), so
    raw counts bury the smaller source. Normalizing each source to a fraction of
    its own observations makes the shapes directly comparable.
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
        weights = np.ones(len(values)) / len(values)
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            alpha=0.55,
            label=source,
            color=SOURCE_COLORS.get(source),
        )


def plot_count_hist(
    frame: pd.DataFrame, column: str, title: str, xlabel: str, path: Path
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
    _draw_hist(ax, frame, column, bins, clip_upper=upper)
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
    frame: pd.DataFrame, column: str, title: str, xlabel: str, path: Path
):
    """Histogram for a positive, heavy-tailed continuous metric (e.g. duration).

    Log-spaced bins on a log x-axis so the bulk and the long tail are both
    visible; normalized per source for cross-source comparison.
    """
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values.gt(0)]
    if values.empty:
        return
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_hist(ax, frame, column, bins, log_x=True)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of that source")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
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
    plot_log_hist(
        runs,
        "duration_seconds",
        "Agent run duration",
        "Duration (seconds, log scale)",
        output_dir / "run_duration_hist.png",
    )
    plot_ecdf(
        runs,
        "duration_seconds",
        "Agent run duration",
        "Duration (seconds, log scale)",
        output_dir / "run_duration_ecdf.png",
        log_x=True,
    )
    plot_log_hist(
        response_gaps,
        "gap_seconds",
        "Idle time from last model token to next user message",
        "Seconds (log scale)",
        output_dir / "response_gap_hist.png",
    )
    plot_ecdf(
        response_gaps,
        "gap_seconds",
        "Idle time from last model token to next user message",
        "Seconds (log scale)",
        output_dir / "response_gap_ecdf.png",
        log_x=True,
    )
    plot_count_hist(
        runs,
        "tool_calls",
        "Tool calls per run",
        "Tool calls",
        output_dir / "tool_calls_per_run_hist.png",
    )
    plot_ecdf(
        runs,
        "tool_calls",
        "Tool calls per run",
        "Tool calls",
        output_dir / "tool_calls_per_run_ecdf.png",
    )
    plot_count_hist(
        threads,
        "user_messages",
        "User-role prompts per thread",
        "User-role prompts",
        output_dir / "user_messages_per_thread_hist.png",
    )
    plot_ecdf(
        threads,
        "user_messages",
        "User-role prompts per thread",
        "User-role prompts",
        output_dir / "user_messages_per_thread_ecdf.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _draw_hist(axes[0], runs, "cache_read_ratio", np.linspace(0, 1, 31))
    axes[0].set(
        title="Run cache-read ratio",
        xlabel="Cached / served input",
        ylabel="Fraction of that source",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    _draw_hist(axes[1], threads, "cache_read_ratio", np.linspace(0, 1, 31))
    axes[1].set(
        title="Thread cache-read ratio",
        xlabel="Cached / served input",
        ylabel="Fraction of that source",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cache_ratios.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(7)
    sample_index = rng.choice(len(runs), size=min(len(runs), 5000), replace=False) if len(runs) else []
    sample = runs.iloc[sample_index] if len(runs) else runs
    for source, group in sample.groupby("source"):
        valid = group[group["served_input_tokens"].gt(0) & group["duration_seconds"].gt(0)]
        axes[0].scatter(
            valid["served_input_tokens"],
            valid["duration_seconds"],
            s=12,
            alpha=0.35,
            label=source,
            color=SOURCE_COLORS.get(source),
        )
        valid = group[group["tool_calls"].gt(0) & group["duration_seconds"].gt(0)]
        axes[1].scatter(
            valid["tool_calls"],
            valid["duration_seconds"],
            s=12,
            alpha=0.35,
            label=source,
            color=SOURCE_COLORS.get(source),
        )
    axes[0].set(
        title="Input load vs. end-to-end duration",
        xlabel="Served input tokens",
        ylabel="Duration seconds",
    )
    axes[1].set(
        title="Tool calls vs. end-to-end duration",
        xlabel="Tool calls",
        ylabel="Duration seconds",
    )
    if (sample["served_input_tokens"].gt(0) & sample["duration_seconds"].gt(0)).any():
        axes[0].set_xlabel("Served input tokens (log)")
        axes[0].set_ylabel("Duration seconds (log)")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
    if (sample["tool_calls"].gt(0) & sample["duration_seconds"].gt(0)).any():
        axes[1].set_xlabel("Tool calls (log)")
        axes[1].set_ylabel("Duration seconds (log)")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "run_load_vs_duration.png", dpi=160)
    plt.close(fig)

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
    if values:
        fig, ax = plt.subplots(figsize=(9, 5))
        boxes = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
        for box, label in zip(boxes["boxes"], labels, strict=True):
            box.set_facecolor(SOURCE_COLORS.get(label.split("\n", 1)[0], "white"))
            box.set_alpha(0.55)
        ax.set_yscale("log")
        ax.set(title="Tokens per run", ylabel="Tokens (log scale)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "tokens_per_run.png", dpi=160)
        plt.close(fig)

    reasoning_calls = model_calls[
        model_calls["reasoning_tokens_available"] & model_calls["output_tokens"].gt(0)
    ].copy()
    if not reasoning_calls.empty:
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
        fig.savefig(output_dir / "reasoning_vs_output_tokens.png", dpi=160)
        plt.close(fig)

    source_names = sorted(tool_calls["source"].dropna().unique())
    if source_names:
        fig, axes = plt.subplots(1, len(source_names), figsize=(7 * len(source_names), 6), squeeze=False)
        for ax, source in zip(axes[0], source_names):
            top_tools = tool_calls[tool_calls["source"].eq(source)]["tool_name"].value_counts().head(12)
            top_tools.sort_values().plot.barh(ax=ax, color=SOURCE_COLORS.get(source))
            ax.set(title=f"Most-used tools: {source}", xlabel="Calls", ylabel="Tool")
            ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "top_tools.png", dpi=160)
        plt.close(fig)

    # These were produced by earlier versions but are not useful for a
    # single-user trace. Remove stale copies when reusing an output directory.
    (output_dir / "daily_load.csv").unlink(missing_ok=True)
    (output_dir / "daily_load.png").unlink(missing_ok=True)
