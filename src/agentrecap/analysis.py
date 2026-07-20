import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .plots import make_plots


def safe_ratio(numerator: pd.Series | float, denominator: pd.Series | float):
    if isinstance(denominator, pd.Series):
        return numerator / denominator.replace(0, np.nan)
    return numerator / denominator if denominator else np.nan


def percentile(series: pd.Series, value: float) -> float:
    values = series.dropna()
    return values.quantile(value) if not values.empty else np.nan


def nullable_sum(values: pd.Series):
    """Sum that stays NaN when every observation is missing."""
    return values.sum(min_count=1)


def count_false(values: pd.Series) -> int:
    return values.eq(False).sum()


REQUIRED_COLUMNS = {
    "source",
    "provider",
    "thread_id",
    "stream_id",
    "file_id",
    "event_index",
    "timestamp",
    "event_kind",
    "raw_event_type",
    "is_run_start",
    "run_end_status",
    "is_sidechain",
    "agent_id",
    "tool_call_id",
    "tool_name",
    "tool_success",
    "duration_ms",
    "time_to_first_token_ms",
    "model",
    "message_id",
    "request_id",
    "speed",
    "service_tier",
    "inference_geo",
    "usage_kind",
    "usage_canonical",
    "usage_dedup_reason",
    "usage_source",
    "call_served_input_tokens",
    "call_cached_input_tokens",
    "call_cache_creation_input_tokens",
    "call_cache_creation_5m_input_tokens",
    "call_cache_creation_1h_input_tokens",
    "call_output_tokens",
    "call_reasoning_output_tokens",
    "reported_cost_usd",
}

# The model's own activity within a run; its latest timestamp is when the model
# stopped emitting output.
ASSISTANT_ACTIVITY_KINDS = ["assistant_message", "reasoning", "tool_call"]

# Token totals summed at every aggregation level (run, thread, model).
TOKEN_SUM_COLUMNS = [
    "served_input_tokens",
    "uncached_input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_5m_input_tokens",
    "cache_creation_1h_input_tokens",
    "output_tokens",
]

# Aggregated counts that become 0 (not NaN) when a group has no observations.
COUNT_COLUMNS = [
    "model_calls",
    "reasoning_token_calls",
    "tool_calls",
    "known_tool_outcomes",
    "failed_tool_calls",
]


def aggregate_model_calls(model_calls: pd.DataFrame, keys, **extra_aggregates) -> pd.DataFrame:
    """Per-group model-call counts and token totals shared by runs and threads."""
    return model_calls.groupby(keys, as_index=False).agg(
        model_calls=("served_input_tokens", "size"),
        **{column: (column, "sum") for column in TOKEN_SUM_COLUMNS},
        reasoning_output_tokens=("reasoning_output_tokens", nullable_sum),
        reasoning_token_calls=("reasoning_tokens_available", "sum"),
        **extra_aggregates,
    )


def aggregate_tool_calls(tool_calls: pd.DataFrame, keys) -> pd.DataFrame:
    return tool_calls.groupby(keys, as_index=False).agg(
        tool_calls=("call_success", "size"),
        known_tool_outcomes=("call_success", "count"),
        failed_tool_calls=("call_success", count_false),
    )


def fill_missing_totals(frame: pd.DataFrame, count_columns: list[str]) -> None:
    """Groups without model or tool calls aggregate to NaN; report them as 0."""
    frame[count_columns] = frame[count_columns].fillna(0).astype(int)
    frame[TOKEN_SUM_COLUMNS] = frame[TOKEN_SUM_COLUMNS].fillna(0)


def add_usage_ratios(frame: pd.DataFrame) -> None:
    """Derived cache/reasoning/tool-outcome metrics shared by every level."""
    frame["cache_read_ratio"] = safe_ratio(frame["cached_input_tokens"], frame["served_input_tokens"])
    frame["cache_creation_ratio"] = safe_ratio(
        frame["cache_creation_input_tokens"], frame["served_input_tokens"]
    )
    frame["reasoning_tokens_available"] = frame["reasoning_token_calls"].gt(0)
    frame["non_reasoning_output_tokens"] = (
        frame["output_tokens"] - frame["reasoning_output_tokens"]
    ).clip(lower=0)
    frame["reasoning_share_of_output"] = safe_ratio(
        frame["reasoning_output_tokens"], frame["output_tokens"]
    )
    frame["tool_failure_ratio"] = safe_ratio(
        frame["failed_tool_calls"], frame["known_tool_outcomes"]
    )


def scope_names(frame: pd.DataFrame) -> list[str]:
    return ["all", *sorted(frame["source"].dropna().unique())]


def scope_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    return frame if scope == "all" else frame[frame["source"].eq(scope)]


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    for column in ["is_sidechain", "is_run_start", "usage_canonical"]:
        events[column] = events[column].eq(True)
    events["is_user_prompt"] = events["event_kind"].eq("user_prompt")
    events["is_top_level_user_prompt"] = events["is_user_prompt"] & ~events["is_sidechain"]

    events = events.sort_values(
        ["source", "thread_id", "stream_id", "timestamp", "event_index"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    stream_keys = ["source", "thread_id", "stream_id"]
    events["run_number"] = events.groupby(stream_keys, sort=False)["is_run_start"].cumsum().astype("int64")
    events["run_id"] = pd.NA
    assigned = events["run_number"].gt(0)
    events.loc[assigned, "run_id"] = (
        events.loc[assigned, "source"].astype(str)
        + ":"
        + events.loc[assigned, "thread_id"].astype(str)
        + ":"
        + events.loc[assigned, "stream_id"].astype(str)
        + ":"
        + events.loc[assigned, "run_number"].astype(str)
    )
    return events


def build_model_calls(events: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical model call, as marked by the source adapters."""
    columns = [
        "source",
        "provider",
        "thread_id",
        "stream_id",
        "file_id",
        "run_id",
        "timestamp",
        "model",
        "message_id",
        "request_id",
        "usage_kind",
        "speed",
        "service_tier",
        "inference_geo",
        "usage_source",
        "served_input_tokens",
        "uncached_input_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "cache_creation_5m_input_tokens",
        "cache_creation_1h_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "reasoning_tokens_available",
        "non_reasoning_output_tokens",
        "reasoning_share_of_output",
        "reported_cost_usd",
    ]
    calls = events[events["usage_canonical"]].copy()
    # The raw per-event usage observations stay in the export for auditing;
    # the normalized call_* columns are the per-call accounting.
    calls = calls.drop(
        columns=[
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "cache_creation_5m_input_tokens",
            "cache_creation_1h_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        ]
    )
    calls = calls.rename(
        columns={
            "call_served_input_tokens": "served_input_tokens",
            "call_cached_input_tokens": "cached_input_tokens",
            "call_cache_creation_input_tokens": "cache_creation_input_tokens",
            "call_cache_creation_5m_input_tokens": "cache_creation_5m_input_tokens",
            "call_cache_creation_1h_input_tokens": "cache_creation_1h_input_tokens",
            "call_output_tokens": "output_tokens",
            "call_reasoning_output_tokens": "reasoning_output_tokens",
        }
    )
    calls["uncached_input_tokens"] = (
        calls["served_input_tokens"]
        - calls["cached_input_tokens"]
        - calls["cache_creation_input_tokens"]
    ).clip(lower=0)
    calls["reasoning_tokens_available"] = calls["reasoning_output_tokens"].notna()
    calls["non_reasoning_output_tokens"] = (
        calls["output_tokens"] - calls["reasoning_output_tokens"]
    ).clip(lower=0)
    calls["reasoning_share_of_output"] = safe_ratio(
        calls["reasoning_output_tokens"], calls["output_tokens"]
    )
    model_calls = calls[columns].copy()
    model_calls["cache_read_ratio"] = safe_ratio(
        model_calls["cached_input_tokens"], model_calls["served_input_tokens"]
    )
    model_calls["cache_creation_ratio"] = safe_ratio(
        model_calls["cache_creation_input_tokens"], model_calls["served_input_tokens"]
    )
    model_calls["unclassified_cache_creation_input_tokens"] = (
        model_calls["cache_creation_input_tokens"]
        - model_calls["cache_creation_5m_input_tokens"]
        - model_calls["cache_creation_1h_input_tokens"]
    ).clip(lower=0)
    return model_calls.sort_values("timestamp", kind="stable").reset_index(drop=True)


def build_tool_calls(events: pd.DataFrame) -> pd.DataFrame:
    requests = events[events["event_kind"].eq("tool_call")].copy()

    with_id = requests[requests["tool_call_id"].notna()].drop_duplicates(
        ["source", "thread_id", "stream_id", "run_id", "tool_call_id"], keep="first"
    )
    without_id = requests[requests["tool_call_id"].isna()]
    requests = pd.concat([with_id, without_id], ignore_index=True)

    # Any event carrying an outcome for a call id counts (tool results and
    # runtime completion echoes alike); the latest observation wins.
    outcomes = events[events["tool_call_id"].notna() & events["tool_success"].notna()].copy()
    if not outcomes.empty:
        outcomes = (
            outcomes.sort_values("timestamp", kind="stable")
            .groupby(["source", "thread_id", "stream_id", "tool_call_id"], as_index=False)
            .tail(1)[["source", "thread_id", "stream_id", "tool_call_id", "tool_success"]]
            .rename(columns={"tool_success": "call_success"})
        )
        requests = requests.merge(
            outcomes,
            on=["source", "thread_id", "stream_id", "tool_call_id"],
            how="left",
        )
    else:
        requests["call_success"] = np.nan

    return requests[
        [
            "source",
            "thread_id",
            "stream_id",
            "file_id",
            "run_id",
            "timestamp",
            "tool_call_id",
            "tool_name",
            "call_success",
        ]
    ].sort_values("timestamp", kind="stable").reset_index(drop=True)


def most_common_non_null(series: pd.Series):
    values = series.dropna()
    return values.mode().iloc[0] if not values.empty else None


def combined_speed_status(series: pd.Series) -> str:
    values = set(series.fillna("unknown"))
    return next(iter(values)) if len(values) == 1 else "mixed"


def build_runs(events: pd.DataFrame, model_calls: pd.DataFrame, tool_calls: pd.DataFrame) -> pd.DataFrame:
    assigned = events[events["run_id"].notna()].copy()
    run_rows = []
    for run_id, group in assigned.groupby("run_id", sort=False):
        source = group["source"].iloc[0]
        prompt_rows = group[group["is_run_start"]]
        start = prompt_rows["timestamp"].min()

        terminal = group[group["run_end_status"].notna()]
        assistant = group[group["event_kind"].isin(ASSISTANT_ACTIVITY_KINDS)]

        if not terminal.empty:
            end = terminal["timestamp"].max()
            duration_values = terminal["duration_ms"].dropna()
            duration_ms = duration_values.iloc[-1] if not duration_values.empty else np.nan
            duration_source = "reported_duration_ms" if not duration_values.empty else "event_timestamps"
            terminal_statuses = set(terminal["run_end_status"])
            if "aborted" in terminal_statuses:
                status = "aborted"
            elif "error" in terminal_statuses:
                status = "error"
            else:
                status = "completed"
        else:
            # No explicit run_end event (e.g. Claude): infer the ending from the
            # final assistant message.
            text_end = group[group["event_kind"].eq("assistant_message")]
            end = text_end["timestamp"].max() if not text_end.empty else assistant["timestamp"].max()
            if pd.isna(end):
                end = group["timestamp"].max()
            duration_ms = np.nan
            duration_source = "event_timestamps"
            status = "inferred_complete" if not text_end.empty else "incomplete"

        if pd.isna(duration_ms) and pd.notna(start) and pd.notna(end):
            duration_ms = (end - start).total_seconds() * 1000

        # Timestamp of the model's last emitted token. A terminal run_end event
        # lands after the final assistant activity; last_output_time is that
        # final activity instead.
        last_output_time = assistant["timestamp"].max()
        if pd.isna(last_output_time):
            last_output_time = end

        ttft = group["time_to_first_token_ms"].dropna()
        run_rows.append(
            {
                "run_id": run_id,
                "source": source,
                "thread_id": group["thread_id"].iloc[0],
                "stream_id": group["stream_id"].iloc[0],
                "file_ids": ",".join(
                    sorted(group["file_id"].dropna().astype(str).unique())
                ),
                "run_number": int(group["run_number"].iloc[0]),
                "is_sidechain": bool(prompt_rows["is_sidechain"].any()),
                "start_time": start,
                "end_time": end,
                "last_output_time": last_output_time,
                "duration_seconds": duration_ms / 1000 if pd.notna(duration_ms) else np.nan,
                "duration_source": duration_source,
                "time_to_first_token_seconds": ttft.iloc[-1] / 1000 if not ttft.empty else np.nan,
                "status": status,
            }
        )

    runs = pd.DataFrame(run_rows)

    call_totals = aggregate_model_calls(
        model_calls[model_calls["run_id"].notna()],
        "run_id",
        model=("model", most_common_non_null),
    )
    tool_totals = aggregate_tool_calls(tool_calls[tool_calls["run_id"].notna()], "run_id")
    runs = runs.merge(call_totals, on="run_id", how="left").merge(tool_totals, on="run_id", how="left")
    fill_missing_totals(runs, COUNT_COLUMNS)

    # Imported/resumed histories can contain an entire prior run whose events
    # were all stamped within a millisecond. Preserve that raw observation for
    # auditing, but do not treat it as a real duration measurement.
    runs["raw_duration_seconds"] = runs["duration_seconds"]
    has_activity = runs[["model_calls", "tool_calls", "output_tokens"]].gt(0).any(axis=1)
    collapsed = (
        runs["duration_source"].eq("event_timestamps")
        & runs["duration_seconds"].le(0.1)
        & has_activity
    )
    runs["duration_quality"] = np.where(collapsed, "collapsed_timestamps", "usable")
    runs.loc[collapsed, "duration_seconds"] = np.nan
    add_usage_ratios(runs)
    runs["end_to_end_output_tokens_per_second"] = safe_ratio(
        runs["output_tokens"], runs["duration_seconds"]
    )
    return runs.sort_values("start_time", kind="stable").reset_index(drop=True)


def build_response_gaps(runs: pd.DataFrame) -> pd.DataFrame:
    """Idle time between the model's last output token and the next user message.

    Within a thread stream each run ends when the model emits its last token
    (last_output_time) and the next run begins when the user sends their next
    message (start_time); the gap between them is the user's think/away time
    between turns. Sidechain (subagent) runs are not user turns, so they are
    excluded before pairing consecutive runs.
    """
    ordered = runs[~runs["is_sidechain"]].sort_values(
        ["source", "thread_id", "stream_id", "run_number"], kind="stable"
    )
    grouped = ordered.groupby(["source", "thread_id", "stream_id"], sort=False)
    next_start = grouped["start_time"].shift(-1)
    next_run_id = grouped["run_id"].shift(-1)
    gaps = pd.DataFrame(
        {
            "source": ordered["source"].to_numpy(),
            "thread_id": ordered["thread_id"].to_numpy(),
            "stream_id": ordered["stream_id"].to_numpy(),
            "from_run_id": ordered["run_id"].to_numpy(),
            "to_run_id": next_run_id.to_numpy(),
            "last_output_time": ordered["last_output_time"].to_numpy(),
            "next_prompt_time": next_start.to_numpy(),
            "gap_seconds": (next_start - ordered["last_output_time"]).dt.total_seconds().to_numpy(),
        }
    )
    return gaps[gaps["gap_seconds"].notna()].reset_index(drop=True)


def build_threads(
    events: pd.DataFrame,
    runs: pd.DataFrame,
    model_calls: pd.DataFrame,
    tool_calls: pd.DataFrame,
) -> pd.DataFrame:
    base = events.groupby(["source", "thread_id"], as_index=False).agg(
        thread_start=("timestamp", "min"),
        thread_end=("timestamp", "max"),
        events=("thread_id", "size"),
        is_sidechain=("is_sidechain", "max"),
        user_messages=("is_user_prompt", "sum"),
        top_level_user_messages=("is_top_level_user_prompt", "sum"),
    )
    base["thread_span_hours"] = (base["thread_end"] - base["thread_start"]).dt.total_seconds() / 3600

    run_totals = runs.groupby(["source", "thread_id"], as_index=False).agg(
        runs=("run_id", "size"),
        active_duration_seconds=("duration_seconds", "sum"),
        median_run_duration_seconds=("duration_seconds", "median"),
    )
    call_totals = aggregate_model_calls(
        model_calls,
        ["source", "thread_id"],
        speed_status=("speed", combined_speed_status),
    )
    tool_totals = aggregate_tool_calls(tool_calls, ["source", "thread_id"])

    threads = base.merge(run_totals, on=["source", "thread_id"], how="left")
    threads = threads.merge(call_totals, on=["source", "thread_id"], how="left")
    threads = threads.merge(tool_totals, on=["source", "thread_id"], how="left")
    fill_missing_totals(threads, ["user_messages", "runs", *COUNT_COLUMNS])
    threads["speed_status"] = threads["speed_status"].fillna("unknown")
    add_usage_ratios(threads)
    return threads.sort_values("thread_start", kind="stable").reset_index(drop=True)


def summarize(events: pd.DataFrame, runs: pd.DataFrame, threads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in scope_names(events):
        scope_events = scope_rows(events, scope)
        scope_runs = scope_rows(runs, scope)
        scope_threads = scope_rows(threads, scope)
        served = scope_runs["served_input_tokens"].sum()
        known_outcomes = scope_runs["known_tool_outcomes"].sum()
        reasoning_runs = scope_runs[scope_runs["reasoning_tokens_available"]]
        reasoning_threads = scope_threads[scope_threads["reasoning_tokens_available"]]
        rows.append(
            {
                "scope": scope,
                "events": len(scope_events),
                "threads": len(scope_threads),
                "runs": len(scope_runs),
                "top_level_runs": (~scope_runs["is_sidechain"]).sum(),
                "sidechain_runs": scope_runs["is_sidechain"].sum(),
                "avg_duration_seconds_per_run": scope_runs["duration_seconds"].mean(),
                "median_duration_seconds_per_run": scope_runs["duration_seconds"].median(),
                "p90_duration_seconds_per_run": percentile(scope_runs["duration_seconds"], 0.90),
                "p95_duration_seconds_per_run": percentile(scope_runs["duration_seconds"], 0.95),
                "p99_duration_seconds_per_run": percentile(scope_runs["duration_seconds"], 0.99),
                "avg_tool_calls_per_run": scope_runs["tool_calls"].mean(),
                "median_tool_calls_per_run": scope_runs["tool_calls"].median(),
                "p95_tool_calls_per_run": percentile(scope_runs["tool_calls"], 0.95),
                "avg_user_messages_per_thread": scope_threads["user_messages"].mean(),
                "avg_top_level_user_messages_per_thread": scope_threads["top_level_user_messages"].mean(),
                "avg_model_calls_per_run": scope_runs["model_calls"].mean(),
                "avg_served_input_tokens_per_run": scope_runs["served_input_tokens"].mean(),
                "avg_output_tokens_per_run": scope_runs["output_tokens"].mean(),
                "avg_served_input_tokens_per_thread": scope_threads["served_input_tokens"].mean(),
                "avg_output_tokens_per_thread": scope_threads["output_tokens"].mean(),
                "runs_with_reasoning_token_data": len(reasoning_runs),
                "avg_reasoning_output_tokens_per_run": reasoning_runs[
                    "reasoning_output_tokens"
                ].mean(),
                "avg_reasoning_output_tokens_per_thread": reasoning_threads[
                    "reasoning_output_tokens"
                ].mean(),
                "aggregate_reasoning_share_of_output": safe_ratio(
                    reasoning_runs["reasoning_output_tokens"].sum(),
                    reasoning_runs["output_tokens"].sum(),
                ),
                "aggregate_cache_read_ratio": safe_ratio(scope_runs["cached_input_tokens"].sum(), served),
                "aggregate_cache_creation_ratio": safe_ratio(
                    scope_runs["cache_creation_input_tokens"].sum(), served
                ),
                "avg_thread_cache_read_ratio": scope_threads["cache_read_ratio"].mean(),
                "avg_time_to_first_token_seconds": scope_runs["time_to_first_token_seconds"].mean(),
                "p95_time_to_first_token_seconds": percentile(
                    scope_runs["time_to_first_token_seconds"], 0.95
                ),
                "aborted_run_ratio": scope_runs["status"].eq("aborted").mean(),
                "known_tool_failure_ratio": safe_ratio(
                    scope_runs["failed_tool_calls"].sum(), known_outcomes
                ),
                "aggregate_end_to_end_output_tokens_per_second": safe_ratio(
                    scope_runs.loc[scope_runs["duration_seconds"].notna(), "output_tokens"].sum(),
                    scope_runs["duration_seconds"].sum(),
                ),
            }
        )
    return pd.DataFrame(rows)


def distribution_summary(
    runs: pd.DataFrame, threads: pd.DataFrame, response_gaps: pd.DataFrame
) -> pd.DataFrame:
    datasets = {
        "response_gap": (response_gaps, ["gap_seconds"]),
        "run": (
            runs,
            [
                "duration_seconds",
                "tool_calls",
                "model_calls",
                "served_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "non_reasoning_output_tokens",
                "reasoning_share_of_output",
                "cache_read_ratio",
                "cache_creation_ratio",
                "time_to_first_token_seconds",
            ],
        ),
        "thread": (
            threads,
            [
                "user_messages",
                "active_duration_seconds",
                "tool_calls",
                "model_calls",
                "served_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "non_reasoning_output_tokens",
                "reasoning_share_of_output",
                "cache_read_ratio",
                "cache_creation_ratio",
            ],
        ),
    }
    rows = []
    for level, (frame, metrics) in datasets.items():
        for scope in scope_names(frame):
            scoped = scope_rows(frame, scope)
            for metric in metrics:
                values = scoped[metric].replace([np.inf, -np.inf], np.nan).dropna()
                rows.append(
                    {
                        "level": level,
                        "scope": scope,
                        "metric": metric,
                        "count": len(values),
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "p50": values.median(),
                        "p90": percentile(values, 0.90),
                        "p95": percentile(values, 0.95),
                        "p99": percentile(values, 0.99),
                        "max": values.max(),
                    }
                )
    return pd.DataFrame(rows)


def analyze_threads(input_path: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(input_path)
    model_calls = build_model_calls(events)
    tool_calls = build_tool_calls(events)
    runs = build_runs(events, model_calls, tool_calls)
    threads = build_threads(events, runs, model_calls, tool_calls)
    response_gaps = build_response_gaps(runs)
    summary = summarize(events, runs, threads)
    distributions = distribution_summary(runs, threads, response_gaps)

    summary.to_csv(output_dir / "summary.csv", index=False)
    distributions.to_csv(output_dir / "distribution_summary.csv", index=False)
    runs.to_csv(output_dir / "runs.csv", index=False)
    threads.to_csv(output_dir / "thread_summary.csv", index=False)
    response_gaps.to_csv(output_dir / "response_gaps.csv", index=False)
    model_calls.to_csv(output_dir / "model_calls.csv", index=False)
    tool_calls.to_csv(output_dir / "tool_calls.csv", index=False)

    model_usage = model_calls.copy()
    model_usage["model"] = model_usage["model"].fillna("unknown")
    model_usage = model_usage.groupby(["source", "model"], as_index=False).agg(
        calls=("model", "size"),
        served_input_tokens=("served_input_tokens", "sum"),
        uncached_input_tokens=("uncached_input_tokens", "sum"),
        cached_input_tokens=("cached_input_tokens", "sum"),
        cache_creation_input_tokens=("cache_creation_input_tokens", "sum"),
        cache_creation_5m_input_tokens=("cache_creation_5m_input_tokens", "sum"),
        cache_creation_1h_input_tokens=("cache_creation_1h_input_tokens", "sum"),
        unclassified_cache_creation_input_tokens=(
            "unclassified_cache_creation_input_tokens",
            "sum",
        ),
        output_tokens=("output_tokens", "sum"),
        reasoning_output_tokens=("reasoning_output_tokens", nullable_sum),
        reasoning_token_calls=("reasoning_tokens_available", "sum"),
        avg_served_input_tokens=("served_input_tokens", "mean"),
        p50_served_input_tokens=("served_input_tokens", "median"),
        p95_served_input_tokens=("served_input_tokens", lambda values: percentile(values, 0.95)),
        avg_output_tokens=("output_tokens", "mean"),
        avg_reasoning_output_tokens=("reasoning_output_tokens", "mean"),
        p95_reasoning_output_tokens=("reasoning_output_tokens", lambda values: percentile(values, 0.95)),
    )
    model_usage["cache_read_ratio"] = safe_ratio(
        model_usage["cached_input_tokens"], model_usage["served_input_tokens"]
    )
    model_usage["cache_creation_ratio"] = safe_ratio(
        model_usage["cache_creation_input_tokens"], model_usage["served_input_tokens"]
    )
    model_usage["reasoning_tokens_available"] = model_usage["reasoning_token_calls"].gt(0)
    model_usage["reasoning_share_of_output"] = safe_ratio(
        model_usage["reasoning_output_tokens"], model_usage["output_tokens"]
    )
    model_usage.sort_values("calls", ascending=False).to_csv(
        output_dir / "model_usage.csv", index=False
    )

    tool_usage = tool_calls.groupby(["source", "tool_name"], as_index=False).agg(
        calls=("tool_name", "size"),
        known_outcomes=("call_success", "count"),
        failures=("call_success", count_false),
    )
    tool_usage["failure_ratio"] = safe_ratio(tool_usage["failures"], tool_usage["known_outcomes"])
    tool_usage.sort_values("calls", ascending=False).to_csv(
        output_dir / "tool_usage.csv", index=False
    )

    data_quality_rows = [
        {"check": "events", "value": len(events)},
        {"check": "invalid_timestamps", "value": events["timestamp"].isna().sum()},
        {
            "check": "threads_without_canonical_user_prompt",
            "value": (threads["user_messages"] == 0).sum(),
        },
        {"check": "model_calls_before_first_prompt", "value": model_calls["run_id"].isna().sum()},
        {"check": "tool_calls_before_first_prompt", "value": tool_calls["run_id"].isna().sum()},
        {
            "check": "runs_with_collapsed_timestamps",
            "value": runs["duration_quality"].eq("collapsed_timestamps").sum(),
        },
        {"check": "runs_without_usable_duration", "value": runs["duration_seconds"].isna().sum()},
        {
            "check": "duplicate_thread_event_indexes",
            "value": events.duplicated(["source", "thread_id", "stream_id", "event_index"]).sum(),
        },
        {
            "check": "non_canonical_usage_rows",
            "value": events["usage_dedup_reason"].notna().sum(),
        },
    ]
    data_quality = pd.DataFrame(data_quality_rows)
    data_quality.to_csv(output_dir / "data_quality.csv", index=False)

    make_plots(runs, threads, model_calls, tool_calls, response_gaps, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coding-agent event threads.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sanitized") / "threads.csv",
        help="Input event CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for CSV tables and PNG plots",
    )
    args = parser.parse_args()

    analyze_threads(args.input, args.output_dir)
    print(f"Wrote analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
