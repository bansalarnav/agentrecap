"""Generate the combined coding-agent usage report."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .analysis import analyze_threads
from .gather_session_data import convert_sessions


MODELS_DEV_URL = "https://models.dev/api.json"

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


def format_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def format_table_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "—"
    if column == "estimated_cost_usd" or column.endswith("_cost_usd"):
        return f"${float(value):,.2f}"
    return format_value(value)


def format_table(frame: pd.DataFrame, columns: list[str] | None = None) -> dict:
    if columns is not None:
        columns = [column for column in columns if column in frame.columns]
        frame = frame[columns]
    return {
        "headers": [column.replace("_", " ") for column in frame.columns],
        "rows": [
            [
                format_table_value(column, value)
                for column, value in zip(frame.columns, values, strict=True)
            ]
            for values in frame.itertuples(index=False, name=None)
        ],
    }


def load_pricing_catalog(data_dir: Path) -> tuple[dict, str]:
    cache_path = data_dir / "models_dev_api.json"
    try:
        request = Request(MODELS_DEV_URL, headers={"User-Agent": "agentrecap/0.1"})
        with urlopen(request, timeout=15) as response:
            catalog = json.loads(response.read().decode("utf-8"))
        cache_path.write_text(json.dumps(catalog), encoding="utf-8", newline="\n")
        return catalog, "current models.dev catalog"
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if cache_path.exists():
            print(f"Warning: could not refresh models.dev pricing ({error}); using cached catalog")
            return json.loads(cache_path.read_text(encoding="utf-8")), "cached models.dev catalog"
        print(f"Warning: could not load models.dev pricing ({error}); costs will be unavailable")
        return {}, "models.dev catalog unavailable"


def add_model_costs(model_usage: pd.DataFrame, catalog: dict) -> pd.DataFrame:
    rows = []
    for row in model_usage.to_dict("records"):
        provider_id = row.get("provider")
        if pd.isna(provider_id):
            provider_id = None
        model = catalog.get(provider_id, {}).get("models", {}).get(str(row["model"]))
        cost = dict(model.get("cost") or {}) if model else None
        row["pricing_provider"] = provider_id
        row["pricing_status"] = "matched" if cost else "unmatched_model"
        row["pricing_tier"] = None
        row["estimated_cost_usd"] = None
        if cost:
            speed = row.get("speed")
            if pd.isna(speed):
                speed = "unknown"
            if speed == "fast":
                fast_cost = (
                    ((model.get("experimental") or {}).get("modes") or {})
                    .get("fast", {})
                    .get("cost")
                )
                if not fast_cost:
                    row["pricing_status"] = "fast_price_unavailable"
                    rows.append(row)
                    continue
                cost = dict(fast_cost)
                row["pricing_tier"] = "fast"
            else:
                served_input_tokens = row.get("served_input_tokens", 0)
                served_input_tokens = (
                    0 if pd.isna(served_input_tokens) else served_input_tokens
                )
                context_tiers = sorted(
                    (
                        tier
                        for tier in cost.get("tiers", [])
                        if (tier.get("tier") or {}).get("type") == "context"
                    ),
                    key=lambda tier: (tier.get("tier") or {}).get("size", 0),
                )
                for tier in context_tiers:
                    threshold = (tier.get("tier") or {}).get("size")
                    if threshold is not None and served_input_tokens > threshold:
                        cost.update({key: value for key, value in tier.items() if key != "tier"})
                        row["pricing_tier"] = f"context_over_{threshold}"

                timestamp = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
                historical_sonnet = str(row["model"]).startswith(
                    ("claude-sonnet-4-20250514", "claude-sonnet-4-5")
                )
                if (
                    provider_id == "anthropic"
                    and not context_tiers
                    and historical_sonnet
                    and served_input_tokens > 200_000
                    and pd.notna(timestamp)
                    and timestamp < pd.Timestamp("2026-04-30", tz="UTC")
                ):
                    cost.update(
                        {
                            "input": 6,
                            "output": 22.5,
                            "cache_read": 0.6,
                            "cache_write": 7.5,
                            "cache_write_1h": 12,
                        }
                    )
                    row["pricing_tier"] = "historical_context_over_200000"

                if speed == "unknown":
                    row["pricing_status"] = "matched_speed_unknown"
                    if row["pricing_tier"] is None:
                        row["pricing_tier"] = "standard_fallback_speed_unknown"
                elif row["pricing_tier"] is None:
                    row["pricing_tier"] = "standard"

            input_price = cost.get("input")
            output_price = cost.get("output")
            if input_price is None or output_price is None:
                row["pricing_status"] = "incomplete_price"
                rows.append(row)
                continue
            cache_read_price = cost.get(
                "cache_read", input_price * 0.1 if provider_id == "anthropic" else input_price
            )
            cache_write_price = cost.get(
                "cache_write", input_price * 1.25 if provider_id == "anthropic" else input_price
            )
            cache_write_5m_price = cost.get("cache_write_5m", cache_write_price)
            cache_write_1h_price = cost.get("cache_write_1h", input_price * 2)
            reasoning_price = cost.get("reasoning", output_price)
            reasoning_tokens = row.get("reasoning_output_tokens")
            reasoning_tokens = 0 if pd.isna(reasoning_tokens) else reasoning_tokens
            non_reasoning_tokens = max(row["output_tokens"] - reasoning_tokens, 0)
            uncached_input_tokens = row.get("uncached_input_tokens")
            if uncached_input_tokens is None or pd.isna(uncached_input_tokens):
                uncached_input_tokens = max(
                    row["served_input_tokens"]
                    - row["cached_input_tokens"]
                    - row["cache_creation_input_tokens"],
                    0,
                )
            cache_creation_tokens = row.get("cache_creation_input_tokens", 0)
            cache_creation_tokens = 0 if pd.isna(cache_creation_tokens) else cache_creation_tokens
            cache_creation_5m_tokens = row.get("cache_creation_5m_input_tokens", 0)
            cache_creation_1h_tokens = row.get("cache_creation_1h_input_tokens", 0)
            cache_creation_5m_tokens = (
                0 if pd.isna(cache_creation_5m_tokens) else cache_creation_5m_tokens
            )
            cache_creation_1h_tokens = (
                0 if pd.isna(cache_creation_1h_tokens) else cache_creation_1h_tokens
            )
            unclassified_cache_creation_tokens = max(
                cache_creation_tokens
                - cache_creation_5m_tokens
                - cache_creation_1h_tokens,
                0,
            )

            geo_multiplier = 1.1 if provider_id == "anthropic" and row.get("inference_geo") == "us" else 1
            row["estimated_cost_usd"] = geo_multiplier * (
                uncached_input_tokens * input_price
                + row["cached_input_tokens"] * cache_read_price
                + unclassified_cache_creation_tokens * cache_write_price
                + cache_creation_5m_tokens * cache_write_5m_price
                + cache_creation_1h_tokens * cache_write_1h_price
                + non_reasoning_tokens * output_price
                + reasoning_tokens * reasoning_price
            ) / 1_000_000
        rows.append(row)
    return pd.DataFrame(rows).sort_values("estimated_cost_usd", ascending=False, na_position="last")


def run_pipeline(inputs: dict[str, Path], output_dir: Path) -> None:
    data_dir = output_dir / "data"
    events_path = data_dir / "threads.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in CHARTS:
        (data_dir / filename).unlink(missing_ok=True)
    convert_sessions(inputs, events_path)
    print(f"Session data saved to {events_path}")
    print("Generating Report...")
    analyze_threads(events_path, data_dir)


def build_report(output_dir: Path, title: str) -> Path:
    data_dir = output_dir / "data"
    transcript_dir = output_dir / "transcripts"
    if transcript_dir.exists():
        shutil.rmtree(transcript_dir)

    pricing_catalog, pricing_source = load_pricing_catalog(data_dir)
    model_calls = pd.read_csv(data_dir / "model_calls.csv")
    model_calls["timestamp"] = pd.to_datetime(model_calls["timestamp"], utc=True, errors="coerce")
    model_calls["model"] = model_calls["model"].fillna("unknown")
    model_calls["calls"] = 1
    priced_calls = add_model_costs(model_calls, pricing_catalog)
    priced_calls.to_csv(data_dir / "model_calls.csv", index=False)
    matched_calls = priced_calls[priced_calls["estimated_cost_usd"].notna()].copy()

    model_usage = pd.read_csv(data_dir / "model_usage.csv")
    call_costs = priced_calls.groupby(["source", "model"], as_index=False).agg(
        estimated_cost_usd=("estimated_cost_usd", lambda values: values.sum(min_count=1)),
        priced_calls=("estimated_cost_usd", "count"),
        speed_status=("speed", lambda values: next(iter(set(values))) if len(set(values)) == 1 else "mixed"),
        pricing_status=("pricing_status", lambda values: ",".join(sorted(set(values)))),
    )
    model_costs = model_usage.merge(call_costs, on=["source", "model"], how="left")
    model_costs.to_csv(data_dir / "model_costs.csv", index=False)
    estimated_cost = matched_calls["estimated_cost_usd"].sum()
    unmatched_models = priced_calls.loc[
        priced_calls["estimated_cost_usd"].isna(), "model"
    ].astype(str).drop_duplicates().tolist()

    now = pd.Timestamp.now(tz=datetime.now().astimezone().tzinfo)
    local_timestamps = matched_calls["timestamp"].dt.tz_convert(now.tzinfo)
    last_30_days_cost = matched_calls.loc[
        local_timestamps.ge(now - pd.Timedelta(days=30)), "estimated_cost_usd"
    ].sum()
    month_start = now.normalize().replace(day=1)
    month_to_date_cost = matched_calls.loc[
        local_timestamps.ge(month_start), "estimated_cost_usd"
    ].sum()

    matched_calls["month"] = local_timestamps.dt.strftime("%Y-%m")
    monthly_costs = matched_calls.groupby(["month", "source"])["estimated_cost_usd"].sum().unstack(
        fill_value=0
    )
    cost_sources = sorted(monthly_costs.columns)
    monthly_costs["combined_cost_usd"] = monthly_costs.sum(axis=1)
    monthly_costs = monthly_costs.rename(
        columns={source: f"{source}_cost_usd" for source in cost_sources}
    ).reset_index()
    monthly_costs.to_csv(data_dir / "monthly_costs.csv", index=False)

    summary = pd.read_csv(data_dir / "summary.csv")
    all_summary = summary[summary["scope"].eq("all")].iloc[0]
    cards = [
        ("Threads", all_summary.get("threads")),
        ("Median run", f"{format_value(all_summary.get('median_duration_seconds_per_run'))} s"),
        ("Tool calls / run", format_value(all_summary.get("avg_tool_calls_per_run"))),
        ("Estimated API cost", f"${estimated_cost:,.2f}" if not matched_calls.empty else "—"),
        ("Cost in last 30 days", f"${last_30_days_cost:,.2f}" if not matched_calls.empty else "—"),
        ("Cost month to date", f"${month_to_date_cost:,.2f}" if not matched_calls.empty else "—"),
    ]
    cards = [{"label": label, "value": format_value(value)} for label, value in cards]

    charts = []
    for filename, (heading, description) in CHARTS.items():
        if (data_dir / filename).exists():
            charts.append(
                {
                    "url": f"data/{quote(filename)}",
                    "heading": heading,
                    "description": description,
                }
            )

    tables = []
    for heading, filename, columns, sort_column in [
        ("Source comparison", "summary.csv", ["scope", "threads", "runs", "avg_duration_seconds_per_run", "avg_tool_calls_per_run", "avg_served_input_tokens_per_run", "avg_output_tokens_per_run", "aggregate_cache_read_ratio"], "runs"),
        ("Models", "model_costs.csv", ["source", "model", "speed_status", "estimated_cost_usd", "calls", "served_input_tokens", "output_tokens", "cache_read_ratio", "reasoning_share_of_output"], "estimated_cost_usd"),
    ]:
        path = data_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            if sort_column in frame.columns:
                frame = frame.sort_values(sort_column, ascending=False, na_position="last")
            table = format_table(frame, columns)
            table.update(
                {
                    "heading": heading,
                    "pricing_source": pricing_source if filename == "model_costs.csv" else None,
                    "unmatched_models": unmatched_models if filename == "model_costs.csv" else [],
                }
            )
            tables.append(table)

    monthly_costs_table = format_table(
        monthly_costs.sort_values("month", ascending=False),
        ["month", *[f"{source}_cost_usd" for source in cost_sources], "combined_cost_usd"],
    )
    monthly_costs_table.update(
        {
            "heading": "Monthly costs",
            "pricing_source": None,
            "unmatched_models": [],
        }
    )
    tables.append(monthly_costs_table)

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    environment = Environment(
        loader=PackageLoader("agentrecap"),
        autoescape=select_autoescape(["html"]),
    )
    document = environment.get_template("report.html").render(
        title=title,
        generated=generated,
        cards=cards,
        charts=charts,
        tables=tables,
    )
    index_path = output_dir / "index.html"
    index_path.write_text(document, encoding="utf-8", newline="\n")
    return index_path
