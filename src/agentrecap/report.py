"""Generate the combined Codex and Claude usage report."""

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


def discover_jsonl(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(path.rglob("*.jsonl"))


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
    provider_by_source = {"codex": "openai", "claude": "anthropic"}
    for row in model_usage.to_dict("records"):
        provider_id = provider_by_source.get(row["source"])
        model = catalog.get(provider_id, {}).get("models", {}).get(str(row["model"]))
        cost = model.get("cost") if model else None
        row["pricing_provider"] = provider_id
        row["pricing_status"] = "matched" if cost else "unmatched"
        row["estimated_cost_usd"] = None
        if cost:
            input_price = cost.get("input")
            output_price = cost.get("output")
            cache_read_price = cost.get("cache_read", input_price)
            cache_write_price = cost.get("cache_write", input_price)
            reasoning_price = cost.get("reasoning", output_price)
            reasoning_tokens = row.get("reasoning_output_tokens")
            reasoning_tokens = 0 if pd.isna(reasoning_tokens) else reasoning_tokens
            non_reasoning_tokens = max(row["output_tokens"] - reasoning_tokens, 0)
            uncached_input_tokens = row.get("uncached_input_tokens")
            if uncached_input_tokens is None:
                uncached_input_tokens = max(
                    row["served_input_tokens"]
                    - row["cached_input_tokens"]
                    - row["cache_creation_input_tokens"],
                    0,
                )

            row["estimated_cost_usd"] = (
                uncached_input_tokens * input_price
                + row["cached_input_tokens"] * cache_read_price
                + row["cache_creation_input_tokens"] * cache_write_price
                + non_reasoning_tokens * output_price
                + reasoning_tokens * reasoning_price
            ) / 1_000_000
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["estimated_cost_usd", "calls"], ascending=False, na_position="last"
    )


def run_pipeline(codex_input: Path, claude_input: Path, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    events_path = data_dir / "threads.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in CHARTS:
        (data_dir / filename).unlink(missing_ok=True)
    convert_sessions(
        codex_input,
        claude_input,
        events_path,
    )
    print(f"Session data saved to {events_path}")
    print("Generating Report...")
    analyze_threads(events_path, data_dir)


def build_report(output_dir: Path, title: str) -> Path:
    data_dir = output_dir / "data"
    transcript_dir = output_dir / "transcripts"
    if transcript_dir.exists():
        shutil.rmtree(transcript_dir)

    pricing_catalog, pricing_source = load_pricing_catalog(data_dir)
    model_costs = add_model_costs(pd.read_csv(data_dir / "model_usage.csv"), pricing_catalog)
    model_costs.to_csv(data_dir / "model_costs.csv", index=False)
    matched_costs = model_costs[model_costs["pricing_status"].eq("matched")]
    estimated_cost = matched_costs["estimated_cost_usd"].sum()
    unmatched_models = model_costs.loc[
        model_costs["pricing_status"].eq("unmatched"), "model"
    ].astype(str).tolist()

    model_calls = pd.read_csv(data_dir / "model_calls.csv")
    model_calls["timestamp"] = pd.to_datetime(model_calls["timestamp"], utc=True, errors="coerce")
    model_calls["model"] = model_calls["model"].fillna("unknown")
    model_calls["calls"] = 1
    priced_calls = add_model_costs(model_calls, pricing_catalog)
    matched_calls = priced_calls[priced_calls["pricing_status"].eq("matched")].copy()

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
    monthly_costs = monthly_costs.reindex(columns=["claude", "codex"], fill_value=0)
    monthly_costs["combined"] = monthly_costs.sum(axis=1)
    monthly_costs = monthly_costs.rename(
        columns={
            "claude": "claude_cost_usd",
            "codex": "codex_cost_usd",
            "combined": "combined_cost_usd",
        }
    ).reset_index()
    monthly_costs.to_csv(data_dir / "monthly_costs.csv", index=False)

    summary = pd.read_csv(data_dir / "summary.csv")
    all_summary = summary[summary["scope"].eq("all")].iloc[0]
    cards = [
        ("Threads", all_summary.get("threads")),
        ("Median run", f"{format_value(all_summary.get('median_duration_seconds_per_run'))} s"),
        ("Tool calls / run", format_value(all_summary.get("avg_tool_calls_per_run"))),
        ("Estimated API cost", f"${estimated_cost:,.2f}" if not matched_costs.empty else "—"),
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
        ("Models", "model_costs.csv", ["source", "model", "estimated_cost_usd", "calls", "served_input_tokens", "output_tokens", "cache_read_ratio", "reasoning_share_of_output"], "estimated_cost_usd"),
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
        ["month", "claude_cost_usd", "codex_cost_usd", "combined_cost_usd"],
    )
    monthly_costs_table.update(
        {"heading": "Monthly costs", "pricing_source": None, "unmatched_models": []}
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
