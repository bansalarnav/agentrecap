"""Generate the combined coding-agent usage report."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .analysis import analyze_threads, combined_speed_status, nullable_sum
from .gather_session_data import convert_sessions
from .plots import CHARTS


MODELS_DEV_URL = "https://models.dev/api.json"
BARE_MODEL_PROVIDERS = ("anthropic", "openai", "google")

# Anthropic bills API traffic pinned to US inference at a 10% premium.
ANTHROPIC_US_GEO_MULTIPLIER = 1.1
# Anthropic's 1h-TTL cache writes bill at 2x the input price. models.dev has
# no TTL-specific cost keys, so this is the one rate the catalog cannot
# supply; every other price comes from the catalog entry.
CACHE_WRITE_1H_MULTIPLIER = 2

# Written by earlier versions; removed when reusing an output directory.
STALE_OUTPUTS = ("daily_load.csv", "daily_load.png")


def format_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def format_table_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "—"
    if column.endswith("_cost_usd"):
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


def _nz(value) -> float:
    """Treat missing numeric fields (None/NaN) as zero."""
    return 0 if value is None or pd.isna(value) else value


def _catalog_cost(catalog: dict, provider_id: str | None, model_id: str) -> tuple[dict | None, dict | None]:
    model = catalog.get(provider_id, {}).get("models", {}).get(model_id)
    cost = dict(model.get("cost") or {}) if model else None
    return model, cost


def _match_model(row: dict, catalog: dict) -> tuple[dict | None, dict | None, str | None, bool]:
    """Find the models.dev entry for a call.

    Falls back to the major providers when the row's own provider does not
    list the model (some tools record bare model ids without a provider).
    Returns (model entry, cost dict, pricing provider, used bare-model fallback).
    """
    provider_id = row.get("provider")
    if pd.isna(provider_id):
        provider_id = None
    model_id = str(row["model"])
    model, cost = _catalog_cost(catalog, provider_id, model_id)
    if cost:
        return model, cost, provider_id, False
    for candidate_provider in BARE_MODEL_PROVIDERS:
        model, cost = _catalog_cost(catalog, candidate_provider, model_id)
        if cost:
            return model, cost, candidate_provider, True
    return None, None, provider_id, False


def _context_tier_cost(row: dict, cost: dict) -> tuple[dict, str | None]:
    """Apply the highest context-window pricing tier the call crossed.

    models.dev encodes long-context pricing two ways: a ``tiers`` list with
    explicit thresholds, and a flat ``context_over_200k`` dict of override
    prices. The list takes precedence when present.
    """
    served_input_tokens = _nz(row.get("served_input_tokens", 0))
    tier_label = None
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
            tier_label = f"context_over_{threshold}"

    flat_tier = cost.get("context_over_200k")
    if not context_tiers and served_input_tokens > 200_000 and isinstance(flat_tier, dict):
        cost.update(flat_tier)
        tier_label = "context_over_200000"
    return cost, tier_label


def _estimated_cost_usd(row: dict, cost: dict) -> float:
    """Price one call's token usage against a resolved cost entry ($ per Mtok)."""
    input_price = cost["input"]
    output_price = cost["output"]
    # Cache rates come straight from the catalog; a provider that lists no
    # cache pricing bills those tokens at the normal input rate. Only the
    # 1h-TTL write surcharge is computed, since the catalog has no TTL keys.
    cache_read_price = cost.get("cache_read", input_price)
    cache_write_price = cost.get("cache_write", input_price)
    cache_write_5m_price = cost.get("cache_write_5m", cache_write_price)
    cache_write_1h_price = cost.get("cache_write_1h", input_price * CACHE_WRITE_1H_MULTIPLIER)
    reasoning_price = cost.get("reasoning", output_price)

    reasoning_tokens = _nz(row.get("reasoning_output_tokens"))
    non_reasoning_tokens = max(row["output_tokens"] - reasoning_tokens, 0)
    uncached_input_tokens = row.get("uncached_input_tokens")
    if uncached_input_tokens is None or pd.isna(uncached_input_tokens):
        uncached_input_tokens = max(
            row["served_input_tokens"]
            - row["cached_input_tokens"]
            - row["cache_creation_input_tokens"],
            0,
        )
    cache_creation_5m_tokens = _nz(row.get("cache_creation_5m_input_tokens", 0))
    cache_creation_1h_tokens = _nz(row.get("cache_creation_1h_input_tokens", 0))
    unclassified_cache_creation_tokens = max(
        _nz(row.get("cache_creation_input_tokens", 0))
        - cache_creation_5m_tokens
        - cache_creation_1h_tokens,
        0,
    )

    # The US-geo surcharge follows the provider that actually served the call
    # (the row's own provider), not the catalog entry used to price it.
    geo_multiplier = (
        ANTHROPIC_US_GEO_MULTIPLIER
        if row.get("provider") == "anthropic" and row.get("inference_geo") == "us"
        else 1
    )
    return geo_multiplier * (
        uncached_input_tokens * input_price
        + row["cached_input_tokens"] * cache_read_price
        + unclassified_cache_creation_tokens * cache_write_price
        + cache_creation_5m_tokens * cache_write_5m_price
        + cache_creation_1h_tokens * cache_write_1h_price
        + non_reasoning_tokens * output_price
        + reasoning_tokens * reasoning_price
    ) / 1_000_000


def _price_model_call(row: dict, catalog: dict) -> None:
    """Attach pricing_provider/status/tier and estimated_cost_usd to one call row."""
    model, cost, pricing_provider, bare_model_fallback = _match_model(row, catalog)
    row["pricing_provider"] = pricing_provider
    if bare_model_fallback:
        row["pricing_status"] = "matched_bare_model_fallback"
    elif cost:
        row["pricing_status"] = "matched"
    else:
        row["pricing_status"] = "unmatched_model"
    row["pricing_tier"] = None
    row["estimated_cost_usd"] = None
    if not cost:
        return

    speed = row.get("speed")
    if pd.isna(speed):
        speed = "unknown"
    if speed == "fast":
        fast_cost = (
            ((model.get("experimental") or {}).get("modes") or {}).get("fast", {}).get("cost")
        )
        if not fast_cost:
            row["pricing_status"] = "fast_price_unavailable"
            return
        cost = dict(fast_cost)
        row["pricing_tier"] = "fast"
    else:
        cost, row["pricing_tier"] = _context_tier_cost(row, cost)
        if speed == "unknown":
            row["pricing_status"] = (
                "matched_bare_model_fallback_speed_unknown"
                if bare_model_fallback
                else "matched_speed_unknown"
            )
            if row["pricing_tier"] is None:
                row["pricing_tier"] = "standard_fallback_speed_unknown"
        elif row["pricing_tier"] is None:
            row["pricing_tier"] = "standard"

    if cost.get("input") is None or cost.get("output") is None:
        row["pricing_status"] = "incomplete_price"
        return
    row["estimated_cost_usd"] = _estimated_cost_usd(row, cost)


def add_model_costs(model_usage: pd.DataFrame, catalog: dict) -> pd.DataFrame:
    rows = model_usage.to_dict("records")
    for row in rows:
        _price_model_call(row, catalog)
    return pd.DataFrame(rows).sort_values("estimated_cost_usd", ascending=False, na_position="last")


def run_pipeline(inputs: dict[str, Path], output_dir: Path) -> None:
    data_dir = output_dir / "data"
    events_path = data_dir / "threads.csv"

    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in (*CHARTS, *STALE_OUTPUTS):
        (data_dir / filename).unlink(missing_ok=True)

    convert_sessions(inputs, events_path)
    print(f"Session data saved to {events_path}")
    print("Generating Report...")
    analyze_threads(events_path, data_dir)


def _price_model_calls(data_dir: Path) -> tuple[pd.DataFrame, str, list[str]]:
    """Price every model call and write the priced call and per-model cost tables.

    model_calls.csv is rewritten in place with the pricing columns added;
    analyze_threads regenerates the unpriced table on every pipeline run.
    """
    pricing_catalog, pricing_source = load_pricing_catalog(data_dir)
    model_calls = pd.read_csv(data_dir / "model_calls.csv")
    model_calls["timestamp"] = pd.to_datetime(model_calls["timestamp"], utc=True, errors="coerce")
    model_calls["model"] = model_calls["model"].fillna("unknown")
    model_calls["calls"] = 1
    priced_calls = add_model_costs(model_calls, pricing_catalog)
    priced_calls.to_csv(data_dir / "model_calls.csv", index=False)

    model_usage = pd.read_csv(data_dir / "model_usage.csv")
    call_costs = priced_calls.groupby(["source", "model"], as_index=False).agg(
        estimated_cost_usd=("estimated_cost_usd", nullable_sum),
        priced_calls=("estimated_cost_usd", "count"),
        speed_status=("speed", combined_speed_status),
        pricing_status=("pricing_status", lambda values: ",".join(sorted(set(values)))),
    )
    model_costs = model_usage.merge(call_costs, on=["source", "model"], how="left")
    model_costs.to_csv(data_dir / "model_costs.csv", index=False)

    unmatched_models = priced_calls.loc[
        priced_calls["estimated_cost_usd"].isna(), "model"
    ].astype(str).drop_duplicates().tolist()
    return priced_calls, pricing_source, unmatched_models


def _build_monthly_costs(
    matched_calls: pd.DataFrame, local_timestamps: pd.Series, data_dir: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Per-month cost totals by source, written to monthly_costs.csv."""
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
    return monthly_costs, cost_sources


def build_report(output_dir: Path, title: str) -> Path:
    data_dir = output_dir / "data"
    transcript_dir = output_dir / "transcripts"
    if transcript_dir.exists():
        shutil.rmtree(transcript_dir)

    priced_calls, pricing_source, unmatched_models = _price_model_calls(data_dir)
    matched_calls = priced_calls[priced_calls["estimated_cost_usd"].notna()].copy()
    estimated_cost = matched_calls["estimated_cost_usd"].sum()

    now = pd.Timestamp.now(tz=datetime.now().astimezone().tzinfo)
    local_timestamps = matched_calls["timestamp"].dt.tz_convert(now.tzinfo)
    last_30_days_cost = matched_calls.loc[
        local_timestamps.ge(now - pd.Timedelta(days=30)), "estimated_cost_usd"
    ].sum()
    month_start = now.normalize().replace(day=1)
    month_to_date_cost = matched_calls.loc[
        local_timestamps.ge(month_start), "estimated_cost_usd"
    ].sum()
    monthly_costs, cost_sources = _build_monthly_costs(matched_calls, local_timestamps, data_dir)

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

    table_specs = [
        (
            "Source comparison",
            "summary.csv",
            [
                "scope",
                "threads",
                "runs",
                "avg_duration_seconds_per_run",
                "avg_tool_calls_per_run",
                "avg_served_input_tokens_per_run",
                "avg_output_tokens_per_run",
                "aggregate_cache_read_ratio",
            ],
            "runs",
        ),
        (
            "Models",
            "model_costs.csv",
            [
                "source",
                "model",
                "speed_status",
                "estimated_cost_usd",
                "calls",
                "served_input_tokens",
                "output_tokens",
                "cache_read_ratio",
                "reasoning_share_of_output",
            ],
            "estimated_cost_usd",
        ),
    ]
    tables = []
    for heading, filename, columns, sort_column in table_specs:
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
