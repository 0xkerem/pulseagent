#!/usr/bin/env python3
"""
PulseAgent — CLI Entry Point
Run the full pipeline from the command line and print a rich summary.

Usage:
    python run.py --product notion
    python run.py --product notion --live --subreddits Notion productivity
    python run.py --product notion --limit 50 --no-fixtures
"""
import asyncio
import sys
import platform
from pathlib import Path

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from loguru import logger

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from graph.pipeline import run_pipeline
from models import PipelineState, Priority

console = Console()

PRIORITY_COLORS = {
    Priority.P0: "bold red",
    Priority.P1: "bold orange1",
    Priority.P2: "bold yellow",
    Priority.P3: "dim",
}


def print_summary(state: PipelineState):
    from collections import Counter

    console.print(Panel(
        f"[bold cyan]PulseAgent[/] — [white]{state.product_name}[/]\n"
        f"Run ID: [dim]{state.run_id}[/]  •  "
        f"Reviews: [bold]{len(state.raw_reviews)}[/]  •  "
        f"Errors: [{'red' if state.errors else 'green'}]{len(state.errors)}[/]",
        title="✅ Pipeline Complete",
        border_style="cyan",
    ))

    if state.classified_reviews:
        cat_dist = Counter(
            r.category.value if hasattr(r.category, "value") else str(r.category)
            for r in state.classified_reviews
        )
        churn = sum(1 for r in state.classified_reviews if r.is_churn_signal)
        console.print(f"\n[bold]Category Distribution:[/]")
        for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
            bar = "█" * count
            console.print(f"  {cat:<25} {bar} [dim]{count}[/]")
        console.print(f"\n  [bold red]🚨 Churn signals: {churn}[/]")

    if state.trend_alerts:
        console.print(f"\n[bold]Trend Alerts ({len(state.trend_alerts)}):[/]")
        for alert in state.trend_alerts:
            icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}[alert.alert_level]
            arrow = "📈" if alert.direction == "rising" else "📉"
            console.print(f"  {icon} {arrow} {alert.theme} [{abs(alert.change_percent):.0f}%]")

    if state.roadmap_items:
        table = Table(title=f"\nRoadmap — {len(state.roadmap_items)} Items", show_lines=True)
        table.add_column("ID", style="dim", width=12)
        table.add_column("Priority", width=6)
        table.add_column("Title", min_width=30)
        table.add_column("Effort", width=8)
        table.add_column("Users", width=8)
        table.add_column("Churn", width=6)

        for item in state.roadmap_items:
            color = PRIORITY_COLORS[item.priority]
            table.add_row(
                item.item_id,
                Text(item.priority.value, style=color),
                item.title,
                item.implementation_effort,
                str(item.affected_users_estimate),
                str(int(item.churn_risk_score)),
            )
        console.print(table)

    if state.draft_responses:
        console.print(
            f"\n[bold green]✉️  {len(state.draft_responses)} draft responses[/] "
            f"generated (awaiting human approval)"
        )

    if state.errors:
        console.print(f"\n[bold red]Errors:[/]")
        for err in state.errors:
            console.print(f"  [red]• {err}[/]")


async def main():
    parser = argparse.ArgumentParser(description="PulseAgent — Product Review Intelligence")
    parser.add_argument("--product", "-p", default="notion", help="Product name")
    parser.add_argument("--limit", "-n", type=int, default=30, help="Max reviews")
    parser.add_argument("--no-fixtures", action="store_true", help="Use live scrapers")
    parser.add_argument("--subreddits", nargs="+", default=[], help="Reddit subreddits")
    parser.add_argument("--app-id", help="App Store app ID")
    parser.add_argument("--fixture-dir", default="./data/fixtures")
    parser.add_argument("--docs-dir", default="./data/docs")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    console.print(f"\n[bold cyan]🔊 PulseAgent[/] starting for [bold]{args.product}[/]...\n")

    with console.status("[cyan]Running pipeline...[/]"):
        state = await run_pipeline(
            product_name=args.product,
            use_fixtures=not args.no_fixtures,
            fixture_dir=args.fixture_dir,
            docs_dir=args.docs_dir if Path(args.docs_dir).exists() else None,
            review_limit=args.limit,
            reddit_subreddits=args.subreddits,
            app_store_app_id=args.app_id,
        )

    print_summary(state)


if __name__ == "__main__":
    # Windows fix: use SelectorEventLoop to avoid SSL cleanup errors on ProactorEventLoop
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())