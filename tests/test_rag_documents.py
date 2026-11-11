"""The P&L ranking document lets the chat answer questions that compare all positions."""

from src.rag.document_loader import load_portfolio_documents
from src.utils.data import load_portfolio


def _ranking(portfolio=None):
    docs = [d for d in load_portfolio_documents(portfolio) if d.metadata["type"] == "pnl_ranking"]
    assert len(docs) == 1
    assert docs[0].metadata["source_id"] == "fund:pnl_ranking"
    return docs[0].page_content


def test_ranking_names_largest_loss_and_gain_from_the_data():
    positions = load_portfolio()["positions"]
    worst = min(positions, key=lambda p: p["unrealized_pnl"])
    worst_pct = min(positions, key=lambda p: p["pnl_pct"])
    best = max(positions, key=lambda p: p["unrealized_pnl"])
    text = _ranking()
    assert f"Largest unrealized loss in dollars: {worst['ticker']} " in text
    assert f"Largest unrealized loss in percent: {worst_pct['ticker']} " in text
    assert f"Largest unrealized gain in dollars: {best['ticker']} " in text
    assert f"across all {len(positions)} positions" in text


def test_ranking_lists_every_position_in_order():
    positions = sorted(load_portfolio()["positions"], key=lambda p: p["unrealized_pnl"])
    text = _ranking().split("largest loss first: ", 1)[1]
    order = [part.split(". ", 1)[1].split(" ", 1)[0] for part in text.rstrip(".").split("; ")]
    assert order == [p["ticker"] for p in positions]


def test_ranking_without_losses_says_so():
    data = load_portfolio()
    data = {**data, "positions": [p for p in data["positions"] if p["unrealized_pnl"] > 0]}
    text = _ranking(data)
    assert "Positions with unrealized losses (0): none." in text
    assert "No position has an unrealized loss." in text


def test_agents_do_not_receive_the_ranking():
    from src.agents.context import market_sources, portfolio_sources, risk_sources

    data = load_portfolio()
    for sources in (market_sources(data), risk_sources(data), portfolio_sources(data)):
        assert all(s["source_id"] != "fund:pnl_ranking" for s in sources)
