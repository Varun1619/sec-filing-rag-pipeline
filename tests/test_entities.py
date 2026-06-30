"""Tests for entity extraction."""

from __future__ import annotations


from src.ingest.entities import extract_entities


def test_ticker_dollar():
    entities = extract_entities("$AAPL reported strong earnings.")
    tickers = [e for e in entities if e.entity_type == "TICKER"]
    assert any(e.value == "AAPL" for e in tickers)


def test_ticker_exchange():
    entities = extract_entities("Trading as (NASDAQ: GOOG) on the exchange.")
    tickers = [e for e in entities if e.entity_type == "TICKER"]
    assert any(e.value == "GOOG" for e in tickers)


def test_money():
    entities = extract_entities("Revenue was $1.2 billion for the quarter.")
    money = [e for e in entities if e.entity_type == "MONEY"]
    assert len(money) >= 1
    assert any("1.2" in e.value for e in money)


def test_fiscal_period():
    entities = extract_entities("Results for Q3 2023 were strong.")
    fp = [e for e in entities if e.entity_type == "FISCAL_PERIOD"]
    assert len(fp) >= 1
    assert any("Q3" in e.value and "2023" in e.value for e in fp)


def test_metric():
    entities = extract_entities("Net income and EBITDA both rose sharply.")
    metrics = {e.value for e in entities if e.entity_type == "METRIC"}
    assert "net income" in metrics
    assert "ebitda" in metrics


def test_cik():
    entities = extract_entities("CIK: 0000320193 filed a 10-K.")
    ciks = [e for e in entities if e.entity_type == "CIK"]
    assert any("320193" in e.value for e in ciks)


def test_empty_text():
    entities = extract_entities("")
    assert entities == []
