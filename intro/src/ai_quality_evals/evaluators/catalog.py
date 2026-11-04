"""Bounded DemoShop price/availability checks; unsupported language needs review."""

import re
from decimal import Decimal
from typing import Literal

from pydantic import Field

from ai_quality_evals.evaluators.matching import normalize
from ai_quality_evals.foundations import Contract, Nonblank

RULE_VERSION = "1.0.0"


class Product(Contract):
    product_id: Nonblank
    name: Nonblank
    price_inr: int = Field(ge=0)
    stock: int = Field(ge=0)


class Verdict(Contract):
    status: Literal["pass", "fail", "review"]
    expected: str
    observed: str | None
    reason: str


def check_catalog(
    answer: str, product: Product, check: Literal["price", "stock"]
) -> Verdict:
    """Compare a fully recognized answer with source facts; never substring-grade."""
    if check not in {"price", "stock"}:
        raise ValueError("Only price and stock checks are supported")
    expected = (
        str(product.price_inr)
        if check == "price"
        else ("available" if product.stock > 0 else "unavailable")
    )
    text = normalize(answer).rstrip(".!")
    if not text:
        return Verdict(
            status="fail", expected=expected, observed=None, reason="missing_answer"
        )
    name = re.escape(normalize(product.name))
    if check == "price":
        number = r"[0-9]+(?:\.[0-9]{1,2})?"
        prefix = rf"(?:(?:the )?{name} (?:costs?|is priced at) )?"
        money = (
            rf"(?:(?:inr|rs\.?|\u20b9)\s*(?P<before>{number})"
            rf"|(?P<after>{number})\s*(?:inr|rupees))"
        )
        match = re.fullmatch(prefix + money, text)
        if match:
            observed = Decimal(match.group("before") or match.group("after"))
            return Verdict(
                status="pass" if observed == Decimal(product.price_inr) else "fail",
                expected=expected,
                observed=str(observed),
                reason="price_matches_catalog"
                if observed == product.price_inr
                else "price_mismatch",
            )
    else:
        pattern = (
            rf"(?:(?P<prefix>yes|no), )?(?:(?:the )?{name} is )?"
            r"(?P<stock>out of stock|not in stock|in stock|unavailable|available)"
        )
        match = re.fullmatch(pattern, text)
        if match:
            available = match.group("stock") in {"in stock", "available"}
            observed = "available" if available else "unavailable"
            prefix = match.group("prefix")
            if prefix and (prefix == "yes") != available:
                return Verdict(
                    status="fail",
                    expected=expected,
                    observed=observed,
                    reason="contradictory_yes_no_prefix",
                )
            return Verdict(
                status="pass" if available == (product.stock > 0) else "fail",
                expected=expected,
                observed=observed,
                reason="availability_matches_catalog"
                if observed == expected
                else "availability_mismatch",
            )
    return Verdict(
        status="review",
        expected=expected,
        observed=None,
        reason="unsupported_or_ambiguous_language",
    )
