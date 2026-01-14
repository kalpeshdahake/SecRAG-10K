from typing import Tuple, Optional
from pipeline.utils import (
    extract_all_money_numbers,
    extract_sentence_containing,
    split_lines,
    extract_date
)


# ---------- Apple ----------

def extract_total_revenue(chunks) -> Tuple[Optional[str], Optional[dict]]:
    for c in chunks:
        if "total net sales" in c["text"].lower():
            for line in split_lines(c["text"]):
                if "total net sales" in line.lower():
                    nums = extract_all_money_numbers(line)
                    if nums:
                        # First column = FY 2024
                        return f"${nums[0]:,} million", c
    return None, None


def extract_shares_outstanding(chunks):
    for c in chunks:
        if "shares of common stock" in c["text"].lower():
            sent = extract_sentence_containing(
                c["text"],
                ["shares", "outstanding"]
            )
            if sent:
                return sent, c
    return None, None


def extract_term_debt(chunks):
    for c in chunks:
        for line in split_lines(c["text"]):
            if "total term debt" in line.lower():
                nums = extract_all_money_numbers(line)
                if nums:
                    return f"${nums[-1]:,} million", c
    return None, None


def extract_unresolved_staff_comments(chunks):
    for c in chunks:
        if "unresolved staff comments" in c["text"].lower():
            if "no" in c["text"].lower():
                return "No", c
    return None, None


def extract_filing_date(chunks):
    for c in chunks:
        if "signature" in c["text"].lower():
            date = extract_date(c["text"])
            if date:
                return date, c
    return None, None


# ---------- Tesla ----------

def extract_total_revenue_tesla(chunks):
    for c in chunks:
        for line in split_lines(c["text"]):
            if "total revenues" in line.lower():
                nums = extract_all_money_numbers(line)
                if nums:
                    return f"${nums[-1]:,} million", c
    return None, None


def extract_automotive_sales_percentage(chunks):
    total = None
    automotive = None

    for c in chunks:
        for line in split_lines(c["text"]):
            l = line.lower()
            nums = extract_all_money_numbers(line)

            if "total revenues" in l and nums:
                total = nums[-1]

            if "automotive sales" in l and nums:
                automotive = nums[-1]

    if total and automotive:
        pct = round((automotive / total) * 100)
        return f"Approximately {pct}%", chunks[0]

    return None, None


def extract_dependency_statement(chunks):
    for c in chunks:
        sent = extract_sentence_containing(
            c["text"],
            ["elon", "musk"]
        )
        if sent:
            return sent, c
    return None, None


def extract_vehicle_list(chunks):
    for c in chunks:
        if "different consumer vehicles" in c["text"].lower():
            sent = extract_sentence_containing(
                c["text"],
                ["vehicles"]
            )
            if sent:
                return sent, c
    return None, None


def extract_lease_passthrough(chunks):
    for c in chunks:
        sent = extract_sentence_containing(
            c["text"],
            ["lease", "pass-through"]
        )
        if sent:
            return sent, c
    return None, None
