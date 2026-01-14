from pipeline.utils import extract_sentence_containing, extract_all_money_numbers


def extract_total_revenue(chunks):
    for c in chunks:
        if c["metadata"].get("item") != "Item 8":
            continue

        t = c["text"].lower()
        if "total net sales" in t or "total revenues" in t:
            nums = extract_all_money_numbers(c["text"])
            if nums:
                return f"${nums[0]:,} million", c
    return None, None


def extract_term_debt(chunks):
    for c in chunks:
        if "total term debt" in c["text"].lower():
            nums = extract_all_money_numbers(c["text"])
            if nums:
                return f"${nums[0]:,} million", c
    return None, None


def extract_unresolved_staff_comments(chunks):
    for c in chunks:
        if c["metadata"].get("item") == "Item 1B":
            if "no unresolved staff comments" in c["text"].lower():
                return "No", c
    return None, None


def extract_shares_outstanding(chunks):
    for c in chunks:
        if "shares" in c["text"].lower() and "outstanding" in c["text"].lower():
            nums = extract_all_money_numbers(c["text"])
            if nums:
                return f"{nums[0]:,} shares", c
    return None, None


def extract_vehicle_list(chunks):
    for c in chunks:
        sent = extract_sentence_containing(
            c["text"],
            ["currently", "manufacture"]
        )
        if sent:
            return sent, c
    return None, None


def extract_automotive_sales_percentage(chunks):
    total = None
    automotive = None

    for c in chunks:
        if c["metadata"].get("item") != "Item 7":
            continue

        t = c["text"].lower()
        nums = extract_all_money_numbers(c["text"])

        if "total revenues" in t:
            total = nums[0] if nums else total

        if "automotive sales" in t and "excluding leasing" in t:
            automotive = nums[0] if nums else automotive

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


def extract_lease_passthrough(chunks):
    for c in chunks:
        sent = extract_sentence_containing(
            c["text"],
            ["lease", "pass-through"]
        )
        if sent:
            return sent.split(".")[0] + ".", c
    return None, None
