from pipeline.utils import extract_sentence_containing, extract_all_money_numbers


def extract_total_revenue(chunks):
    for c in chunks:
        t = c["text"].lower()
        if "total net sales" in t or "total revenues" in t:
            nums = extract_all_money_numbers(c["text"])
            if nums:
                return f"${max(nums):,} million", c
    return None, None


def extract_term_debt(chunks):
    for c in chunks:
        if "total term debt" in c["text"].lower():
            nums = extract_all_money_numbers(c["text"])
            if nums:
                # choose the final total (not principal)
                return f"${min(nums):,} million", c
    return None, None


def extract_unresolved_staff_comments(chunks):
    for c in chunks:
        if c["metadata"].get("item") == "Item 1B":
            if "no" in c["text"].lower():
                return "No", c
    return None, None


def extract_vehicle_list(chunks):
    for c in chunks:
        if "manufacture" in c["text"].lower():
            sentence = extract_sentence_containing(
                c["text"],
                ["manufacture"]
            )
            if sentence:
                return sentence, c
    return None, None


def extract_automotive_sales_percentage(chunks):
    total = None
    automotive = None

    for c in chunks:
        t = c["text"].lower()
        nums = extract_all_money_numbers(c["text"])

        if "total revenues" in t:
            total = max(nums) if nums else total
        if "automotive sales" in t:
            automotive = max(nums) if nums else automotive

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
            ["lease", "pass"]
        )
        if sent:
            return sent, c
    return None, None
