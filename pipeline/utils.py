import re
from typing import List, Optional


def split_lines(text: str) -> List[str]:
    return [l.strip() for l in text.split("\n") if l.strip()]


def split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text)


def extract_all_money_numbers(text: str) -> List[int]:
    nums = re.findall(r"\$?\d{1,3}(?:,\d{3})+", text)
    return [int(n.replace("$", "").replace(",", "")) for n in nums]


def extract_sentence_containing(text: str, keywords: List[str]) -> Optional[str]:
    for s in split_sentences(text):
        if all(k.lower() in s.lower() for k in keywords):
            return s.strip()
    return None


def extract_date(text: str) -> Optional[str]:
    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        text
    )
    return match.group(0) if match else None
