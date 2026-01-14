import re

def extract_sentence_containing(text: str, keywords):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        if all(k.lower() in s.lower() for k in keywords):
            return s.strip()
    return None


def extract_all_money_numbers(text: str):
    nums = re.findall(r"\$?\d{1,3}(?:,\d{3})+", text)
    return [int(n.replace("$", "").replace(",", "")) for n in nums]
