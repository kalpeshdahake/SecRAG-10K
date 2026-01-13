from transformers import pipeline

class LLMGenerator:
    _qa = None

    def __init__(self):
        if LLMGenerator._qa is None:
            print("🔄 Loading extractive QA model...")
            LLMGenerator._qa = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                tokenizer="deepset/roberta-base-squad2"
            )

        self.qa = LLMGenerator._qa

    def extract(self, question: str, context: str):
        return self.qa(question=question, context=context)
