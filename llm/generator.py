from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMGenerator:
    _model = None
    _tokenizer = None

    def __init__(self, model_name="google/flan-t5-base"):
        if LLMGenerator._model is None:
            print("🔄 Loading lightweight LLM (one-time)...")
            LLMGenerator._tokenizer = AutoTokenizer.from_pretrained(model_name)
            LLMGenerator._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.tokenizer = LLMGenerator._tokenizer
        self.model = LLMGenerator._model

    def generate(self, prompt: str, max_new_tokens=128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
