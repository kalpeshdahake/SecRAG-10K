from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMGenerator:
    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,   # VERY IMPORTANT (no hallucination)
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
