import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

_loaded_llm = {}

def load_llm(model_name: str):
    if model_name in _loaded_llm:
        return _loaded_llm[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,  # torch_dtype → dtype
        device_map="auto"
    )
    model.eval()

    _loaded_llm[model_name] = (tokenizer, model)
    return tokenizer, model