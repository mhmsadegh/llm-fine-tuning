# llm-fine-tuning
# 🧵 Gemma-3B Fashion Query Rewriting (LoRA)

This is a fine-tuned version of [`unsloth/gemma-3-4b-it`](https://huggingface.co/unsloth/gemma-3-4b-it) using the [Unsloth](https://github.com/unslothai/unsloth) library and LoRA (Low-Rank Adaptation).

## 🔍 Task

> **Fashion Query Rewriting**: Extract product attributes from the input query.
> For example, converting:
> - "پیژامه هلو کیتی"" →"{\"Category\": [\"pants\"], \"Persian-Product-Name\": [\"پیژامه\"], \"Gender\": [], \"English-Query\": [\"Hello Kitty pajamas\"], \"Color\": [\"pink\"], \"Similar-Keywords\": []}"}"

## 🧪 Model Details

- **Base Model**: `unsloth/gemma-3-4b-it`
- **LoRA**: r=16, alpha=16
- **Fine-tuning Framework**: [Unsloth](https://github.com/unslothai/unsloth)
- **Precision**: 16-bit merged

## 🚀 Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mhmsadegh/gemma-3-4b-it-fashion-query-rewriting")
tokenizer = AutoTokenizer.from_pretrained("mhmsadegh/gemma-3-4b-it-fashion-query-rewriting")

prompt = "Query: red hoodie\nRewrite:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))‍‍‍‍‍
```

## 📦 Files

This repo includes:
	•	Tokenizer files
	•	Merged model weights (16-bit)
	•	Chat template & config files
	•	Special tokens for alignment

## Author

Fine-tuned by https://www.linkedin.com/in/mhmsadegh/

