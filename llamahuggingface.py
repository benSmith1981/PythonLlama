from transformers import AutoTokenizer
from huggingface_hub import login

# Authenticate using your Hugging Face token
login(token="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")
# Load tokenizer directly from Hugging Face
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input text
input_text = "Explain the concept of machine learning."
tokens = tokenizer(input_text, return_tensors="pt")

# Decode tokens (simulate processing)
decoded_text = tokenizer.decode(tokens["input_ids"][0])
print("\nDecoded Input:")
print(decoded_text)
