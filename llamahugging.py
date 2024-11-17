from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model name and the token
model_name = "meta-llama/Llama-3.2-1B-Instruct"
token = "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"  # Replace with your actual token

# Load the tokenizer and model with the token
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

# Sample input text
input_text = "Explain the concept of machine learning."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Response:")
print(response)
