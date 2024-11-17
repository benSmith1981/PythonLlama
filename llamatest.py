from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"MPS available: {torch.backends.mps.is_available()}")
# Define the local path to the model
# local_model_path = "/Users/darrenhuxtable/.llama/checkpoints/Llama3.2-1B-Instruct"

# Specify the local path to the model
local_model_path = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the desired model

# Load the model and tokenizer from the local path
print("Loading the model. This might take a while...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

print("Model loaded successfully!")

# Interactive loop for asking questions
print("You can now ask questions. Type 'exit' to quit.\n")

while True:
    # Get user input
    question = input("Your question: ")
    if question.lower() == "exit":
        print("Exiting. Goodbye!")
        break

    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt")

    # Generate a response
    print("Generating response...")
    outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model Response: {response}\n")

