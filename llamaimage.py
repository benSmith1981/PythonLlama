from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch

# Load the tokenizer, processor (for multimodal inputs), and model
model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)  # For vision-related tasks
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Define a multimodal input (text + image)
text_prompt = "Describe this image in detail and provide instructions: "
image_path = "image.png"  # Replace with the path to your image

# Preprocess the text and image inputs
inputs = processor(images=image_path, text=text_prompt, return_tensors="pt")
inputs = {key: val.to("cuda") for key, val in inputs.items()}  # Send inputs to GPU (if available)

# Generate a response
outputs = model.generate(**inputs, max_length=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the model's response
print("Model Response:")
print(response)
