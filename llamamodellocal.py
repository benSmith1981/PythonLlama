import subprocess

# Path to the llama-cli binary and model
LLAMA_CLI_PATH = "/Users/darrenhuxtable/Documents/PythonLlama/llama.cpp/build/bin/llama-cli"

MODEL_PATH = "/Users/darrenhuxtable/Documents/PythonLlama/llama.cpp/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

def run_model(prompt, n_tokens=150, temp=0.8, top_k=50, top_p=0.95):
    """
    Runs the model with the given prompt and parameters.
    """
    try:
        # Construct the command to call llama-cli
        command = [
            LLAMA_CLI_PATH,
            "-m", MODEL_PATH,
            "-p", prompt,
            "-n", str(n_tokens),
            "--temp", str(temp),
            "--top-k", str(top_k),
            "--top-p", str(top_p)
        ]
        
        # Run the command and capture the output
        result = subprocess.run(command, text=True, capture_output=True)
        
        # If the command failed, print the error
        if result.returncode != 0:
            print("Error running the model:")
            print(result.stderr)
        else:
            # Print the output from the model
            print("Model Response:")
            print(result.stdout)
    
    except FileNotFoundError:
        print("Error: Could not find the llama-cli binary. Make sure the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    print("Welcome to the Llama Model!")
    print("Type your prompt below. Type 'exit' to quit.")
    while True:
        # Get the user's input prompt
        prompt = input("\nYour Prompt: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        
        # Run the model with the provided prompt
        run_model(prompt)

if __name__ == "__main__":
    main()
