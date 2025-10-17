import subprocess

def reactive_agent(prompt):
    """
    A simple reactive agent that processes a prompt and returns a response.
    This agent uses a subprocess to call an external command-line tool for processing.
    """
    try:
        # Call an external command-line tool to process the prompt.
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        response = result.stdout.strip()
        return response
    except FileNotFoundError:
        # If the external tool isn't installed or not on PATH
        return f"(fallback) {prompt}"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"
    
if __name__ == "__main__":
    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in ['exit', 'quit']:
            break
        response = reactive_agent(user_prompt)
        print(f"Agent: {response}")