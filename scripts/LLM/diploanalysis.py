import os
import cohere

# Initialize Cohere client
api_key = os.getenv('COHERE_API_KEY')
if not api_key:
    raise ValueError("Set your COHERE_API_KEY environment variable!")
co_client = cohere.ClientV2(api_key)

def run_prompt(target, text):
    """Run a single prompt using Cohere's R7B model."""
    prompt = (
        f"Rewrite the context in which the target state is mentioned into a single sentence.\n\n"
        f"Target: {target}\n"
        f"Text: {text}\n\n"
        f"Instructions:\n"
        f"1. Begin with the target state.\n"
        f"2. Use as much of the original language as possible.\n"
        f"3. Ensure grammatical correctness.\n"
        f"4. Preserve the essence and meaning of the original text."
    )

    response = co_client.chat(
        model="command-r7b-12-2024",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.text.strip()

# Example usage
target = "France"
text = "France has played a significant role in shaping the climate policies discussed at the summit."
result = run_prompt(target, text)
print(f"Output:\n{result}")
