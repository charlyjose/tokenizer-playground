import tiktoken
from typing import List

# Initialize the global tokenizer for encoding prompts
tokenizer = tiktoken.get_encoding("cl100k_base")


def get_token(prompt: str) -> List[int]:
    """Encode a prompt into token IDs using the global tokenizer."""
    return tokenizer.encode(prompt)


def main():
    prompt = "Twinkle, twinkle, little"
    print(f"Original Prompt: {prompt}")

    tokens = get_token(prompt)
    for token_id in tokens:
        # repr() makes whitespace characters visible in the output
        print(f"{token_id}: {repr(tokenizer.decode([token_id]))}")


if __name__ == "__main__":
    main()
