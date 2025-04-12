from .gemini import generate_with_gemini
from .prompt import PROMPT_ZERO_SHOT


def main():
    prompt = PROMPT_ZERO_SHOT.format(ela_features="")
    response = generate_with_gemini(prompt)
    print(response)
