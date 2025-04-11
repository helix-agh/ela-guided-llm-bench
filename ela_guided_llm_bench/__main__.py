from .gemini import generate_with_gemini
from .prompt import PROMPT


def main():
    prompt = PROMPT.format(
        ela_features={},
        example_source_code="",
        example_error="",
    )
    response = generate_with_gemini(prompt)
    print(response)
