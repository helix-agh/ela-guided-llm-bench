import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from .ela import get_distance, get_ela_features

logger = logging.getLogger(__name__)


def features_to_prompt(features: dict) -> str:
    rounded_features = {k: round(v, 2) for k, v in features.items()}
    return json.dumps(rounded_features)


@dataclass(frozen=True)
class FunctionInfo:
    function: Callable
    source_code: str
    description: str
    ela_features: dict[str, float]
    distance_to_target: float

    def __str__(self) -> str:
        ela_features_formatted = features_to_prompt(self.ela_features)
        ela_features_str = f"**ELA Features:**\n{ela_features_formatted}"
        source_code_formatted = f"```python\n{self.source_code.strip()}\n```"
        source_code_str = f"**Previously Generated Function:**\n{source_code_formatted}"
        error_str = f"**Error (Previous ELA - Target ELA):**\n{round(self.distance_to_target, 2)}"
        return f"<function_info>{source_code_str}\n{ela_features_str}\n{error_str}</function_info>"


class FunctionParser:
    def __init__(
        self,
        ela_dim: int,
        target_ela_features: dict[str, float],
        random_seed: int = 42,
    ):
        self.ela_dim = ela_dim
        self.random_seed = random_seed
        self.target_ela_features = target_ela_features

    def parse(self, model_response: str) -> FunctionInfo | None:
        function_str = self.extract_code(model_response)
        if not function_str:
            logger.error("No function found in response")
            return None

        if not self.validate_function_syntax(function_str):
            logger.error("Invalid function syntax")
            return None

        if "import numpy" not in function_str:
            function_str = "import numpy as np\n\n" + function_str

        docstring = self.extract_docstring(function_str)
        namespace: dict[str, Any] = {}
        exec(function_str, namespace)
        ela_features = get_ela_features(namespace["problem"], self.ela_dim, self.random_seed)
        return FunctionInfo(
            function=namespace["problem"],
            source_code=function_str,
            description=docstring,
            ela_features=ela_features,
            distance_to_target=get_distance(ela_features, self.target_ela_features),
        )

    def validate_function_syntax(self, function_str: str) -> bool:
        try:
            ast.parse(function_str)
            return True
        except SyntaxError:
            return False

    def extract_code(self, text: str) -> str | None:
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None

    def extract_docstring(self, function_str: str) -> str | None:
        pattern = r"\"\"\"(.*?)\"\"\""
        match = re.search(pattern, function_str, re.DOTALL)
        return match.group(1) if match else None
