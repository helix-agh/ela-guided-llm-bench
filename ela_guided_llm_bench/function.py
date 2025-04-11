import ast
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from .ela import get_distance, get_ela_features

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    function: Callable
    source_code: str
    description: str
    ela_features: dict[str, float]
    distance_to_target: float


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
