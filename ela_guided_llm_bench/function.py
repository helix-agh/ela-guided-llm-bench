import re
import ast
import logging
from dataclasses import dataclass
import numpy as np
from .ela import get_ela_features

logger = logging.getLogger(__name__)


@dataclass
class Function:
    source_code: str
    description: str
    ela_features: dict[str, float]
    # target: float


class FunctionParser:
    def __init__(
        self,
        ela_dim: int,
        # target_ela_features: dict[str, float],
        random_seed: int = 42,
    ):
        self.ela_dim = ela_dim
        self.random_seed = random_seed
        # self.target_ela_features = target_ela_features

    def parse(self, model_response: str) -> Function | None:
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

        ela_features = self.get_ela_features(function_str)
        # target = np.linalg.norm(
        #     np.array(list(ela_features.values()))
        #     - np.array(list(self.target_ela_features.values()))
        # )
        return Function(
            source_code=function_str,
            description=docstring,
            ela_features=ela_features,
            # target=target,
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

    def get_ela_features(self, function_str: str) -> dict[str, float]:
        namespace = {}
        exec(function_str, namespace)
        return get_ela_features(namespace["problem"], self.ela_dim, self.random_seed)
