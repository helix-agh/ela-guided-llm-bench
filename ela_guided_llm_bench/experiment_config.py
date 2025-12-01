import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExperimentConfig:
    model: str
    fid: int
    iid: int
    dim: int
    method: str
    parent_dir: str | None = None

    @classmethod
    def from_dir(
        cls,
        dir_name: str,
        parent_dir: str | None = None,
    ) -> "ExperimentConfig":
        method = cls._extract_method_name(dir_name)
        fid = cls._extract_function_id(dir_name)
        model = cls._extract_model_name(dir_name)
        iid = cls._extract_iid(dir_name)
        dim = cls._extract_dim(dir_name)
        return cls(method=method, fid=fid, model=model, iid=iid, dim=dim, parent_dir=parent_dir)

    @property
    def dir_name(self) -> str:
        model_clean = self.model.replace("-", "_").replace(".", "_").replace("/", "_")
        date_str = datetime.now().strftime("%Y_%m_%d")
        method = self.method
        parent_dir = self.parent_dir or f"./{method}_dim{self.dim}_{date_str}"
        return f"{parent_dir}/{method}_{model_clean}_f{self.fid}_iid{self.iid}_dim{self.dim}"

    @property
    def csv_path(self) -> str:
        return f"{self.dir_name}/generated_functions_info.csv"

    @property
    def target_ela_features_path(self) -> str:
        return f"{self.dir_name}/target_ela_features.json"

    @classmethod
    def _extract_function_id(cls, dir_name: str) -> int:
        m = re.search(r"f_?(\d+)", dir_name)
        if not m:
            raise ValueError(f"No function ID found in '{dir_name}'")
        return int(m.group(1))

    @classmethod
    def _extract_iid(cls, dir_name: str) -> int:
        m = re.search(r"iid_?(\d+)", dir_name)
        if not m:
            raise ValueError(f"No instance ID found in '{dir_name}'")
        return int(m.group(1))

    @classmethod
    def _extract_dim(cls, dir_name: str) -> int:
        m = re.search(r"dim_?(\d+)", dir_name)
        if not m:
            raise ValueError(f"No dimension found in '{dir_name}'")
        return int(m.group(1))

    @classmethod
    def _extract_model_name(cls, dir_name: str) -> str:
        _, remainder = dir_name.split("_", 1)
        f_match = re.search(r"_f_?\d+", remainder)
        model = remainder[: f_match.start()]
        return model

    @classmethod
    def _extract_method_name(cls, dir_name: str) -> str:
        return dir_name.split("_")[0]
