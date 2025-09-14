from datetime import datetime


def generate_dir_name(model: str, fid: int, iid: int, dim: int, method: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    model_clean = model.replace("-", "_").replace(".", "_")
    return f"./eoh_results_{dim}_{iid}_{dim}_{date_str}/{method}_{model_clean}_f{fid}_iid{iid}_dim{dim}"
