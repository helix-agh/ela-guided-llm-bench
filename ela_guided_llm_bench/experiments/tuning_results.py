import json

import pandas as pd
from ela_guided_llm_bench.experiment_loader import row_to_function_info

IID = 2
DIM = 2
DIR = "results_27_04"

for fid in range(1, 20):
    df = pd.read_csv(f"{DIR}/llm_sr_2.5_flash_f{fid}_iid{IID}_dim{DIM}/generated_functions_info.csv")
    target_ela_features = json.load(open(f"{DIR}/llm_sr_2.5_flash_f{fid}_iid{IID}_dim{DIM}/target_ela_features.json"))
    best_row = df.iloc[df["distance_to_target"].argmin()]
    function_info = row_to_function_info(best_row)
    original_distance = function_info.distance_to_target
    function_info.optimize_params(target_ela_features=target_ela_features, max_evals=1000, algorithm="L-BFGS-B")
    distance_after_tuning = function_info.distance_to_target
    print(f"Function {fid} original distance: {original_distance}, distance after tuning: {distance_after_tuning}")
