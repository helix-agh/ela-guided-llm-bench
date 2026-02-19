import pandas as pd

if __name__ == "__main__":
    eotf_time = pd.read_csv("./time_measurements/eotf_google/gemini-2.0-flash-001_dim2_iid1/timing_results.csv")
    gp_time = pd.read_csv("./time_measurements/gp_baseline_gp_dim2_iid1/timing_results.csv")
    zero_shot_time = pd.read_csv(
        "./time_measurements/zero_shot_google/gemini-2.0-flash-001_dim2_iid1/timing_results.csv"
    )
    llamea_time = pd.read_csv("./time_measurements/llamea_google/gemini-2.0-flash-001_dim2_iid1/timing_results.csv")

    label_to_time_df = {
        "EoTF": eotf_time,
        "GP": gp_time,
        "ZeroShot": zero_shot_time,
        "LLaMEA": llamea_time,
    }

    for label, time_df in label_to_time_df.items():
        print(label)
        print(time_df["time_seconds"].mean())
