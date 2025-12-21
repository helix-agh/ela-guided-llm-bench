from ela_guided_llm_bench.experiment import BenchmarkExperiment

if __name__ == "__main__":
    bbob_results_2d = BenchmarkExperiment.from_dir("./eoh_dim2_2025_06_19")
    bbob_results_2d.plot_contour_grid(function_ids=[3, 8, 12, 23], file_path="images/eoh_bbob_2d_contours.png")
