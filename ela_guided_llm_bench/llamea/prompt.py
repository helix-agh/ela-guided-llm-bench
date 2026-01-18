INITIAL_PROMPT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design.
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below.

**Target Normalized ELA Features:**
(These are the values the generated function's landscape should ideally exhibit)
{ela_features}

**ELA Feature Descriptions:**
- **`ela_meta.lin_simple.adj_r2`**: Adjusted R^2 of a linear model. High values suggest linearity.
- **`ela_meta.lin_w_interact.adj_r2`**: Adjusted R^2 of a linear model with pairwise interactions.
- **`ela_meta.quad_simple.adj_r2`**: Adjusted R^2 of a quadratic model without interactions.
- **`ela_meta.quad_w_interact.adj_r2`**: Adjusted R^2 of a full quadratic model.
- **`ela_distr.skewness`**: Skewness of the objective value distribution.
- **`nbc.nb_fitness.cor`**: Correlation between fitness and nearest-better connectivity.
- **`nbc.nn_nb.sd_ratio`**: Ratio of standard deviations (nearest neighbor distance / nearest-better distance).
- **`fitness_distance.fitness_std`**: Standard deviation of objective values.

**Code Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        pass
    ```
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`.
5.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.
6.  **The function must be deterministic**: Do not use np.random or any stochastic elements.

Provide the Python code and a one-line description with the main idea (without enters).
Description should explain main idea of the function (it should mention mathematical operations used).
Description should be highly specific, do not use general terms like "function", "problem", "matching ELA features" or "approximating ELA targets".
This description is super important, it is used to evolve the function.
Give the response in the format:
<example>
# Description: <short-description>
# Code:
```python
<code>
```
</example>
"""

EVOLUTION_PROMPT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design.
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below.

**Target Normalized ELA Features:**
(These are the values the generated function's landscape should ideally exhibit)
{ela_features}

**ELA Feature Descriptions:**
- **`ela_meta.lin_simple.adj_r2`**: Adjusted R^2 of a linear model. High values suggest linearity.
- **`ela_meta.lin_w_interact.adj_r2`**: Adjusted R^2 of a linear model with pairwise interactions.
- **`ela_meta.quad_simple.adj_r2`**: Adjusted R^2 of a quadratic model without interactions.
- **`ela_meta.quad_w_interact.adj_r2`**: Adjusted R^2 of a full quadratic model.
- **`ela_distr.skewness`**: Skewness of the objective value distribution.
- **`nbc.nb_fitness.cor`**: Correlation between fitness and nearest-better connectivity.
- **`nbc.nn_nb.sd_ratio`**: Ratio of standard deviations (nearest neighbor distance / nearest-better distance).
- **`fitness_distance.fitness_std`**: Standard deviation of objective values.

**Code Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        pass
    ```
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`.
5.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.
6.  **The function must be deterministic**: Do not use np.random or any stochastic elements.

The current population of algorithms already evaluated is:
{population_summary}

The selected solution to update is:
{description}

With code:
{source_code}

**IMPORTANT:**
{mutation_operator}

Provide the Python code and a one-line description with the main idea (without enters).
Description should explain main idea of the function (it should mention mathematical operations used).
Description should be highly specific, do not use general terms like "2D function", "problem", "matching ELA features", or "approximating ELA targets".
This description is super important, it is used to evolve the function.

Give the response in the format:
<example>
# Description: <short-description>
# Code:
```python
<code>
```
</example>
"""
