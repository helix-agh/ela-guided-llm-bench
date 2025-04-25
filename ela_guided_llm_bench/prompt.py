PROMPT_FEW_SHOT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design.
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below.
You will be given a previous attempt, its ELA features, and the error between its features and the target.
Your objective is to **significantly modify** the previous function to **reduce this error**, particularly for the features with the largest errors.

**Target Normalized ELA Features:**
(These are the values the generated function's landscape should ideally exhibit)
{ela_features}

**ELA Feature Descriptions:**
- **`ela_meta.lin_simple.adj_r2`**: Adjusted R² of a linear model. High values suggest linearity.
- **`ela_meta.lin_w_interact.adj_r2`**: Adjusted R² of a linear model with pairwise interactions. High values suggest strong linear interactions.
- **`ela_meta.quad_simple.adj_r2`**: Adjusted R² of a quadratic model without interactions. High values suggest simple curvature.
- **`ela_meta.quad_w_interact.adj_r2`**: Adjusted R² of a full quadratic model. High values suggest complex interactions and curvature.
- **`ela_distr.skewness`**: Skewness of the objective value distribution. Positive skew means a longer tail towards high values; negative skew means a longer tail towards low values.
- **`nbc.nb_fitness.cor`**: Correlation between fitness and nearest-better connectivity. High values often indicate funnel-like structures.
- **`nbc.nn_nb.sd_ratio`**: Ratio of standard deviations (nearest neighbor distance / nearest-better distance). Values > 1 might indicate deception or multi-modality.
- **`fitness_distance.fitness_std`**: Standard deviation of objective values. Indicates the overall spread/range of the function values.
Remember that these different ELA features are not independent, and changes in one feature may affect others.

**Previous Attempts:**
{context}

**Guidance for Improvement based on Error:**

1.  **Analyze the Error:** Carefully examine the error of examples
2.  **Relate ELA to Function Structure:** Consider how different mathematical operations influence ELA features.
3.  **Perform Significant Mutations:** Based on the error analysis, design a new function that significantly improves the error.
4.  **Focus on the Goal:** Remember, the objective is not necessarily a 'nice' or 'standard' benchmark function, but one that specifically exhibits the target ELA features by minimizing the provided error.

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain (e.g., avoid `log(0)` or negative numbers if inputs can be negative).
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used (e.g., "Combines a quadratic bowl with sinusoidal modulation and pairwise interactions").
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.

Remember, your goal is to generate a function that minimizes the error between the ELA features and the target.
The best possible error is 0.
Diversity is key, each new example should be significantly different from the previous ones. Analyse previous attempts and try to add, replace, or remove mathematical components.
Try to generate functions with different landscape characteristics.
"""


PROMPT_ZERO_SHOT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design.
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below.

**Target Normalized ELA Features:**
(These are the values the generated function's landscape should ideally exhibit)
{ela_features}

**ELA Feature Descriptions:**
- **`ela_meta.lin_simple.adj_r2`**: Adjusted R² of a linear model. High values suggest linearity.
- **`ela_meta.lin_w_interact.adj_r2`**: Adjusted R² of a linear model with pairwise interactions. High values suggest strong linear interactions.
- **`ela_meta.quad_simple.adj_r2`**: Adjusted R² of a quadratic model without interactions. High values suggest simple curvature.
- **`ela_meta.quad_w_interact.adj_r2`**: Adjusted R² of a full quadratic model. High values suggest complex interactions and curvature.
- **`ela_distr.skewness`**: Skewness of the objective value distribution. Positive skew means a longer tail towards high values; negative skew means a longer tail towards low values.
- **`nbc.nb_fitness.cor`**: Correlation between fitness and nearest-better connectivity. High values often indicate funnel-like structures.
- **`nbc.nn_nb.sd_ratio`**: Ratio of standard deviations (nearest neighbor distance / nearest-better distance). Values > 1 might indicate deception or multi-modality.
- **`fitness_distance.fitness_std`**: Standard deviation of objective values. Indicates the overall spread/range of the function values.

**Guidance for Improvement based on Error:**

1.  **Relate ELA to Function Structure:** Consider how different mathematical operations influence ELA features.
2.  **Focus on the Goal:** Remember, the objective is not necessarily a 'nice' or 'standard' benchmark function, but one that specifically exhibits the target ELA features by minimizing the provided error.

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain (e.g., avoid `log(0)` or negative numbers if inputs can be negative).
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used (e.g., "Combines a quadratic bowl with sinusoidal modulation and pairwise interactions").
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.
"""

LLM_SR_PROMPT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design.
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below.

**Target Normalized ELA Features:**
(These are the values the generated function's landscape should ideally exhibit)
{ela_features}

**ELA Feature Descriptions:**
- **`ela_meta.lin_simple.adj_r2`**: Adjusted R² of a linear model. High values suggest linearity.
- **`ela_meta.lin_w_interact.adj_r2`**: Adjusted R² of a linear model with pairwise interactions. High values suggest strong linear interactions.
- **`ela_meta.quad_simple.adj_r2`**: Adjusted R² of a quadratic model without interactions. High values suggest simple curvature.
- **`ela_meta.quad_w_interact.adj_r2`**: Adjusted R² of a full quadratic model. High values suggest complex interactions and curvature.
- **`ela_distr.skewness`**: Skewness of the objective value distribution. Positive skew means a longer tail towards high values; negative skew means a longer tail towards low values.
- **`nbc.nb_fitness.cor`**: Correlation between fitness and nearest-better connectivity. High values often indicate funnel-like structures.
- **`nbc.nn_nb.sd_ratio`**: Ratio of standard deviations (nearest neighbor distance / nearest-better distance). Values > 1 might indicate deception or multi-modality.
- **`fitness_distance.fitness_std`**: Standard deviation of objective values. Indicates the overall spread/range of the function values.

**Previous Attempts:**
{context}

**Guidance for Improvement based on Error:**

1.  **Analyze the Error:** Carefully examine the error of examples
2.  **Relate ELA to Function Structure:** Consider how different mathematical operations influence ELA features.
3.  **Perform Significant Mutations:** Based on the error analysis, design a new function that significantly improves the error.
4.  **Focus on the Goal:** Remember, the objective is not necessarily a 'nice' or 'standard' benchmark function, but one that specifically exhibits the target ELA features by minimizing the provided error.
5.  **Analyse Params**: Analyse tuned parameters of the previous functions.

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray, params: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Params**: `params` is a 1D NumPy array. Each element of this array is a hyperparameter of the function. These hyperparameters will be optimized by the CMA-ES algorithm.
    These should be knobs that can be tuned to generate a function with the desired ELA features. For example, if the function consists of multiple components, the params could be the coefficients of the linear combination.
    Please, do not use more than 5 parameters. All parameters should be floating point numbers between 0 and 1. Hyperparameter optimization will start from 0.5 for all parameters.
    **IMPORTANT**: include immediately after your function docstring a one-line comment which explains number of parameters e.g. # n_params = 4
5.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain (e.g., avoid `log(0)` or negative numbers if inputs can be negative).
6.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used (e.g., "Combines a quadratic bowl with sinusoidal modulation and pairwise interactions").
7.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.

Remember, your goal is to generate a function that minimizes the error between the ELA features and the target.
The best possible error is 0.
Diversity is key, each new example should be significantly different from the previous ones.
Do not use very similar functions as hyperparameters are optimized.
Analyse previous attempts and try to add, replace, or remove mathematical components.
Try to generate functions with different landscape characteristics.
"""
