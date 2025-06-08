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
You are an expert mathematical function designer specializing in Exploratory Landscape Analysis (ELA), optimization benchmarks, and high-dimensional problems.
Your task is to generate a single, novel synthetic benchmark function in Python intended for testing global optimization algorithms.

The primary objective is to create a function whose ELA features, after linear scaling of function values to [0, 1], closely match the target values provided below.
You will receive the target ELA features, details of the previous function generated, its calculated ELA features, and the error (difference) between its features and the target.
Your goal is to **design a new function**, making **significant structural modifications** to the previous attempt, to **minimize the error**.


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
6.  **Be Creative**: Use varied mathematical operations.

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

LLM_SR_PROMPT_CROSSOVER = """
You are an expert mathematical function designer specializing in Exploratory Landscape Analysis (ELA), optimization benchmarks, and high-dimensional problems.
Your task is to generate a single, novel synthetic benchmark function in Python by **combining elements from two provided parent functions**.

The primary objective is to create a new 'child' function whose ELA features, after linear scaling of function values to [0, 1], closely match the target values provided below.
You will receive the target ELA features, details of **two parent functions**, their calculated ELA features, and their respective errors relative to the target.
Your goal is to **design a hybrid function** by **intelligently blending components or structures** from both parents to **minimize the error** against the target ELA features, potentially inheriting beneficial characteristics from each parent.

**Target Normalized ELA Features:**
(The landscape characteristics the generated function should ideally exhibit)
{ela_features}

**ELA Feature Descriptions:**
- `ela_meta.lin_simple.adj_r2`: Measures linearity (higher means more linear).
- `ela_meta.lin_w_interact.adj_r2`: Measures linearity considering pairwise interactions.
- `ela_meta.quad_simple.adj_r2`: Measures simple quadratic curvature (without interactions).
- `ela_meta.quad_w_interact.adj_r2`: Measures complex quadratic curvature and interactions.
- `ela_distr.skewness`: Measures asymmetry of the objective value distribution.
- `nbc.nb_fitness.cor`: Correlation between fitness and nearest-better connectivity (high values suggest funnels).
- `nbc.nn_nb.sd_ratio`: Ratio of standard deviations (nearest neighbor distance / nearest-better distance) (values > 1 hint at multi-modality/deception).
- `fitness_distance.fitness_std`: Standard deviation of objective values (overall range/spread).

**Parent Function Details:**
(Context including code, ELA features, and error vectors for two parent functions)

**Parent 1:**
{parent_1_context}

**Parent 2:**
{parent_2_context}

**Guidance for Crossover:**

1.  **Analyze Parents:** Examine the structure, ELA features, and errors of *both* parent functions. Identify which parent performs better for specific ELA features. Note the key mathematical components in each.
2.  **Identify Components for Combination:** Select specific terms, structures, or mathematical operations from each parent function that could be beneficial when combined.
3.  **Design a Hybrid Structure:** Create the new function by blending the selected components.
4.  **Utilize Parameters Effectively:** The `params` array (1 to 5 parameters) for the *new* function should control meaningful aspects of the combination or the properties of the integrated components. They allow CMA-ES to fine-tune the blend.
5.  **Focus on the Goal:** The primary aim is minimizing the ELA feature error of the *child* function by leveraging the strengths of the parents. The resulting function might be more complex than either parent.

**Implementation Requirements:**

1.  **Language & Libraries**: Python 3. Use **only NumPy** (`import numpy as np`). No other libraries.
2.  **Function Signature**: Must match exactly:
    ```python
    def problem(x: np.ndarray, params: np.ndarray) -> float:
        # Docstring explaining the hybrid structure and how parent components are combined.
        # n_params = K  <- IMPORTANT: Include this comment immediately after docstring with K = number of params used.
        # Ensure K is between 1 and 5 (inclusive).

        # --- Function implementation ---
        pass # Replace with your function code
    ```
3.  **Input `x`**: A 1D NumPy array `x` of shape `(N,)`, where `N` is the problem dimension.
4.  **Input `params`**: A 1D NumPy array `params` for the *new* function, shape `(K,)`. All elements must be used and treated as values between 0 and 1. $1 \le K \le 5$.
5.  **Domain**: The function must be well-defined for `x` within the hypercube `[-5, 5]^N`. Handle potential numerical issues (e.g., `log(0)`, `sqrt(negative)`, division by zero).
6.  **Output**: A single floating-point number.
7.  **Docstring**: Provide a concise docstring explaining the mathematical components, specifying which parts originate from which parent (conceptually), and how they are combined.
8.  **Code Block**: The final output must be a single Python code block containing only the `import numpy as np` statement and the `problem` function definition. No extra text or explanations outside the function.

Generate the Python code for the *new, hybrid* `problem` function based on combining elements from the provided parent functions to match the target ELA features.
"""

I1_PROMPT = """
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

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
    Your code MUST BE included in a markdown code block.
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain.
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used.
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.
"""

E1_PROMPT = """
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

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
    Your code MUST BE included in a markdown code block.
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain.
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used.
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.

**HISTORY:**
You already generated these functions:
{context}

**INSTRUCTIONS:**
Please help me create a new function that has a totally different form from the given ones.
"""


E2_PROMPT = """
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

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
    Your code MUST BE included in a markdown code block.
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain.
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used.
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.

**HISTORY:**
You already generated these functions:
{context}

**INSTRUCTIONS:**
Please help me create a new function that has a totally different form from the given ones but can be motivated from them.
Firstly, identify the common backbone idea in the provided functions.
Secondly, based on the backbone idea create a new solution.
"""

M1_PROMPT = """
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

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
    Your code MUST BE included in a markdown code block.
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain.
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used.
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.

**GENERATED FUNCTION:**
You already generated this function:
{context}

**INSTRUCTIONS:**
Please assist me in creating a new function that has a different form but can be a modified version of the function provided.
"""

M2_PROMPT = """
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

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
    Your code MUST BE included in a markdown code block.
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain.
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used.
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.

**GENERATED FUNCTION:**
You already generated this function:
{context}

**INSTRUCTIONS:**
Please identify the main parameters of the generated function and assist me in creating a new version of the function with improved parameter settings.
"""

M3_PROMPT = """
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

**Implementation Requirements:**

1.  **Language & Libraries**: Implement the function in Python, using **only NumPy** for mathematical operations.
2.  **Function Signature**:
    ```python
    def problem(x: np.ndarray) -> float:
        # Docstring goes here
        pass
    ```
    Your code MUST BE included in a markdown code block.
3.  **Input**: `x` is a 1D NumPy array of shape `(N,)`.
4.  **Domain**: The function should be designed considering the domain `[-5, 5]^N`. Ensure operations are valid within this domain.
5.  **Docstring**: Include a concise docstring explaining the mathematical structure of the function. If possible, include the formula. Be specific about the components used.
6.  **Self-Contained Code**: The final output block should only contain the necessary import (`import numpy as np`) and the function definition.

**GENERATED FUNCTION:**
You already generated this function:
{context}

First, you need to identify the main components in the function above.
Next, analyze whether any of these components can be overfit to the specific sample of points used to calculate ELA features.
Then, based on your analysis, simplify the components to enhance the generalization to other samples.
"""
