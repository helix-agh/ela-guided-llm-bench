ELA_FEATURE_DESCRIPTIONS = """**ELA Feature Descriptions:**
- **`ela_meta.lin_simple.adj_r2`**: Adjusted R^2 of a linear model. High values suggest linearity.
- **`ela_meta.lin_w_interact.adj_r2`**: Adjusted R^2 of a linear model with pairwise interactions.
- **`ela_meta.quad_simple.adj_r2`**: Adjusted R^2 of a quadratic model without interactions.
- **`ela_meta.quad_w_interact.adj_r2`**: Adjusted R^2 of a full quadratic model.
- **`ela_distr.skewness`**: Skewness of the objective value distribution.
- **`nbc.nb_fitness.cor`**: Correlation between fitness and nearest-better connectivity.
- **`nbc.nn_nb.sd_ratio`**: Ratio of standard deviations (nearest neighbor distance / nearest-better distance).
- **`fitness_distance.fitness_std`**: Standard deviation of objective values."""


I1_PROMPT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design.
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below.

**Target Normalized ELA Features:**
(These are the values the generated function's landscape should ideally exhibit)
{ela_features}

{ela_feature_descriptions}

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
7.  **The function must be deterministic**: Do not use np.random or any stochastic elements.
"""

E1_PROMPT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design.
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below.

**Target Normalized ELA Features:**
(These are the values the generated function's landscape should ideally exhibit)
{ela_features}

{ela_feature_descriptions}

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
7.  **The function must be deterministic**: Do not use np.random or any stochastic elements.

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

{ela_feature_descriptions}

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
7.  **The function must be deterministic**: Do not use np.random or any stochastic elements.

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

{ela_feature_descriptions}

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
7.  **The function must be deterministic**: Do not use np.random or any stochastic elements.

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

{ela_feature_descriptions}

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
7.  **The function must be deterministic**: Do not use np.random or any stochastic elements.

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

{ela_feature_descriptions}

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
7.  **The function must be deterministic**: Do not use np.random or any stochastic elements.

**GENERATED FUNCTION:**
You already generated this function:
{context}

**INSTRUCTIONS:**
First, you need to identify the main components in the function above.
Next, analyze whether any of these components can be overfit to the specific sample of points used to calculate ELA features.
Then, based on your analysis, simplify the components to enhance the generalization to other samples.
"""
