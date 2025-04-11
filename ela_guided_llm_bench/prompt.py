PROMPT = """
You are an expert in Exploratory Landscape Analysis (ELA), advanced optimization benchmarks, and high-dimensional function design. 
Your task is to generate a single, synthetic benchmark function in Python for testing global optimization algorithms.

The primary goal is to create a function whose ELA features closely match the target values provided below. 
You will be given a previous attempt, its ELA features, and the error between its features and the target. 
Your objective is to **significantly modify** the previous function to **reduce this error**, particularly for the features with the largest errors.

**Target ELA Features:**
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

**Context from Previous Attempt:**

* **Previously Generated Function:**
    ```python
    {example_source_code}
    ```
* **Its ELA Features:**
    {example_ela_features}
* **Error (Previous ELA - Target ELA):**
    {example_error}

**Guidance for Improvement based on Error:**

1.  **Analyze the Error:** Carefully examine the `example_error`. Identify the ELA features with the **largest absolute errors**. These are the features your modifications should primarily target.
2.  **Relate ELA to Function Structure:** Consider how different mathematical operations influence ELA features:
    * **Linearity (`lin_...`)**: Affected by the strength of linear terms vs. non-linear/interaction terms. To increase linearity, strengthen linear components or weaken others. To decrease, add stronger non-linearities (powers, sin/cos, exp) or interactions.
    * **Curvature (`quad_...`)**: Affected by quadratic terms, higher powers, and non-linear transformations. To increase `quad_simple`, add stronger `xᵢ²` terms. To increase `quad_w_interact`, add stronger `xᵢxⱼ` interaction terms or combine quadratic effects.
    * **Interactions (`..._w_interact`)**: Primarily affected by terms involving multiple variables (e.g., `xᵢxⱼ`, `sin(xᵢ + xⱼ)`). Add or strengthen these to increase interaction R² values.
    * **Distribution (`skewness`, `fitness_std`)**: Affected by overall function shape and transformations. Applying `exp()` can increase positive skew and standard deviation. `log()` might decrease them. Asymmetric terms (e.g., odd powers if the domain allows) can influence skewness. Scaling the function output directly affects `fitness_std`.
    * **Neighborhood (`nbc_...`)**: Affected by modality, smoothness, and local features. Adding high-frequency components (like `cos(k*xᵢ)`) can increase local optima and affect `nbc` metrics. Modifying interaction strengths can change the 'path' to better solutions. Increasing ruggedness might decrease `nbc.nb_fitness.cor` and affect `nbc.nn_nb.sd_ratio`.
3.  **Perform Significant Mutations:** Based on the error analysis, make **substantial changes** to the `example_source_code`. Don't just tweak coefficients slightly. Consider:
    * **Changing base mathematical forms:** Replace or combine terms (e.g., replace a quadratic part with an exponential or sinusoidal part).
    * **Adding/Removing Terms:** Introduce new types of terms (linear, quadratic, interactions, periodic, exponential) or remove existing ones that contribute negatively to the target ELA profile.
    * **Altering Interactions:** Change how variables interact (e.g., switch from pairwise `xᵢxⱼ` to sums inside non-linear functions like `sin(Σxᵢ)`).
    * **Applying Transformations:** Wrap parts of the function or the entire output in transformations (e.g., `np.exp(...)`, `np.log(...)`, `np.abs(...)`, power functions).
    * **Adjusting Coefficients Drastically:** Significantly change the weights of different components.
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
"""