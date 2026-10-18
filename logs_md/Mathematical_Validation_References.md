# Mathematical Proofs and References for Validation Mechanisms

This document outlines the mathematical formulas used to ensure logical consistency within the PaddyCare AI system's validation algorithms (`predict.py`), along with academic and mathematical references for the theorems applied.

---

## 1. Information Theory: Shannon Entropy for OOD Detection

### The Mathematical Formula
To measure the neural network's uncertainty when classifying an image, we calculate the Shannon Entropy $H(X)$ of the softmax probability distribution $P = \{p_1, p_2, ..., p_N\}$:

$$H(X) = - \sum_{i=1}^{N} p_i \ln(p_i)$$

For $N=4$ classes, the theoretical maximum entropy (complete uncertainty) occurs when all probabilities are equal ($p_i = 0.25$):
$$H_{max} = \ln(4) \approx 1.386 \text{ nats}$$

### Logical Consistency
By setting an `OOD_ENTROPY_THRESHOLD` mathematically bounded between $0$ and $1.386$, the system is guaranteed to catch evenly distributed (ambiguous) predictions that signify an Out-of-Distribution (OOD) image.

### References & Citations
1. **Foundational Theorem:** 
   * Shannon, C. E. (1948). *"A Mathematical Theory of Communication"*. Bell System Technical Journal, 27(3), 379-423.
   * *Relevance:* This paper introduced the formula for information entropy used to quantify uncertainty in a probability distribution.
2. **Machine Learning Application (OOD Detection):**
   * Hendrycks, D., & Gimpel, K. (2017). *"A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"*. Proceedings of the International Conference on Learning Representations (ICLR).
   * *Relevance:* This paper establishes the standard practice of using Softmax distribution statistics (like entropy and maximum probability) to detect out-of-distribution inputs that the neural network was not trained on.

---

## 2. Set Theory & Expected Values: Color-Space Validation Heuristic

### The Mathematical Formula
We use sets to represent valid hue ranges on a 0-255 scale:
* **Plant Hues ($S_p$):** $[10, 75]$
* **Strict Green Hues ($S_g$):** $[24, 75]$

Based on Set Theory, because $[24, 75]$ is entirely contained within $[10, 75]$, we establish the strict subset axiom:
$$S_g \subset S_p$$

Using indicator functions $I_g(x)$ and $I_p(x)$ for individual pixels, and taking the Expected Value (the ratio) across the image domain $D$, the subset axiom guarantees the inequality:
$$E[I_g] \le E[I_p]$$

### Logical Consistency
The system's validation checks if:
`E[I_p] >= 0.08 AND E[I_g] >= 0.02`

Because $0.02 \le 0.08$, the threshold constraints logically respect the mathematical inequality $E[I_g] \le E[I_p]$. This prevents programmatic contradictions (such as requiring more green pixels than total plant pixels) while allowing for diseased leaves (yellow/brown) as long as a minimum biological baseline of green is met.

### References & Citations
1. **Set Theory & Subset Axioms:**
   * Halmos, P. R. (1960). *"Naive Set Theory"*. Undergraduate Texts in Mathematics. Springer.
   * *Relevance:* Defines the fundamental properties of subsets ($A \subset B$) and the logical implications of containment, which our hue bounding boxes rely on.
2. **Probability & Expected Values (Indicator Variables):**
   * Ross, S. M. (2014). *"A First Course in Probability"*. Pearson. (Chapter on Expectation of Indicator Random Variables).
   * *Relevance:* Provides the proof that if event A implies event B (a pixel being green implies it is a plant hue), then the probability (or expected ratio in a finite image sample) of A must be less than or equal to the probability of B: $P(A) \le P(B)$.
3. **Computer Vision (HSV Color Space representation):**
   * Smith, A. R. (1978). *"Color Gamut Transform Pairs"*. SIGGRAPH '78 Proceedings.
   * *Relevance:* Defines the HSV (Hue, Saturation, Value) color space used by the PIL library to isolate chrominance (Hue) from luminance, allowing the mathematical formulas to operate independently of lighting conditions.
