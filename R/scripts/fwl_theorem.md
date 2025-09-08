# Mathematical Proof
## The Frisch-Waugh-Lovell Theorem

---

## Mathematical Foundation

### Problem Setup

Consider the linear regression model:

$$\boxed{y = X_1\beta_1 + X_2\beta_2 + u}$$

where the components are defined as follows:

- $y \in \mathbb{R}^{n \times 1}$ is the vector of outcomes
- $X_1 \in \mathbb{R}^{n \times k_1}$ is the matrix of regressors of interest with $\text{rank}(X_1) = k_1$
- $X_2 \in \mathbb{R}^{n \times k_2}$ is the matrix of control variables with $\text{rank}(X_2) = k_2$
- $\beta_1 \in \mathbb{R}^{k_1 \times 1}$ is the parameter vector of primary interest
- $\beta_2 \in \mathbb{R}^{k_2 \times 1}$ is the parameter vector for control variables
- $u \in \mathbb{R}^{n \times 1}$ is the error vector with $\mathbb{E}[u|X_1,X_2] = 0$

### Definition: Projection and Annihilator Matrices

For a full-rank matrix $X_2 \in \mathbb{R}^{n \times k_2}$, we define:

$$P_{X_2} = X_2(X_2'X_2)^{-1}X_2' \quad \text{(Projection matrix)}$$

$$M_{X_2} = I_n - P_{X_2} = I_n - X_2(X_2'X_2)^{-1}X_2' \quad \text{(Annihilator matrix)}$$

These matrices satisfy the following fundamental properties:

(i) $P_{X_2}$ and $M_{X_2}$ are symmetric: $P_{X_2}' = P_{X_2}$, $M_{X_2}' = M_{X_2}$

(ii) $P_{X_2}$ and $M_{X_2}$ are idempotent: $P_{X_2}^2 = P_{X_2}$, $M_{X_2}^2 = M_{X_2}$

(iii) $M_{X_2}X_2 = 0$ and $P_{X_2}X_2 = X_2$

(iv) $P_{X_2} + M_{X_2} = I_n$

---

## Main Result

### The Frisch-Waugh-Lovell Theorem

**Theorem (Frisch-Waugh-Lovell):** The OLS estimate of $\beta_1$ in the full regression of $y$ on $[X_1 \quad X_2]$ is identical to the OLS estimate obtained from the following two-step partialling-out procedure:

**Step 1:** Regress $y$ on $X_2$ and obtain residuals: $\tilde{y} = M_{X_2}y$

**Step 2:** Regress $X_1$ on $X_2$ and obtain residuals: $\tilde{X_1} = M_{X_2}X_1$

**Step 3:** Regress $\tilde{y}$ on $\tilde{X_1}$ to obtain: $\hat{\beta_1}^{\text{FWL}} = (\tilde{X_1}'\tilde{X_1})^{-1}\tilde{X_1}'\tilde{y}$

**Formal Statement:**
$$\boxed{\hat{\beta_1} = \hat{\beta_1}^{\text{FWL}} = (\tilde{X_1}'\tilde{X_1})^{-1}\tilde{X_1}'\tilde{y}}$$

---

## Proof

We establish the equivalence by demonstrating that both approaches yield identical coefficient estimates through rigorous matrix algebra.

### Part I: Full Regression Setup

The full regression model in partitioned form is:

$$y = [X_1 \quad X_2]\begin{bmatrix} \beta_1 \\ \beta_2 \end{bmatrix} + u = X\beta + u$$

where $X = [X_1 \quad X_2] \in \mathbb{R}^{n \times (k_1+k_2)}$ and $\beta = [\beta_1' \quad \beta_2']' \in \mathbb{R}^{(k_1+k_2) \times 1}$.

The OLS estimator is given by:

$$\hat{\beta} = (X'X)^{-1}X'y = \begin{bmatrix} \hat{\beta_1} \\ \hat{\beta_2} \end{bmatrix}$$

### Part II: Matrix Partitioning

We partition the cross-product matrices:

$$X'X = \begin{bmatrix} X_1'X_1 & X_1'X_2 \\ X_2'X_1 & X_2'X_2 \end{bmatrix} = \begin{bmatrix} A & B \\ B' & D \end{bmatrix}$$

$$X'y = \begin{bmatrix} X_1'y \\ X_2'y \end{bmatrix}$$

where:

$$A = X_1'X_1 \in \mathbb{R}^{k_1 \times k_1}, \quad B = X_1'X_2 \in \mathbb{R}^{k_1 \times k_2}, \quad D = X_2'X_2 \in \mathbb{R}^{k_2 \times k_2}$$

### Part III: Partitioned Matrix Inverse

Using the block matrix inversion formula:

$$(X'X)^{-1} = \begin{bmatrix} 
(A - BD^{-1}B')^{-1} & -(A - BD^{-1}B')^{-1}BD^{-1} \\
-D^{-1}B'(A - BD^{-1}B')^{-1} & D^{-1} + D^{-1}B'(A - BD^{-1}B')^{-1}BD^{-1}
\end{bmatrix}$$

### Part IV: Key Algebraic Identity

We establish the fundamental relationship:

$$A - BD^{-1}B' = X_1'X_1 - X_1'X_2(X_2'X_2)^{-1}X_2'X_1$$

$$= X_1'(I_n - X_2(X_2'X_2)^{-1}X_2')X_1$$

$$= X_1'M_{X_2}X_1$$

### Part V: Extracting $\hat{\beta_1}$

From the normal equations $(X'X)\hat{\beta} = X'y$, the first block gives us:

$$\hat{\beta_1} = (A - BD^{-1}B')^{-1}(X_1'y - BD^{-1}X_2'y)$$

$$= (X_1'M_{X_2}X_1)^{-1}(X_1'y - X_1'X_2(X_2'X_2)^{-1}X_2'y)$$

$$= (X_1'M_{X_2}X_1)^{-1}X_1'(I_n - X_2(X_2'X_2)^{-1}X_2')y$$

$$= (X_1'M_{X_2}X_1)^{-1}X_1'M_{X_2}y$$

### Part VI: Two-Step Procedure Analysis

The partialling-out procedure yields:

$$\tilde{y} = M_{X_2}y \quad \text{(Step 1)}$$

$$\tilde{X_1} = M_{X_2}X_1 \quad \text{(Step 2)}$$

$$\hat{\beta_1}^{\text{FWL}} = (\tilde{X_1}'\tilde{X_1})^{-1}\tilde{X_1}'\tilde{y} \quad \text{(Step 3)}$$

### Part VII: Establishing Equivalence

Substituting the definitions from Steps 1 and 2:

$$\hat{\beta_1}^{\text{FWL}} = ((M_{X_2}X_1)'(M_{X_2}X_1))^{-1}(M_{X_2}X_1)'(M_{X_2}y)$$

$$= (X_1'M_{X_2}'M_{X_2}X_1)^{-1}X_1'M_{X_2}'M_{X_2}y$$

### Part VIII: Applying Matrix Properties

Using the symmetry and idempotency of $M_{X_2}$ from the definition:

$$M_{X_2}' = M_{X_2} \quad \text{(symmetry)}$$

$$M_{X_2}M_{X_2} = M_{X_2} \quad \text{(idempotency)}$$

Therefore:

$$\hat{\beta_1}^{\text{FWL}} = (X_1'M_{X_2}M_{X_2}X_1)^{-1}X_1'M_{X_2}M_{X_2}y$$

$$= (X_1'M_{X_2}X_1)^{-1}X_1'M_{X_2}y$$

### Part IX: Final Equivalence

Comparing the expressions from Parts V and VIII:

$$\hat{\beta_1} = (X_1'M_{X_2}X_1)^{-1}X_1'M_{X_2}y = \hat{\beta_1}^{\text{FWL}}$$

This establishes the desired result:

$$\boxed{\hat{\beta_1} = (\tilde{X_1}'\tilde{X_1})^{-1}\tilde{X_1}'\tilde{y}} \quad \blacksquare$$

---

### Summary

The Frisch-Waugh-Lovell theorem demonstrates that the coefficient estimate for the variables of interest $X_1$ can be obtained equivalently through:

1. **Direct estimation**: Running the full regression of $y$ on $[X_1 \quad X_2]$
2. **Partialling-out**: Removing the effect of control variables $X_2$ from both $y$ and $X_1$, then regressing the residuals

This result is fundamental in econometrics and provides both computational advantages and theoretical insights into the nature of multiple regression.