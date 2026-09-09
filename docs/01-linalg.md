# 01 — Linear Algebra

## Power Iteration

Iterates

$$
v_{k+1} = \frac{Av_k}{\lVert Av_k \rVert}.
$$

The eigenvalue is estimated by the Rayleigh quotient

$$
\lambda = \frac{v^\top A v}{v^\top v},
$$

which, for normalized vectors, reduces to $v^\top A v$.

### Why does it converge, and at what speed?

The method converges because the initial vector can be written as a linear combination of the matrix's eigenvectors. If $A$ has eigenvectors $v_1, v_2, \ldots, v_n$, then

$$
v_0 = c_1v_1 + c_2v_2 + \cdots + c_nv_n.
$$

Multiplying successively by $A$, each component is scaled by its respective eigenvalue:

$$
A^k v_0 = c_1\lambda_1^k v_1 + c_2\lambda_2^k v_2 + \cdots + c_n\lambda_n^k v_n.
$$

If $|\lambda_1| > |\lambda_2| \geq \cdots$, the component associated with the largest eigenvalue grows relatively faster than the others. After each normalization, the smaller components become progressively less significant, causing the iterated vector to approach the dominant eigenvector.

The speed of this convergence is geometric and depends on the ratio

$$
\left|\frac{\lambda_2}{\lambda_1}\right|.
$$

The smaller this ratio, the faster the error shrinks at each iteration. This explains the concept of a **spectral gap**: when there's a large difference between the largest and second-largest eigenvalue in magnitude, convergence is much faster. This behavior is exactly what's observed in the `test_converges_faster_with_larger_spectral_gap` test, where matrices with greater separation between eigenvalues converge in fewer iterations.

### Why is the Rayleigh quotient quadratically accurate?

Numerical observation (from the prototype): with a symmetric $6 \times 6$ $A$, the **eigenvalue** error was around $10^{-11}$ while the residual $\lVert Av - \lambda v \rVert$ (governed by the **eigenvector** error) was around $10^{-5}$.

Fact: the eigenvalue converges at $O(\varepsilon^2)$, while the eigenvector converges at $O(\varepsilon)$.

The Rayleigh quotient has an important property: the error in the eigenvalue estimate is second order relative to the eigenvector error.

If the approximate vector is written as

$$
v = v^* + \varepsilon u,
$$

where $v^*$ is the true eigenvector and $\varepsilon$ represents a small error, the first-order correction of the Rayleigh quotient vanishes. This happens because the true eigenvector is a stationary point of the Rayleigh quotient, so only terms proportional to $\varepsilon^2$ remain.

**The proof**, to close the argument without leaving it as a loose claim:

Let $v = v^* + \varepsilon u$, with $u \perp v^*$ and $\lVert v^* \rVert = 1$. Then

$$
R(v) = \frac{v^\top A v}{v^\top v}
= \frac{\lambda^* + 2\varepsilon u^\top A v^* + \varepsilon^2 u^\top A u}
       {1 + \varepsilon^2 \lVert u \rVert^2}.
$$

Since $v^*$ is an eigenvector, $Av^* = \lambda^* v^*$, and since $u \perp v^*$, the cross term vanishes exactly:

$$
2\varepsilon u^\top A v^* = 2\varepsilon \lambda^* (u^\top v^*) = 0.
$$

It's not an approximation — it's an identity that holds for any $\varepsilon$.
What's left is just

$$
R(v) = \lambda^* + \varepsilon^2\left(u^\top A u - \lambda^* \lVert u \rVert^2\right) + O(\varepsilon^4).
$$

The first-order term in $\varepsilon$ never existed: it vanishes because
$v^*$ is a stationary point of the Rayleigh quotient, not by numerical
coincidence. It's this identity that guarantees, from first principles, that an
$O(\varepsilon)$ error in the eigenvector produces an $O(\varepsilon^2)$ error in the eigenvalue.

As a consequence, while the eigenvector error is of order

$$
O(\varepsilon),
$$

the eigenvalue error is of order

$$
O(\varepsilon^2).
$$

This behavior explains the experimental observation: even with a residual around $10^{-5}$, the eigenvalue error already reached around $10^{-11}$. In practice, this means a reasonably good approximation of the eigenvector already produces an extremely precise estimate of the eigenvalue. This is one of the main reasons the Rayleigh quotient is widely used in eigenvalue computation algorithms.

---

## Hotelling Deflation

After extracting $(\lambda_1, v_1)$, one constructs

$$
A' = A - \lambda_1 v_1v_1^\top.
$$

Then the power method is applied again.

### Why does it work?

The matrix $v_1v_1^\top$ represents the projection onto the direction of the first eigenvector. Multiplying it by $\lambda_1$ and subtracting that term from $A$ exactly eliminates the contribution corresponding to the dominant eigenvalue.

Indeed,

$$
\begin{aligned}
A'v_1
&= (A-\lambda_1v_1v_1^\top)v_1 \\
&= \lambda_1v_1-\lambda_1v_1 \\
&= 0.
\end{aligned}
$$

So the first eigenvalue becomes zero.

For any other eigenvector $v_i$, since the matrix is symmetric, its eigenvectors are orthogonal to each other. Therefore,

$$
v_1^\top v_i = 0.
$$

Hence,

$$
\begin{aligned}
A'v_i
&= Av_i-\lambda_1v_1(v_1^\top v_i) \\
&= \lambda_i v_i.
\end{aligned}
$$

In other words, the remaining eigenvalues and eigenvectors stay unchanged. This way, the next-largest eigenvalue becomes dominant, allowing its extraction by the next run of the power method.

### The honest trade-off: why you would NOT use this in practice

Numerical observation: in the prototype, the eigenvalues matched NumPy up to $10^{-14}$, but the orthogonality of the recovered eigenvectors dropped to around $10^{-7}$ — the last vectors are noticeably less accurate than the first ones.

Although Hotelling deflation works very well for small matrices, it has an important problem: numerical errors accumulate at each step.

In practice, the first eigenvector is never computed exactly. So the deflation matrix is built using an approximation of $v_1$. As a consequence, the first eigenvalue's contribution isn't removed perfectly, introducing small errors into the new matrix. These errors go on to influence the computation of the second eigenvector, which in turn generates a new, also imperfect, deflation. This process continues successively, accumulating errors over the iterations.

This phenomenon explains why, in the prototype, the eigenvalues remained extremely precise (error close to $10^{-14}$), while the orthogonality of the recovered eigenvectors dropped to around $10^{-7}$. The last eigenvectors end up significantly less precise than the first ones.

In small matrices, like a $6 \times 6$ matrix, this effect is practically irrelevant. However, in real problems involving thousands of dimensions, the accumulation of errors can seriously compromise the quality of the eigenvectors obtained.

For this reason, sequential deflation is rarely used in large-scale applications. In practice, the QR algorithm is used, which performs orthogonal similarity transformations while simultaneously preserving the matrix's entire spectral structure. Since these transformations maintain orthogonality in a much more numerically stable way, the QR algorithm avoids the progressive accumulation of errors seen in Hotelling deflation and produces all eigenvalues and eigenvectors with high precision.

---

## QR Decomposition: Gram-Schmidt vs. Householder

Every matrix $A \in \mathbb{R}^{m\times n}$ ($m \geq n$, linearly
independent columns) decomposes as $A = QR$, with $Q$ having orthonormal columns and
$R$ upper triangular. The two methods below reach the same theoretical result via
numerically very different paths.

### Classic Gram-Schmidt

Builds $Q$ column by column: the $j$-th column of $A$ has its
projection onto all previous columns of $Q$ removed, and the result is
normalized.

$$
v_j = a_j - \sum_{i < j} (q_i^\top a_j) q_i, \qquad q_j = \frac{v_j}{\lVert v_j \rVert}
$$

The problem isn't the formula — it's floating-point arithmetic. When two
columns of $A$ are nearly parallel, $v_j$ is a **difference between two
nearly equal quantities** ($a_j$ and its projection). This kind of subtraction is the
classic case of *catastrophic cancellation*: the significant digits that
remain after the subtraction come mostly from the rounding error of
each term, not from the real signal. The error from one projection contaminates the next
column, which contaminates the next one — and $Q$'s orthogonality degrades
cumulatively and silently, without the factorization ever looking "broken": $QR$ still
reconstructs $A$ to machine precision, only $Q$ stops being truly orthogonal.

The test fixture makes this concrete: with three nearly parallel columns
(a difference of $10^{-7}$ between them), $Q$'s orthogonality error jumps from
machine precision to $\approx 1.9\times10^{-2}$ — four orders of
magnitude of degradation, and the test matrix isn't even pathologically extreme.

### Householder Reflections

Instead of projecting and subtracting, each step applies an **orthogonal reflection**
$H = I - 2vv^\top$ (with $\lVert v \rVert = 1$) chosen to zero out everything
below the diagonal in the current column.

The structural difference is what matters: a Householder reflection is an
**exact isometry by construction** — it preserves the norm and angle between any
vectors, not as the result of a successful calculation, but because
$H^\top H = I$ is an algebraic identity, true at every step
regardless of how ill-conditioned the input matrix is. There's no
subtraction of nearly equal quantities hidden in the process — catastrophic
cancellation simply has nowhere to happen.

The same fixture proves this: orthogonality error $\approx 1.4\times10^{-15}$,
machine precision, on the same matrix where classic Gram-Schmidt failed.

### The pattern that already showed up before

This is the same trade-off structure as Hotelling deflation, and it's worth
naming the general pattern: **methods that operate via successive differences (GS,
deflation) accumulate rounding error at each step; methods that operate
via exactly orthogonal transformations at each step (Householder, and the
QR algorithm that follows) don't accumulate error, because each step is an
isometry by definition, not by numerical luck.**

This isn't a coincidence between two examples — it's the numerical stability
criterion that separates "textbook method" from "production method" in almost all
numerical linear algebra: prefer orthogonal transformations over projections
whenever stability matters more than the simplicity of the formula.

### Practical consequence

This is also why the QR algorithm for eigenvalues (next cycle)
replaces Hotelling deflation: it uses exactly these Householder reflections
to reduce the matrix and then iterates $A_{k+1} = R_k Q_k$
(factorization and product in swapped order) without ever accumulating the error that
sequential deflation carries.

---

## The QR Algorithm: closing the module

At each step, factors $A_k = Q_k R_k$ and recomposes in swapped order:
$A_{k+1} = R_k Q_k$.

### The identity that makes it all work

$$
A_{k+1} = R_k Q_k = Q_k^\top (Q_k R_k) Q_k = Q_k^\top A_k Q_k.
$$

Each step is an **orthogonal similarity transform**. This matters
because similarity preserves the spectrum exactly: $A_k$ and $A_{k+1}$ have the
same eigenvalues, always, for any $k$. The algorithm doesn't "compute" the
eigenvalues — it just changes the basis in which the matrix is represented, until the
chosen basis is one in which the matrix is already diagonal. In that
basis, the eigenvalues are, by definition, on the diagonal.

### The trade-off tying the module's three cycles together

The QR algorithm solves exactly the problem Hotelling deflation
had: since it uses Householder at every factorization, there's no
subtraction of nearly equal quantities, no catastrophic cancellation, and the
orthogonality of the recovered eigenvectors doesn't degrade — the test
`test_no_orthogonality_degradation_across_deflation_like_use` proves this to
$10^{-8}$ even after reducing 8 dimensions.

But it inherits, without disguise, power iteration's speed problem.
Structurally, the QR algorithm is a **simultaneous subspace iteration** —
instead of chasing a single dominant eigenvector, it chases an entire
subspace at once, but the convergence mechanism is the same: geometric,
at the ratio $|\lambda_{k+1}/\lambda_k|$ between consecutive eigenvalues. The
experiment in the prototype made this concrete: a wide gap ($\lambda_1{=}10$,
$\lambda_2{=}1$) converged in 32 iterations; the same matrix size with a
narrow gap ($\lambda_1{=}10$, $\lambda_2{=}9.999$) didn't even converge in 3000.

There's no free lunch here: you traded "imprecise near the end of the
spectrum" for "slow near close eigenvalues". None of the three methods in
this module — power iteration, deflation, unshifted QR — escapes one of the
two problems.

### What's left out, for honesty's sake

The production fix for the slowness is the **Wilkinson shift**: subtracting
an estimate of the closest eigenvalue from $A_k$ before each factorization,
which speeds up convergence from linear to cubic — a few iterations suffice
even with narrow gaps. Implementing it is out of this module's scope, but it's
worth noting that the problem has a known solution, and what its underlying principle is.

### Closing Module 1's cycle

Three methods, one goal (eigenvalues), three different trade-offs:
power iteration is simple, but only gives the dominant eigenvalue; deflation extends
it to the whole spectrum, but accumulates error; QR fixes the error, but inherits the
slowness. It's exactly this kind of map — not "which method is best", but
"which pain each method trades for which other" — that separates judging
from first principles from memorizing which function to call.

---

# PCA Applied — Closing the Cycle with HVI Data

This section applies the module's `qr_algorithm` to a real problem: reducing the
eight HVI parameters to a handful of components that capture most of
the variance. This is where the abstract math of the three previous cycles
meets Module 0's synthetic generator.

## PCA via covariance

$$
\Sigma = \frac{1}{n-1} X_c^\top X_c, \qquad X_c = X - \bar{X}
$$

The principal components are the eigenvectors of $\Sigma$, ordered by the
corresponding eigenvalue (the variance explained by that direction). It's the
most direct route: explicitly forms the covariance matrix and decomposes
it with the `qr_algorithm` already validated in the previous cycle.

## The gotcha: PCA is sensitive to feature scale

Running `pca_via_covariance` without standardizing on the real HVI data (5000
bales, seed 2024), the first principal component came out almost purely as a
single feature:

| Feature         | Loading on PC1 (raw) |
|-----------------|------------------------|
| `rd`            | **0.985**              |
| `plus_b`        | -0.161                 |
| all others | < 0.05                 |

The cause isn't correlation — it's scale. The raw variance of each feature:

| Feature      | Raw variance |
|--------------|-----------------|
| `rd`         | 9.07            |
| `strength`   | 6.12            |
| `uniformity` | 2.22            |
| `plus_b`     | 1.00            |
| `uhml`       | 1.45            |
| `elongation` | 0.63            |
| `micronaire` | 0.15            |
| `trash`      | 0.09            |

`rd` has the largest standard deviation (3.0, reflectance measured on a 0–100 scale)
simply because that variable's unit of measurement produces larger
numbers — not because it's more informative than the others. The covariance
matrix mixes scale with correlation, and when the two compete,
**scale wins**. PCA on raw covariance isn't "wrong PCA" — it's PCA
answering exactly the question that was asked: "which direction has the most
variance in original units", which is rarely the question you actually wanted
to ask.

## The fix: standardize before decomposing

Dividing each feature by its standard deviation before forming the matrix
(`standardize=True`) decomposes the **correlation** matrix, not the
covariance one. Every feature ends up contributing on equal footing — the
diagonal of the correlation matrix is always 1, so the trace is always $p$ (the
number of features), and that's exactly why
`test_standardized_explained_variance_sums_to_number_of_features` works
as a pure invariant, with no oracle needed: the sum of the eigenvalues of
any 8-feature correlation matrix is 8, always, by construction.

With standardization, PC1 stops being hostage to `rd` and becomes dominated
by `uhml` — and looking at the full *loadings* (not just the dominant one), the
real structure appears: PC1 concentrates `uhml`, `uniformity`, and `strength` with
consistent signs — the **fiber quality** block that Module 0's generator
deliberately planted as correlated. PC2 concentrates `rd`, `plus_b`, and
`trash` — the **color quality** block. The separation the dev.to article
for `cotton-desk-tasks` already sensed in the domain comes here *for free* from linear
algebra, without any class label or prior hypothesis — it's literally
what "unsupervised learning" means.

## Why this matters beyond this module

This isn't a PCA quirk. **Any method based on distance
or inner products** — k-means, KNN, L2 regularization, and gradient
descent itself when features have very different scales (Module 4) — suffers
from the same sensitivity. That's why `StandardScaler` shows up in
practically every ML pipeline as a first step, not out of habit, but
because the math behind it — covariance, Euclidean distance, gradient
norm — treats "numerically large" as synonymous with "important",
unless you normalize first.

## Validation

`pca_via_covariance` was validated against `numpy.linalg.svd` on the raw
data: explained variance matches to $10^{-14}$, the components satisfy
$\Sigma v = \lambda v$ to $10^{-12}$, and the components' orthonormality
sits at $10^{-14}$ — the same precision `qr_algorithm` already delivered in
the previous cycle, as expected, since PCA here is nothing more than a
direct application of it.

---

## One-Sided Jacobi SVD

Instead of forming $X^\top X$ and decomposing that $p \times p$ matrix, this method
operates directly on the columns of $X$ ($n \times p$). At each step,
it picks two columns $(a_i, a_j)$ and applies a $2\times2$ rotation that zeroes
$\langle a_i, a_j\rangle$ **exactly**, by geometric construction — not
by a numerical difference between close quantities. Repeating over all
pairs (a "sweep") and repeating sweeps until convergence, the columns
become mutually orthogonal: the norm of each one is the singular value, its
direction is the column of $U$, and the accumulated rotation is $V$.

The point that sets this method apart from the previous one: each inner product
$\langle a_i, a_j \rangle$ is recomputed at each sweep, from the
already partially refined columns — at no point are all $p(p+1)/2$
inner products of $X^\top X$ computed at once, before any
refinement happens.

## Why this matters: the squared condition number

Condition number of a matrix: $\kappa(X) = \sigma_{\max}(X) /
\sigma_{\min}(X)$ — how much $X$ "stretches" space in the worst
direction compared to the best. It measures how much input error is amplified in the
output of any computation with $X$.

The eigenvalues of $X^\top X$ are $\sigma_i(X)^2$ — that's the very
definition of a singular value. Hence:

$$
\kappa(X^\top X) = \frac{\sigma_{\max}(X)^2}{\sigma_{\min}(X)^2} = \kappa(X)^2
$$

Forming $X^\top X$ explicitly raises the condition number to the
**square**, even before any decomposition algorithm comes into
play. This isn't a quirk of `qr_algorithm` — it would happen with
any eigensolver, no matter how good, because the problem is already in
the input matrix, not in the method that decomposes it.

The floating-point consequence: double precision carries
$\varepsilon \approx 2.2\times10^{-16}$ of relative precision per operation
(about 16 decimal digits). The expected error when solving an
eigenvalue problem with direct methods scales with $\kappa \cdot \varepsilon$. If
$\kappa(X) \approx 2\times10^{8}$ (the test fixture), then
$\kappa(X^\top X) \approx 4\times10^{16}$ — **larger than $1/\varepsilon$**.
This literally means there are no significant digits left to
represent the smallest eigenvalue: it sits below the very rounding
noise of the matrix that contains it.

## The proof, with measured numbers

On the same ill-conditioned matrix ($\kappa(X) \approx 2\times10^8$, three
nearly duplicated columns — the kind of collinearity that really appears
between correlated HVI features, like `uhml` and `uniformity`,
amplified here to make the effect visible):

| Route                                          | Relative error on the smallest component         |
|-----------------------------------------------|-------------------------------------------|
| via covariance ($X^\top X$ + `qr_algorithm`) | **325%** — signal lost                  |
| via SVD (Jacobi, direct on $X$)               | $1.2\times10^{-10}$ — machine precision |

The `test_covariance_route_loses_smallest_component_on_ill_conditioned_data`
test proves that the covariance route **gets it wrong on purpose** in this regime — it's not
a bug to fix, it's the math of $\kappa^2$ manifesting. And
`test_agrees_with_covariance_route_when_well_conditioned` proves the other
side: when $\kappa(X)$ is reasonable (as in the generator's real HVI data),
the two routes agree to $10^{-6}$ — the difference only shows up, and only
matters, near the precision limit.

## An important distinction from the previous cycle's trade-off

This isn't the same argument as Gram-Schmidt vs. Householder. There, the
problem was catastrophic cancellation — subtracting nearly equal quantities
amplifying rounding error at each step. Here, the problem happens
**before any subtraction**: it's the very formation of
$X^\top X$ that already squares $\kappa$, all at once, in the first
line of code. These are two different numerical diseases with the same
symptom — loss of precision — and it's worth not confusing the two when
diagnosing a real problem: "my result is imprecise" could be
catastrophic cancellation OR a squared condition number, and the remedy is
different in each case.

## Practical consequence, outside the lab

This isn't academic curiosity: it's why `sklearn.decomposition.PCA`
uses SVD internally by default, not eigendecomposition of the covariance
matrix — even though they're mathematically equivalent on paper. Any
real dataset with correlated features (and HVI has them — length and
uniformity move together by construction) pushes $\kappa$ up, and the
"obvious" route (covariance) is exactly the one that degrades first.

## Closing Module 1's capstone

Three eigenvalue methods (power iteration, deflation, QR), two QR
decomposition methods (Gram-Schmidt, Householder), two PCA methods
(covariance, SVD) — six implementations, three trade-offs, one common pattern
emerging in all of them: **the "obvious" way to solve a numerical problem
is rarely the stable one**, and the difference only shows up when you know
to look for it — in narrow spectral gaps, in nearly collinear columns, in high
condition numbers. This is exactly the vigilance Phase 1 was meant to
train.
