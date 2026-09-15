import numpy as np
from scipy.optimize import nnls

# ============================================================
# 1. Compute A
# ============================================================

def compute_A(X, Z):
    """
    Compute the coefficients A that represent each observation
    as a convex combination of the archetypes.

    Parameters
    ----------
    X : ndarray, shape (n, m)
        Data matrix.

    Z : ndarray, shape (p, m)
        Archetypes.

    Returns
    -------
    A : ndarray, shape (n, p)
        Convex combination coefficients.
    """

    n = X.shape[0]
    p = Z.shape[0]

    A = np.zeros((n, p))

    for i in range(n):
        A[i], _ = convex_least_squares(X[i], Z)

    return A

# ============================================================
# 2. Convex Least Squares
# ============================================================

def convex_least_squares(x, Z, M=1000):
    """
    Solve the convex least-squares problem:

        min ||x - Z.T @ a||^2

    subject to:

        a >= 0
        sum(a) = 1

    Parameters
    ----------
    x : ndarray, shape (m,)
        Observation.

    Z : ndarray, shape (p, m)
        Archetypes.

    M : float
        Penalty used to enforce sum(a) = 1.

    Returns
    -------
    a : ndarray, shape (p,)
        Convex coefficients.

    residual : float
        NNLS residual.
    """

    # Z has shape (p, m)
    # We need a matrix with shape (m, p)
    T = Z.T

    # Artificial observation to enforce sum(a) = 1
    T_augmented = np.vstack([
        T,
        M * np.ones((1, Z.shape[0]))
    ])

    x_augmented = np.append(x, M)

    # Non-negative least squares
    a, residual = nnls(T_augmented, x_augmented)

    return a, residual

# ============================================================
# 3. Compute Z
# ============================================================

def compute_Z(beta, X):
    """
    Compute the archetypes from beta and X.

        Z = beta @ X
    """

    return beta @ X

# ============================================================
# 4. Residuals for updating one archetype
# ============================================================

def compute_residuals_for_archetype(X, A, Z, k):
    """
    Compute the residuals used to update archetype k.

        r_i = x_i - sum_{l != k} a_il z_l
    """

    n = X.shape[0]
    residuals = np.zeros_like(X)

    for i in range(n):

        contribution = np.zeros(X.shape[1])

        for l in range(Z.shape[0]):

            if l != k:
                contribution += A[i, l] * Z[l]

        residuals[i] = X[i] - contribution

    return residuals

# ============================================================
# 5. Compute beta for one archetype
# ============================================================

def compute_beta(X, A, Z, k, M=100):
    n, m = X.shape

    numerator = np.zeros(m)
    denominator = 0.0

    for i in range(n):
        if A[i, k] > 1e-12:
            contribution = np.zeros(m)

            for l in range(Z.shape[0]):
                if l != k:
                    contribution += A[i, l] * Z[l]

            v_i = (X[i] - contribution) / A[i, k]

            numerator += (A[i, k] ** 2) * v_i
            denominator += A[i, k] ** 2

    if denominator == 0:
        return np.zeros(n)

    v = numerator / denominator

    T = X.T
    T_augmented = np.vstack([T, M * np.ones((1, n))])
    v_augmented = np.append(v, M)

    beta_k, _ = nnls(T_augmented, v_augmented)

    return beta_k

# ============================================================
# 6. Compute RSS
# ============================================================

def compute_rss(X, A, Z):
    """
    Compute the reconstruction error:

        RSS = sum ||X - A @ Z||^2
    """

    X_reconstructed = A @ Z

    residuals = X - X_reconstructed

    rss = np.sum(residuals ** 2)

    return rss

# ============================================================
# 7. One alternating iteration
# ============================================================

def one_iteration(X, Z, tol=1e-6, max_inner_iter=100):
    A = compute_A(X, Z)
    p = Z.shape[0]
    beta = np.zeros((p, X.shape[0]))

    Z_new = Z.copy()

    previous_rss = np.inf

    for inner_iteration in range(max_inner_iter):
        for k in range(p):
            beta[k] = compute_beta(X, A, Z_new, k)
            Z_new[k] = compute_Z(beta[k:k+1], X)[0]

        rss = compute_rss(X, A, Z_new)

        improvement = previous_rss - rss

        if improvement < tol:
            break

        previous_rss = rss

    A_new = compute_A(X, Z_new)
    rss = compute_rss(X, A_new, Z_new)

    return A_new, beta, Z_new, rss


# ============================================================
# 8. Complete Archetypal Analysis
# ============================================================

def archetypal_analysis(
    X,
    Z_initial,
    max_iter=100,
    tol=1e-6
):
    """
    Run the alternating Archetypal Analysis algorithm.

    Parameters
    ----------
    X : ndarray
        Data matrix.

    Z_initial : ndarray
        Initial archetypes.

    max_iter : int
        Maximum number of iterations.

    tol : float
        Convergence tolerance.

    Returns
    -------
    A : ndarray
        Final coefficients.

    beta : ndarray
        Final archetype coefficients.

    Z : ndarray
        Final archetypes.

    rss : float
        Final reconstruction error.
    """

    # Copy the initial archetypes so that the original
    # array is not modified.
    Z = Z_initial.copy()

    previous_rss = np.inf

    for iteration in range(max_iter):

        # Perform one alternating iteration
        A, beta, Z_new, rss = one_iteration(
            X,
            Z, tol
        )

        # Improvement in RSS
        improvement = previous_rss - rss

        print(
            f"Iteration {iteration + 1}: "
            f"RSS = {rss:.6f}, "
            f"improvement = {improvement:.6f}"
        )

        # Check convergence
        if (
            improvement >= 0
            and improvement < tol
        ):
            Z = Z_new
            break

        # Prepare next iteration
        Z = Z_new

        previous_rss = rss

    return A, beta, Z, rss


# ============================================================
# 9. Example
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Small example dataset
    # --------------------------------------------------------

    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [2.0, 2.0]
    ])

    # --------------------------------------------------------
    # Initial archetypes
    # --------------------------------------------------------

    Z_initial = np.array([
        [0.5, 0.5],
        [1.5, 1.5]
    ])

    # --------------------------------------------------------
    # Run Archetypal Analysis
    # --------------------------------------------------------

    A, beta, Z, rss = archetypal_analysis(
        X,
        Z_initial,
        max_iter=100,
        tol=1e-6
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n=== Final result ===")

    print("\nA:")
    print(A)

    print("\nBeta:")
    print(beta)

    print("\nZ:")
    print(Z)

    print("\nRSS:")
    print(rss)