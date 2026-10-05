import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ============================================================
# 1. Convex Least Squares
# ============================================================

def convex_least_squares(x, V):
    """
    Solve

        min ||x - V @ a||^2

    subject to

        a >= 0
        sum(a) = 1

    Parameters
    ----------
    x : ndarray, shape (d,)
        Vector to represent.
    V : ndarray, shape (d, q)
        Matrix whose q columns are the candidate archetypes.

    Returns
    -------
    a : ndarray, shape (q,)
        Convex-combination coefficients.
    rss : float
        Squared reconstruction error.
    """
    q = V.shape[1]

    def objective(a):
        residual = x - V @ a
        return np.dot(residual, residual)

    def jacobian(a):
        return -2.0 * V.T @ (x - V @ a)

    constraints = {
        "type": "eq",
        "fun": lambda a: np.sum(a) - 1.0
    }

    bounds = [(0.0, None)] * q
    a0 = np.full(q, 1.0 / q)

    result = minimize(
        objective,
        a0,
        method="SLSQP",
        jac=jacobian,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500}
    )

    # Numerical safeguard.
    a = np.maximum(result.x, 0.0)
    s = np.sum(a)

    if s > 0:
        a /= s
    else:
        a = a0

    return a, objective(a)


# ============================================================
# 2. Compute alpha
# ============================================================

def compute_alpha(X, Z, gamma):
    """
    Compute alpha.

    X     : (n, m)
    Z     : (k, c)
    gamma : (c, m)

    Since

        X ~= alpha @ Z @ gamma,

    each observation x_i is represented as a convex
    combination of the k rows of Z @ gamma.

    Returns
    -------
    alpha : (n, k)
    """
    n = X.shape[0]

    observation_archetypes = Z @ gamma   # (k, m) obtenemos los perfiles completos de las observaciones arquetípicas

    alpha = np.zeros((n, Z.shape[0]))

    for i in range(n): # resolvemos x_i = alfa_i(Z @ gamma) para cada observación
        alpha[i], _ = convex_least_squares(
            X[i],
            observation_archetypes.T
        )

    return alpha


# ============================================================
# 3. Compute gamma
# ============================================================

def compute_gamma(X, alpha, Z):
    """
    Compute gamma.

    The model is

        X ~= alpha @ Z @ gamma.

    Therefore, for every original variable j, X[:, j] is
    represented as a convex combination of the c columns of
    alpha @ Z.

    Returns
    -------
    gamma : (c, m)
    """
    variable_archetypes = alpha @ Z   # (n, c) cada columna de X tiene que ser representada como una combinación convexa de las c columnas de alpha @ Z (V)

    gamma = np.zeros((Z.shape[1], X.shape[1]))

    for j in range(X.shape[1]): # x._j = gamma._j (alpha @ Z) para cada variable
        gamma[:, j], _ = convex_least_squares(
            X[:, j],
            variable_archetypes
        )

    return gamma


# ============================================================
# 4. Compute Z from alpha, X and gamma
# ============================================================

def compute_Z_from_alpha(alpha, X, gamma):
    """
    Recover the biarchetype matrix Z from

        X ~= alpha @ Z @ gamma.

    Multiplying by the transpose of alpha and gamma gives the
    least-squares-type update

        Z = alpha.T @ X @ gamma.T.

    Returns
    -------
    Z : (k, c)
    """
    return alpha.T @ X @ gamma.T # .T = matriu transposada


# ============================================================
# 5. Compute beta
# ============================================================

def compute_beta(Z, X, theta):
    """
    Compute beta from

        Z = beta @ X @ theta.

    Each row of Z is represented as a convex combination of
    the n rows of X @ theta.

    Returns
    -------
    beta : (k, n)
    """
    observation_candidates = X @ theta   # (n, c) 

    beta = np.zeros((Z.shape[0], X.shape[0]))

    for g in range(Z.shape[0]): # para cada arquetipo de observación, resolvemos Z[g] = beta[g] (X @ theta) para obtener los coeficientes convexos beta[g]
        beta[g], _ = convex_least_squares(
            Z[g],
            observation_candidates.T
        )

    return beta


# ============================================================
# 6. Compute theta
# ============================================================

def compute_theta(Z, beta, X):
    """
    Compute theta from

        Z = beta @ X @ theta.

    Each column of Z is represented as a convex combination of
    the m columns of beta @ X.

    Returns
    -------
    theta : (c, m)
    """
    variable_candidates = beta @ X   # (k, m)

    theta = np.zeros((Z.shape[1], X.shape[1]))

    for h in range(Z.shape[1]): # obtenemos cada fila de theta
        theta[h], _ = convex_least_squares(
            Z[:, h],
            variable_candidates
        )

    return theta


# ============================================================
# 7. Compute RSS
# ============================================================

def compute_rss(X, alpha, Z, gamma):
    """
    Compute

        RSS = ||X - alpha @ Z @ gamma||^2.
    """
    X_reconstructed = alpha @ Z @ gamma
    residuals = X - X_reconstructed

    return np.sum(residuals ** 2)


# ============================================================
# 8. Biarchetypal Analysis
# ============================================================

def biarchetypal_analysis(
    X,
    k,
    c,
    max_iter=100,
    tol=1e-6,
    seed=42,
    verbose=True
):
    """
    Biarchetypal Analysis.

    Parameters
    ----------
    X : ndarray, shape (n, m)
        Data matrix.

    k : int
        Number of observation archetypes.

    c : int
        Number of variable archetypes.

    max_iter : int
        Maximum number of outer iterations.

    tol : float
        Convergence tolerance on the RSS improvement.

    seed : int
        Random seed.

    verbose : bool
        Print RSS at every iteration.

    Returns
    -------
    alpha : (n, k)
    beta  : (k, n)
    gamma : (c, m)
    theta : (c, m)
    Z     : (k, c)
        Biarchetype matrix.

    rss : float
        Final reconstruction error.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional matrix.")

    n, m = X.shape

    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n.")

    if not (1 <= c <= m):
        raise ValueError("c must satisfy 1 <= c <= m.")

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Initialization
    #
    # beta  : k x n
    # theta : c x m
    #
    # Every row is a convex combination.
    # --------------------------------------------------------

    beta = rng.dirichlet(np.ones(n), size=k)
    theta = rng.dirichlet(np.ones(m), size=c)

    # Z = beta X theta
    Z = beta @ X @ theta.T

    # Initial gamma: uniform convex combinations.
    gamma = np.full((c, m), 1.0 / c)

    previous_rss = np.inf

    # --------------------------------------------------------
    # Alternating optimization
    # --------------------------------------------------------

    for iteration in range(max_iter):

        # ---- Step 1: update alpha ---------------------------
        alpha = compute_alpha(X, Z, gamma)

        # ---- Step 2: update gamma ---------------------------
        gamma = compute_gamma(X, alpha, Z)

        # ---- Step 3: update Z -------------------------------
        Z = compute_Z_from_alpha(alpha, X, gamma)

        # ---- Step 4: update beta ----------------------------
        beta = compute_beta(Z, X, theta)

        # ---- Step 5: update theta ---------------------------
        theta = compute_theta(Z, beta, X)

        # ---- Step 6: update Z from beta and theta ----------
        Z = beta @ X @ theta

        # Re-estimate alpha and gamma for the current Z
        alpha = compute_alpha(X, Z, gamma)
        gamma = compute_gamma(X, alpha, Z)

        rss = compute_rss(X, alpha, Z, gamma)

        improvement = previous_rss - rss

        if verbose:
            print(
                f"Iteration {iteration + 1:3d}: "
                f"RSS = {rss:.8f}, "
                f"improvement = {improvement:.8f}"
            )

        if improvement >= 0 and improvement < tol:
            break

        previous_rss = rss

    return alpha, beta, gamma, theta, Z, rss


# ============================================================
# 9. Example
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Generate a small synthetic data set
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    n_per_cluster = 100

    X1 = rng.multivariate_normal(
        mean=[-3.0, -1.0],
        cov=[[0.6, 0.2],
             [0.2, 0.4]],
        size=n_per_cluster
    )

    X2 = rng.multivariate_normal(
        mean=[0.0, 3.0],
        cov=[[0.5, -0.1],
             [-0.1, 0.5]],
        size=n_per_cluster
    )

    X3 = rng.multivariate_normal(
        mean=[3.0, -1.0],
        cov=[[0.6, 0.15],
             [0.15, 0.5]],
        size=n_per_cluster
    )

    X = np.vstack([X1, X2, X3])

    print("Shape of X:", X.shape)

    # Number of archetypes for observations and variables.
    k = 3
    c = 2

    alpha, beta, gamma, theta, Z, rss = biarchetypal_analysis(
        X,
        k=k,
        c=c,
        max_iter=100,
        tol=1e-5,
        seed=42,
        verbose=True
    )

    print("\nFinal results")
    print("------------")
    print("alpha shape:", alpha.shape)
    print("beta shape :", beta.shape)
    print("gamma shape:", gamma.shape)
    print("theta shape:", theta.shape)
    print("Z shape    :", Z.shape)
    print("Final RSS  :", rss)

    print("\nBiarchetype matrix Z:")
    print(Z)

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    X_reconstructed = alpha @ Z @ gamma

    plt.figure(figsize=(9, 7))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        alpha=0.5,
        label="Observations X"
    )

    plt.scatter(
        X_reconstructed[:, 0],
        X_reconstructed[:, 1],
        s=40,
        marker="x",
        label="Reconstructed X"
    )

    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        s=180,
        marker="X",
        label="Biarchetype rows Z"
    )

    plt.xlabel("Variable 1")
    plt.ylabel("Variable 2")
    plt.title(f"Biarchetypal Analysis - RSS = {rss:.4f}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()