import numpy as np
from scipy.optimize import nnls, minimize
import matplotlib.pyplot as plt

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

def convex_least_squares(x, Z):
    """
    Solve the convex least-squares problem:

        min ||x - Z.T @ a||^2

    subject to:

        a >= 0
        sum(a) = 1
    """

    p = Z.shape[0]    
    T = Z.T

    def objective(a):
        diff = x - T @ a
        return np.dot(diff, diff)

    def jacobian(a):
        return -2 * T.T @ (x - T @ a)

    constraints = {'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0}
    bounds = [(0, None) for _ in range(p)]
    a0 = np.full(p, 1.0 / p)

    res = minimize(
        objective,
        a0,
        method='SLSQP',
        jac=jacobian,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )

    a = np.maximum(0, res.x)
    a_sum = a.sum()
    if a_sum > 0:
        a /= a_sum
    else:
        a = a0

    return a, res.fun

# ============================================================
# 3. Compute Z
# ============================================================

def compute_Z(B, X):
    """
    Compute the archetypes from beta and X.

        Z = B @ X
    """

    return B @ X

# ============================================================
# 5. Compute beta for one archetype
# ============================================================

def compute_B(X, A, Z, k, M=1.0):
    n, m = X.shape

    numerator = np.zeros(m)
    denominator = 0.0

    for i in range(n):
        if A[i, k] > 1e-8:  # Relajamos un poco el umbral numérico
            contribution = np.zeros(m)
            for l in range(Z.shape[0]):
                if l != k:
                    contribution += A[i, l] * Z[l]

            v_i = (X[i] - contribution) / A[i, k]
            numerator += (A[i, k] ** 2) * v_i
            denominator += A[i, k] ** 2

    if denominator == 0:
        return np.ones(n) / n

    v = numerator / denominator

    # Para evitar divergencia numérica en NNLS:
    # 1. Normalizamos T (X.T)
    T = X.T
    
    # 2. Construimos la matriz aumentada usando M reducido (M=1.0 por defecto)
    T_augmented = np.vstack([T, M * np.ones((1, n))])
    v_augmented = np.append(v, M)

    try:
        B_k, _ = nnls(T_augmented, v_augmented, maxiter=5000)
    except RuntimeError:
        B_k = np.ones(n) / n

    # Aseguramos la suma = 1 mediante proyección directa
    sum_B = np.sum(B_k)
    if sum_B > 0:
        B_k = B_k / sum_B

    return B_k

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
    B = np.zeros((p, X.shape[0]))

    Z_new = Z.copy()

    previous_rss = np.inf

    for inner_iteration in range(max_inner_iter):
        for k in range(p):
            B[k] = compute_B(X, A, Z_new, k)
            Z_new[k] = compute_Z(B[k:k+1], X)[0]

        rss = compute_rss(X, A, Z_new)

        improvement = previous_rss - rss

        if improvement < tol:
            break

        previous_rss = rss

    A_new = compute_A(X, Z_new)
    rss = compute_rss(X, A_new, Z_new)

    return A_new, B, Z_new, rss


# ============================================================
# 8. Complete Archetypal Analysis
# ============================================================

def archetypal_analysis(
    X,
    p,
    max_iter,
    max_inner_iter,
    tol
):
    rng = np.random.default_rng(42)
    n = X.shape[0]
   
    B = rng.dirichlet(np.ones(n), size=p)

    Z = compute_Z(B, X)
    Z_initial = Z.copy()

    previous_rss = np.inf

    for iteration in range(max_iter):

        # Perform one alternating iteration
        A, B, Z_new, rss = one_iteration(
            X,
            Z,
            tol = tol,
            max_inner_iter=max_inner_iter
        )

        # Improvement in RSS
        improvement = previous_rss - rss

        print(
            f"Iteration {iteration + 1}: "
            f"RSS = {rss:.6f}, "
            f"improvement = {improvement:.6f}"
        )

        # Check convergence
        if (improvement < tol):
            Z = Z_new
            break

        # Prepare next iteration
        Z = Z_new

        previous_rss = rss

    return A, B, Z, Z_initial, rss


# ============================================================
# 9. Example
# ============================================================

if __name__ == "__main__":

    # 1. Generación de datos 2D (Mezcla de 3 Gaussianas)
    rng = np.random.default_rng(42)
    n_per_cluster = 100

    # Cluster 1: centrado en (-3, -1)
    X1 = rng.multivariate_normal(
        mean=[-3.0, -1.0],
        cov=[[0.6, 0.2],
             [0.2, 0.4]],
        size=n_per_cluster
    )

    # Cluster 2: centrado en (0, 3)
    X2 = rng.multivariate_normal(
        mean=[0.0, 3.0],
        cov=[[0.5, -0.1],
             [-0.1, 0.5]],
        size=n_per_cluster
    )

    # Cluster 3: centrado en (3, -1)
    X3 = rng.multivariate_normal(
        mean=[3.0, -1.0],
        cov=[[0.6, 0.15],
             [0.15, 0.5]],
        size=n_per_cluster
    )

    # Matriz final de observaciones X con dimensión (300, 2)
    X = np.vstack([X1, X2, X3])
    print("Shape de X:", X.shape)

    p_archetypes = 3
    A, B, Z, Z_initial, rss = archetypal_analysis(X, p=p_archetypes, max_iter=100, max_inner_iter=100, tol=1e-4)

    # 3. Mostrar Resultados Finales
    plt.figure(figsize=(9, 7))

    plt.scatter(X[:, 0], X[:, 1], c='skyblue', alpha=0.6, edgecolors='k', linewidths=0.3, label='Datos $X$ (300 muestras)')
    plt.scatter(Z_initial[:, 0], Z_initial[:, 1], c='gray', s=120, marker='o', label='Z Iniciales (Aleatorios)')
    plt.scatter(Z[:, 0], Z[:, 1], c='red', s=200, marker='X', label='Arquetipos Finales $Z$')

    # Envolvente convexo
    Z_polygon = np.vstack([Z, Z[0]])
    plt.plot(Z_polygon[:, 0], Z_polygon[:, 1], 'r--', linewidth=2, label='Envolvente Arquetípico')

    plt.title(f"Archetypal Analysis en 2D\nRSS Final = {rss:.4f}")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()