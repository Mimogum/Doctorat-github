import numpy as np
from scipy.optimize import nnls
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

def compute_beta(X, A, Z, k, M=1.0):
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
        return np.zeros(n)

    v = numerator / denominator

    # Para evitar divergencia numérica en NNLS:
    # 1. Normalizamos T (X.T)
    T = X.T
    
    # 2. Construimos la matriz aumentada usando M reducido (M=1.0 por defecto)
    T_augmented = np.vstack([T, M * np.ones((1, n))])
    v_augmented = np.append(v, M)

    # 3. Pequeña regularización en la diagonal para evitar ill-conditioning
    eps = 1e-6
    T_augmented += eps * np.random.randn(*T_augmented.shape)

    try:
        beta_k, _ = nnls(T_augmented, v_augmented)
    except RuntimeError:
        # Fallback de seguridad en caso de no convergencia de Scipy
        beta_k = np.ones(n) / n

    # Aseguramos la suma = 1 mediante proyección directa
    sum_beta = np.sum(beta_k)
    if sum_beta > 0:
        beta_k = beta_k / sum_beta

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
    max_inner_iter=100,
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
        if (improvement < tol):
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

    # 2. Inicialización de los 3 arquetipos Z rodeando a los clusters
    Z_initial = np.array([
        [-4.0, -2.0],
        [ 0.0,  4.0],
        [ 4.0, -2.0]
    ])

    # 3. Ejecución del Algoritmo
    A, beta, Z, rss = archetypal_analysis(
        X,
        Z_initial,
        max_iter=50,
        max_inner_iter=50,
        tol=1e-4
    )

    # 4. Mostrar Resultados Finales
    plt.figure(figsize=(9, 7))

    # 1. Graficar los datos (Gaussianas 2D)
    plt.scatter(X[:, 0], X[:, 1], c='skyblue', alpha=0.6, edgecolors='k', linewidths=0.3, label='Datos $X$ (300 muestras)')

    # 2. Graficar los Arquetipos Iniciales
    plt.scatter(Z_initial[:, 0], Z_initial[:, 1], c='gray', s=120, marker='o', linestyle='--', label='Z Iniciales')

    # 3. Graficar los Arquetipos Finales
    plt.scatter(Z[:, 0], Z[:, 1], c='red', s=200, marker='X', label='Arquetipos Finales $Z$')

    # 4. Dibujar el Envolvente Convexo (Simplex) formado por los arquetipos finales
    Z_polygon = np.vstack([Z, Z[0]])  # Cerrar el triángulo
    plt.plot(Z_polygon[:, 0], Z_polygon[:, 1], 'r--', linewidth=2, label='Envolvente Arquetípico')

    plt.title(f"Archetypal Analysis en 2D (Gaussianas)\nRSS Final = {rss:.4f}")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()