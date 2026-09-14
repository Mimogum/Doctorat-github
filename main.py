import numpy as np
from scipy.optimize import nnls

def convex_least_squares(u, T, M=1000):
    """
    Solve the Convex Least Squares problem:

        minimize ||u - T @ w||^2

    subject to:

        w >= 0
        sum(w) = 1

    using the NNLS formulation described in
    Cutler & Breiman (1994).
    """

    # Number of candidate vectors
    q = T.shape[1]

    # Add the artificial observation that forces sum(w) ≈ 1
    T_augmented = np.vstack([
        T,
        M * np.ones(q)
    ])

    u_augmented = np.append(u, M)

    # Solve the non-negative least squares problem
    w, residual = nnls(T_augmented, u_augmented)

    # Normalize to make the convex coefficients sum exactly to 1
    if w.sum() > 0:
        w = w / w.sum()

    return w

T = np.array([[0.0, 2.0], [0.0, 2.0]])

u = np.array([1.0, 1.0])

w = convex_least_squares(u, T)

print("Weights:", w)
print("Sum of weights:", w.sum())
print("Reconstructed u:", T @ w)
print("Original u:", u)

def compute_A(X, Z):
    """
    Compute the matrix A.

    Each row A[i] contains the convex coefficients
    used to reconstruct observation X[i] from the
    archetypes Z.
    """

    n = X.shape[0]
    p = Z.shape[0]

    A = np.zeros((n, p))

    for i in range(n):
        A[i] = convex_least_squares(
            X[i],
            Z.T
        )

    return A

X = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 2.0],
    [2.0, 2.0]
])

Z = np.array([
    [0.0, 0.0],
    [2.0, 2.0]
])

A = compute_A(X, Z)

print("\nA:")
print(A) # Es projecta al punt mig (1,1) el (2,0) i el (0,2)
print("\nSum of each row:")
print(A.sum(axis=1))