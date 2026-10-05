import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Archetypal Analysis - Mørup & Hansen (2012)
# ============================================================
#
# Convención de este archivo:
#
#   X : (N, M)
#       N observaciones, M variables/features.
#
# El artículo usa las observaciones como COLUMNAS:
#
#   X_art : (M, N)
#
# Por tanto, en nuestra convención:
#
#   C : (N, D)       -> coeficientes que definen los arquetipos
#   S : (D, N)       -> coeficientes que representan las observaciones
#
# Los arquetipos son:
#
#   Z = C.T @ X      -> (D, M)
#
# y la reconstrucción es:
#
#   X_hat = S.T @ Z
#         = S.T @ C.T @ X
#
# que es exactamente la transpuesta de:
#
#   X_art @ C @ S
#
# del artículo.
#
# Las restricciones son:
#
#   C >= 0,     sum_n C[n,d] = 1  para cada d
#   S >= 0,     sum_d S[d,n] = 1  para cada n
#
# ============================================================


# ============================================================
# 1. FURTHESTSUM
# ============================================================

def furthest_sum(X, D, rng=None):
    """
    FURTHESTSUM de Mørup & Hansen (2012).

    X tiene forma (N, M): observaciones en filas.

    Devuelve D índices de observaciones candidatas para inicializar C.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D.")

    N, M = X.shape

    if D < 1:
        raise ValueError("D debe ser >= 1.")

    if D > N:
        raise ValueError(
            "D no puede ser mayor que el número de observaciones."
        )

    if rng is None:
        rng = np.random.default_rng()

    # Paso 1 del Algorithm 1:
    # elegir una observación aleatoria.
    j = int(rng.integers(N))
    selected = [j]

    # q_i = ||x_i - x_j||_F
    q = np.linalg.norm(X - X[j], axis=1)

    # q_hat = q
    q_hat = q.copy()

    # El algoritmo añade D puntos y después elimina
    # el primer punto aleatorio.
    for t in range(1, D + 1):

        if t == D:
            q = q - q_hat

        available = np.ones(N, dtype=bool)
        available[selected] = False

        j = int(np.argmax(np.where(available, q, -np.inf)))
        selected.append(j)

        distances = np.linalg.norm(X - X[j], axis=1)
        q = q + distances

    # El primer punto aleatorio se elimina.
    # Esto implementa literalmente la observación del artículo:
    # "the first point selected by random is removed".
    return np.asarray(selected[1:], dtype=int)


# Funciones auxiliares para la normalización y el cálculo de gradientes
# ============================================================
# 2. NORMALIZACIÓN
# ============================================================

def normalize_columns(A, eps=1e-15):
    """
    Normaliza cada columna para que sume 1.

    Además impone no negatividad.
    """

    A = np.maximum(A, 0.0)

    sums = A.sum(axis=0, keepdims=True)
    sums = np.maximum(sums, eps)

    return A / sums


# Función objetivo y gradientes para la actualización de C y S
# ============================================================
# 3. FUNCIÓN OBJETIVO
# ============================================================

def aa_objective(X, C, S):
    """
    Objetivo de Archetypal Analysis:

        ||X - X_art C S||_F^2

    expresado con X en filas:

        ||X - S.T C.T X||_F^2
    """

    X_hat = S.T @ C.T @ X
    residual = X - X_hat

    return np.sum(residual ** 2)


# ============================================================
# 4. GRADIENTE RESPECTO DE S
# ============================================================

def gradient_S(XTX, C, S):
    """
    Gradiente normalizado respecto de S.

    El artículo, usando observaciones como columnas, da:

        G_S =
            C.T X.T X C S
            - C.T X.T X

    donde aquí:

        XTX = X_art.T X_art
            = X @ X.T

    Después se utiliza la corrección de invariancia a la
    normalización descrita en el artículo:

        G_tilde[d,n] =
            G[d,n]
            - sum_d' G[d',n] S[d',n]

    """

    # C.T X.T X C
    CtXC = C.T @ XTX @ C

    # C.T X.T X
    CtXX = C.T @ XTX

    # Gradiente indicado en el artículo
    G = CtXC @ S - CtXX

    # Corrección de invariancia de normalización
    correction = np.sum(
        G * S,
        axis=0,
        keepdims=True
    )

    G_tilde = G - correction

    return G_tilde


# ============================================================
# 5. GRADIENTE RESPECTO DE C
# ============================================================

def gradient_C(XTX, C, S):
    """
    Gradiente normalizado respecto de C.

    El artículo da:

        G_C =
            X.T X C S S.T
            - X.T X S.T

    """

    # X.T X C S S.T
    G = XTX @ C @ S @ S.T

    # - X.T X S.T
    G -= XTX @ S.T

    # Corrección de invariancia de normalización
    correction = np.sum(
        G * C,
        axis=0,
        keepdims=True
    )

    G_tilde = G - correction

    return G_tilde


# ============================================================
# 6. ACTUALIZACIÓN PROYECTADA DE S
# ============================================================

def update_S(XTX, C, S, step):
    """
    Una actualización projected-gradient de S.

    El artículo utiliza:

        S_new =
            max(
                S - step * G_tilde_S,
                0
            )

    seguida de la normalización L1.
    """

    G = gradient_S(XTX, C, S)

    S_new = S - step * G

    # Proyección sobre el ortante no negativo
    S_new = np.maximum(S_new, 0.0)

    # Normalización para imponer sum_d S[d,n] = 1
    S_new = normalize_columns(S_new)

    return S_new


# ============================================================
# 7. ACTUALIZACIÓN PROYECTADA DE C
# ============================================================

def update_C(XTX, C, S, step):
    """
    Una actualización projected-gradient de C.
    """

    G = gradient_C(XTX, C, S)

    C_new = C - step * G

    # No negatividad
    C_new = np.maximum(C_new, 0.0)

    # sum_n C[n,d] = 1
    C_new = normalize_columns(C_new)

    return C_new


# ============================================================
# 8. LINE SEARCH PARA S
# ============================================================

def line_search_S(
    X,
    XTX,
    C,
    S,
    initial_step=1.0,
    max_search=10
):
    """
    Line search por backtracking.

    El artículo especifica que el tamaño de paso se ajusta
    mediante line-search y que se realizan 10 line-search
    updates por actualización alternante.

    El artículo NO especifica la regla exacta de backtracking.
    Por ello usamos aquí la regla estándar de dividir el paso
    por 2 hasta encontrar una mejora.
    """

    current_loss = aa_objective(X, C, S)

    step = initial_step

    for _ in range(max_search):

        candidate = update_S(
            XTX,
            C,
            S,
            step
        )

        candidate_loss = aa_objective(
            X,
            C,
            candidate
        )

        if candidate_loss < current_loss:
            return candidate, step, candidate_loss

        step *= 0.5

    # Si ningún paso mejora la función, mantenemos S.
    return S.copy(), step, current_loss


# ============================================================
# 9. LINE SEARCH PARA C
# ============================================================

def line_search_C(
    X,
    XTX,
    C,
    S,
    initial_step=1.0,
    max_search=10
):
    """
    Line search por backtracking para C.
    """

    current_loss = aa_objective(X, C, S)

    step = initial_step

    for _ in range(max_search):

        candidate = update_C(
            XTX,
            C,
            S,
            step
        )

        candidate_loss = aa_objective(
            X,
            candidate,
            S
        )

        if candidate_loss < current_loss:
            return candidate, step, candidate_loss

        step *= 0.5

    return C.copy(), step, current_loss


# ============================================================
# 10. INICIALIZACIÓN DE C
# ============================================================

def initialize_C(X, D, rng):
    """
    Inicializa C mediante FURTHESTSUM.

    Cada arquetipo comienza siendo exactamente una observación:

        C[index, d] = 1
    """

    selected = furthest_sum(
        X,
        D,
        rng=rng
    )

    N = X.shape[0]

    C = np.zeros((N, D), dtype=float)

    for d, index in enumerate(selected):
        C[index, d] = 1.0

    return C, selected


# ============================================================
# 11. INICIALIZACIÓN DE S
# ============================================================

def initialize_S(D, N, rng):
    """
    Inicialización aleatoria de S y normalización por columnas.
    """

    S = rng.random((D, N))

    return normalize_columns(S)


# ============================================================
# 12. ARQUETIPOS
# ============================================================

def compute_archetypes(X, C):
    """
    Calcula los arquetipos:

        Z = C.T @ X

    Z tiene forma (D, M).
    """

    return C.T @ X


# ============================================================
# 13. GRADIENT CHECK
# ============================================================

def check_gradients(
    X,
    C,
    S,
    epsilon=1e-6
):
    """
    Comprueba numéricamente que las expresiones analíticas
    del gradiente coinciden con derivadas por diferencias finitas.

    Se compara el gradiente sin el factor 2.

    La derivada real del objetivo ||...||² contiene un factor 2,
    mientras que las ecuaciones del artículo omiten ese factor,
    que queda absorbido por el tamaño del paso.
    """

    XTX = X @ X.T

    Gs = gradient_S(XTX, C, S)
    Gc = gradient_C(XTX, C, S)

    # Para comprobar la derivada matemática completa usamos
    # 2 * gradiente. La corrección de normalización NO forma
    # parte de la derivada euclídea simple; por ello para esta
    # prueba comprobamos primero el gradiente base.
    Gs_base = C.T @ XTX @ C @ S - C.T @ XTX
    Gc_base = XTX @ C @ S @ S.T - XTX @ S.T

    # ---- S ----
    i_s, j_s = 0, 0

    S_plus = S.copy()
    S_minus = S.copy()

    S_plus[i_s, j_s] += epsilon
    S_minus[i_s, j_s] -= epsilon

    numerical_S = (
        aa_objective(X, C, S_plus)
        - aa_objective(X, C, S_minus)
    ) / (2 * epsilon)

    analytical_S = 2 * Gs_base[i_s, j_s]

    # ---- C ----
    i_c, j_c = 0, 0

    C_plus = C.copy()
    C_minus = C.copy()

    C_plus[i_c, j_c] += epsilon
    C_minus[i_c, j_c] -= epsilon

    numerical_C = (
        aa_objective(X, C_plus, S)
        - aa_objective(X, C_minus, S)
    ) / (2 * epsilon)

    analytical_C = 2 * Gc_base[i_c, j_c]

    print("\n" + "=" * 60)
    print("COMPROBACIÓN NUMÉRICA DE LOS GRADIENTES")
    print("=" * 60)

    print("\nS:")
    print("  derivada numérica :", numerical_S)
    print("  derivada analítica:", analytical_S)
    print(
        "  error absoluto    :",
        abs(numerical_S - analytical_S)
    )

    print("\nC:")
    print("  derivada numérica :", numerical_C)
    print("  derivada analítica:", analytical_C)
    print(
        "  error absoluto    :",
        abs(numerical_C - analytical_C)
    )

    # Variables Gs/Gc se calculan para asegurar que las funciones
    # de gradiente completas son ejecutables.
    _ = Gs, Gc


# ============================================================
# 14. ARQUETYPAL ANALYSIS
# ============================================================

def archetypal_analysis(
    X,
    p,
    max_iter=100,
    max_inner_iter=10,
    tol=1e-6,
    random_state=42,
    initial_step_S=1.0,
    initial_step_C=1.0,
    line_search_steps=10,
    verbose=True
):
    """
    Archetypal Analysis mediante el projected gradient de
    Mørup & Hansen (2012).

    Parámetros
    ----------
    X : ndarray, shape (N, M)
        Datos con observaciones en filas.

    p : int
        Número de arquetipos.

    max_iter : int
        Número máximo de actualizaciones alternantes.

    max_inner_iter : int
        Número de line-search updates iniciales de S.

    tol : float
        Tolerancia relativa de convergencia.

    random_state : int, Generator o None
        Semilla/generador.

    initial_step_S : float
        Paso inicial para S.

    initial_step_C : float
        Paso inicial para C.

    line_search_steps : int
        Número de intentos del backtracking.

    verbose : bool
        Mostrar la evolución de la función objetivo.

    Returns
    -------
    A : ndarray, shape (N, p)
        Matriz C del artículo.

    B : ndarray, shape (p, N)
        Matriz S del artículo.

    Z : ndarray, shape (p, M)
        Arquetipos.

    history : ndarray
        Historial del objetivo.

    selected : ndarray
        Observaciones iniciales elegidas por FURTHESTSUM.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D.")

    N, M = X.shape

    if p < 1:
        raise ValueError("p debe ser >= 1.")

    if p > N:
        raise ValueError(
            "p no puede ser mayor que el número de observaciones."
        )

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # --------------------------------------------------------
    # Precomputación
    #
    # En la notación del artículo:
    #
    #   X_art.T @ X_art
    #
    # Como aquí X está en filas:
    #
    #   X @ X.T
    # --------------------------------------------------------

    XTX = X @ X.T

    # --------------------------------------------------------
    # Inicializar C mediante FURTHESTSUM
    # --------------------------------------------------------

    C, selected = initialize_C(
        X,
        p,
        rng
    )

    # --------------------------------------------------------
    # Inicializar S aleatoriamente
    # --------------------------------------------------------

    S = initialize_S(
        p,
        N,
        rng
    )

    for _ in range(max_inner_iter):

        S_new, initial_step_S, _ = line_search_S(
            X,
            XTX,
            C,
            S,
            initial_step=initial_step_S,
            max_search=line_search_steps
        )

        S = S_new

    # --------------------------------------------------------
    # Objetivo inicial
    # --------------------------------------------------------

    current_loss = aa_objective(X, C, S)

    history = [current_loss]

    if verbose:
        print("=" * 70)
        print("ARCHETYPAL ANALYSIS - PROJECTED GRADIENT")
        print("=" * 70)
        print(f"N observaciones : {N}")
        print(f"M variables     : {M}")
        print(f"D arquetipos    : {p}")
        print(f"Loss inicial    : {current_loss:.8e}")
        print(f"FURTHESTSUM     : {selected}")
        print()

    # --------------------------------------------------------
    # Optimización alternante
    # --------------------------------------------------------

    for iteration in range(max_iter):

        # ====================================================
        # Actualizar S
        # ====================================================

        S_new, initial_step_S, loss_S = line_search_S(
            X,
            XTX,
            C,
            S,
            initial_step=initial_step_S,
            max_search=line_search_steps
        )

        S = S_new

        # ====================================================
        # Actualizar C
        # ====================================================

        C_new, initial_step_C, loss_C = line_search_C(
            X,
            XTX,
            C,
            S,
            initial_step=initial_step_C,
            max_search=line_search_steps
        )

        C = C_new

        # ====================================================
        # Objetivo
        # ====================================================

        loss = aa_objective(X, C, S)

        history.append(loss)

        relative_change = (
            abs(current_loss - loss)
            / max(abs(current_loss), 1e-16)
        )

        if verbose:
            print(
                f"Iteración {iteration + 1:4d} | "
                f"Loss = {loss:.8e} | "
                f"cambio relativo = {relative_change:.3e} | "
                f"step_S = {initial_step_S:.3e} | "
                f"step_C = {initial_step_C:.3e}"
            )

        if relative_change < tol:
            break

        current_loss = loss

    # --------------------------------------------------------
    # Arquetipos finales
    # --------------------------------------------------------

    Z = compute_archetypes(X, C)

    return C, S, Z, np.asarray(history), selected


# ============================================================
# 15. PRUEBA
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # Crear datos sintéticos con 3 arquetipos conocidos.
    #
    # Cada observación es una combinación convexa de:
    #
    #   z1 = (0, 0)
    #   z2 = (1, 0)
    #   z3 = (0, 1)
    #
    # --------------------------------------------------------

    true_Z = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    n_samples = 100

    true_S = rng.dirichlet(
        np.ones(3),
        size=n_samples
    ).T

    X = true_S.T @ true_Z

    print("Shape de X:", X.shape)

    # --------------------------------------------------------
    # Ejecutar AA
    # --------------------------------------------------------

    C, S, Z, history, selected = archetypal_analysis(
        X,
        p=3,
        max_iter=100,
        max_inner_iter=10,
        tol=1e-8,
        random_state=42,
        verbose=True
    )

    # --------------------------------------------------------
    # Comprobaciones
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPROBACIONES")
    print("=" * 70)

    print("\nArquetipos verdaderos:")
    print(true_Z)

    print("\nArquetipos encontrados:")
    print(Z)

    print("\nÍndices FURTHESTSUM:")
    print(selected)

    print("\nSuma de columnas de C:")
    print(C.sum(axis=0))

    print("\nSuma de columnas de S:")
    print(S.sum(axis=0)[:10], "...")

    print("\nMínimo de C:", C.min())
    print("Mínimo de S:", S.min())

    print("\nLoss inicial:", history[0])
    print("Loss final:", history[-1])

    print(
        "\n¿La loss es no creciente?:",
        np.all(np.diff(history) <= 1e-10)
    )

    reconstruction = S.T @ C.T @ X
    reconstruction_error = np.sum(
        (X - reconstruction) ** 2
    )

    print(
        "Error de reconstrucción:",
        reconstruction_error
    )

    # --------------------------------------------------------
    # Gradient check
    # --------------------------------------------------------

    check_gradients(
        X,
        C,
        S
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 7))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c="skyblue",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
        label="Datos X"
    )

    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c="red",
        s=220,
        marker="X",
        label="Arquetipos encontrados"
    )

    plt.scatter(
        true_Z[:, 0],
        true_Z[:, 1],
        c="black",
        s=100,
        marker="o",
        label="Arquetipos verdaderos"
    )

    Z_polygon = np.vstack([Z, Z[0]])

    plt.plot(
        Z_polygon[:, 0],
        Z_polygon[:, 1],
        "r--",
        linewidth=2,
        label="Envolvente arquetípica"
    )

    plt.title(
        f"Archetypal Analysis - Projected Gradient\n"
        f"Loss final = {history[-1]:.6e}"
    )

    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()