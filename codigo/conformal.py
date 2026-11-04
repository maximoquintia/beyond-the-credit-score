# conformal.py: Conformal Selection (Jin y Candès, 2023), la herramienta de decisión de todo el trabajo.
#
# La idea: queremos aprobar a la mayor cantidad posible de solicitantes con una garantía, que entre los aprobados la
# proporción esperada de morosos (la FDR) no pase de alfa. Para eso:
#   1) a cada solicitante nuevo le damos un p-valor que mide qué tan "parecido a un moroso" es según su score,
#      comparándolo con los morosos REALES de una muestra de calibración (ecuación 2 del informe);
#   2) Benjamini-Hochberg mira todos los p-valores juntos y decide a cuántos aprobar (ecuación 3).
# La garantía vale en promedio y para cualquier score, siempre que los solicitantes nuevos sean intercambiables con
# los de calibración. Del score solo importa el ORDEN: un modelo que ordena mejor no cambia la garantía, solo hace que
# se apruebe a más gente.
import numpy as np

def benj_hochberg(p, q):
    """Benjamini-Hochberg: decide a quién aprobar a partir de los p-valores. Devuelve True/False por solicitante.
    p: vector de p-valores.
    q: nivel de FDR (alfa).
    Pasos: ordenamos los p-valores de menor a mayor, comparamos el k-ésimo con el umbral q*k/m, buscamos el MAYOR k
    que cumple y aprobamos a los k de p-valor más chico. Intuición: aprobar a k solicitantes cuela, como mucho, unos
    m*p(k) morosos, o sea una fracción m*p(k)/k entre los aprobados; pedir que eso sea <= q es pedir p(k) <= q*k/m."""
    m = len(p)                                     # cantidad de solicitantes a decidir
    orden = np.argsort(p)                          # posiciones ordenadas de menor a mayor p-valor
    p_ord = p[orden]
    umbral = q * np.arange(1, m + 1) / m           # q*k/m para k = 1..m
    cumplimiento = p_ord <= umbral                 # True donde el k-ésimo p-valor queda bajo su umbral
    if cumplimiento.any():
        k = np.nonzero(cumplimiento)[0].max() + 1  # el mayor k que cumple (aunque alguno anterior no cumpla)
    else:
        k = 0                                      # nadie cumple: no se aprueba a nadie
    aprobado = np.zeros(m, dtype=bool)
    aprobado[orden[:k]] = True                     # se aprueban los k de menor p-valor
    return aprobado


def conformal_pvalues(mu_cal, y_cal, mu_test, rng):
    """p-valor conformal de cada solicitante nuevo (ecuación 2 del informe).
    mu_cal / y_cal: scores (prob. de buen pago) y TARGET (1 = moroso) de la calibración.
    mu_test: scores de los solicitantes a decidir.
    rng: generador de números aleatorios para el desempate (la U de la fórmula).
    El p-valor es, más o menos, la fracción de la calibración que son morosos con un score MEJOR que el del
    solicitante: (morosos con mu mayor + U * (1 + morosos con mu igual)) / (n + 1), con n el tamaño total de la
    calibración. Un p-valor chico quiere decir que muy pocos morosos se ven tan bien como él, así que probablemente
    es buen pagador. La U al azar reparte los empates de forma justa, y es lo que hace que la garantía sea exacta."""
    n = len(mu_cal)                                # total de la calibración (va en el denominador)
    mu_morosos = np.sort(mu_cal[y_cal == 1])       # mu de los morosos de calibración, ordenados
    pos_der = np.searchsorted(mu_morosos, mu_test, side="right")  # cuántos morosos tienen mu <= el del solicitante
    pos_izq = np.searchsorted(mu_morosos, mu_test, side="left")   # cuántos morosos tienen mu < el del solicitante
    mayores = len(mu_morosos) - pos_der            # morosos con mu MAYOR (se ven mejor que el solicitante)
    iguales = pos_der - pos_izq                    # morosos con mu IGUAL (empates)
    U = rng.uniform(size=len(mu_test))             # un desempate al azar por solicitante
    return (mayores + U * (1 + iguales)) / (n + 1)


def conformal_select(mu_cal, y_cal, mu_test, alpha, rng):
    """Conformal Selection completo: p-valores y después Benjamini-Hochberg. Devuelve True = aprobado.
    mu_cal, y_cal, mu_test, rng: como en conformal_pvalues.
    alpha: FDR objetivo (proporción máxima esperada de morosos entre los aprobados)."""
    p = conformal_pvalues(mu_cal, y_cal, mu_test, rng)
    return benj_hochberg(p, alpha)
