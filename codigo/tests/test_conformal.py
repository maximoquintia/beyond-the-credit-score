# tests/test_conformal.py: chequeamos por simulación que conformal.py cumple la garantía.
# Se corre con `pytest tests/` (tarda un par de minutos). La idea: simulamos muchas veces un problema donde sabemos
# quién es moroso, corremos Conformal Selection con alfa = 5% y miramos la FDR promedio. Probamos con tres scores:
#   - oráculo: la probabilidad verdadera de buen pago (el mejor score posible): tiene que cumplir y aprobar a muchos;
#   - ruido: un score al azar, que no sabe nada: tiene que cumplir igual, pero aprobando a casi nadie;
#   - logística: un modelo de verdad, entrenado con otra muestra: tiene que cumplir.
# Es la misma lección del informe: la garantía no depende del score; lo que depende del score es cuántos se aprueban.
import numpy as np
from sklearn.linear_model import LogisticRegression
from conformal import conformal_select

ALPHA = 0.05     # nivel de FDR
REPS = 500       # cuántas veces repetimos la simulación (la garantía es en promedio)
N = 20000        # solicitantes por repetición (la mitad calibra y la otra mitad se decide)

def simular(rng, n=N):
    """Datos de juguete: dos variables normales y una probabilidad de mora logística (mora media ~9%).
    Devuelve las variables x, la probabilidad verdadera de mora y quién fue moroso (y = 1)."""
    x = rng.normal(size=(n, 2))
    p_def = 1 / (1 + np.exp(-(-3 + x[:, 0] + x[:, 1])))
    y = rng.binomial(1, p_def)
    return x, p_def, y

def correr(tipo_mu, seed):
    """Repite REPS veces: simula, arma el score según tipo_mu ("oraculo", "ruido" o la logística), calibra con la
    primera mitad y decide sobre la segunda. Devuelve la FDR promedio y la cantidad promedio de aprobados."""
    rng = np.random.default_rng(seed)
    fdrs, n_aprobados = [], []
    for _ in range(REPS):
        x, p_def, y = simular(rng)
        if tipo_mu == "oraculo":
            mu = 1 - p_def                          # la verdad: prob. de buen pago
        elif tipo_mu == "ruido":
            mu = rng.uniform(size=N)                # un score que no sabe nada
        else:
            x_tr, _, y_tr = simular(rng)            # una muestra aparte para entrenar la logística
            mu = LogisticRegression().fit(x_tr, y_tr).predict_proba(x)[:, 0]   # columna 0 = prob. de buen pago
        m = N // 2
        aprobado = conformal_select(mu[:m], y[:m], mu[m:], ALPHA, rng)   # calibra con la 1ra mitad, decide la 2da
        moroso = y[m:] == 1
        fdrs.append((aprobado & moroso).sum() / max(aprobado.sum(), 1))
        n_aprobados.append(aprobado.sum())
    return np.mean(fdrs), np.mean(n_aprobados)

def test_oraculo():
    # con el mejor score: cumple (dejamos 1 punto de margen por el ruido de 500 repeticiones) y aprueba a muchos
    fdr, aprob = correr("oraculo", seed=1)
    print("oráculo: FDR", fdr, "| aprobados", aprob)
    assert fdr <= ALPHA + 0.01
    assert aprob > 1000

def test_ruido():
    # con un score inútil: igual cumple, pero casi no aprueba a nadie (no hay forma de distinguir)
    fdr, aprob = correr("ruido", seed=2)
    print("ruido: FDR", fdr, "| aprobados", aprob)
    assert fdr <= ALPHA + 0.01
    assert aprob < 100

def test_logistica():
    # con un modelo entrenado de verdad: cumple
    fdr, aprob = correr("logistica", seed=3)
    print("logística: FDR", fdr, "| aprobados", aprob)
    assert fdr <= ALPHA + 0.01
