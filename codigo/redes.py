# redes.py: las piezas de PyTorch del proyecto. Las usan el 03 (GP sobre variables), el 05 (SNGP), el 06 (simulación),
# el 07 (medias lunas), el 08 (ensamble) y el 10 (mapa de la simulación).
#
# La idea general, en criollo: queremos un modelo que, además de decir "este solicitante tiene X% de probabilidad de
# mora", diga qué tan seguro está de ese número. Un Proceso Gaussiano (GP) hace eso: su incertidumbre es chica cerca
# de los datos que vio y grande lejos. El problema es que un GP exacto con 184 mil filas no entra en memoria (hay que
# invertir una matriz de 184 mil x 184 mil), así que lo aproximamos con Random Fourier Features (Rahimi y Recht, 2007).
#
# El archivo tiene dos partes:
#   1) GP con random features (GPHead) y las funciones que sirven para cualquier modelo: entrenar, ajustar_covarianza
#      y predecir. Con un "cuerpo" que no hace nada (Identidad), es el GP del 03 sobre las 20 variables de Lasso.
#   2) La SNGP (Liu et al., 2023): se reemplaza el cuerpo vacío por una red con normalización espectral. La cabeza GP,
#      el entrenamiento y la varianza quedan exactamente iguales.
import copy   # para guardar una copia del mejor modelo durante el entrenamiento
import math   # raíces, pi

import torch
import torch.nn as nn

from torch.nn.utils.parametrizations import spectral_norm   # normalización espectral de pesos


# ---------------------------------------------------------------------------------------------------------------------
# PARTE 1. El GP con random features
# ---------------------------------------------------------------------------------------------------------------------
# Truco de las random features: transformamos cada solicitante en D = 1024 números con
#     phi(x) = raíz(2/D) * cos(W x + b),
# donde W (direcciones al azar) y b (desfases al azar) se sortean UNA vez y quedan fijos. Con esa transformación, el
# producto phi(x)·phi(x') aproxima al kernel gaussiano k(x, x') = exp(-||x - x'||² / (2 scale²)), que es la medida de
# "parecido" entre dos solicitantes que usa el GP. Sobre esos 1024 números ajustamos una regresión logística (beta):
# es lo único que se entrena, y una logística sobre estas features se comporta como un GP.
# La "scale" (el largo de escala del kernel) controla qué tan rápido dos solicitantes dejan de parecerse a medida que
# se alejan: scale grande = el modelo es más suave y "ve cerca" a más gente.

class GPHead(nn.Module):
    """Capa de salida tipo Proceso Gaussiano, aproximada con random Fourier features.
    d: cantidad de variables que le entran (las k de Lasso en el GP, o el ancho de la red en la SNGP).
    D: cantidad de random features (1024).
    scale: largo de escala del kernel; si no se pasa, se usa raíz(d).
    ridge: con qué valor arranca la matriz de precisión P (la "prior"; evita que P sea singular)."""

    def __init__(self, d, D=1024, scale=None, ridge=1.0):
        super().__init__()                      # inicializa la maquinaria interna de nn.Module
        scale = scale or math.sqrt(d)           # escala del kernel; si no la damos, raíz(d)
        # W y b son aleatorios y FIJOS: se guardan como buffers, no como parámetros (el optimizador no los toca).
        # W: D direcciones aleatorias; dividir por scale equivale a medir las distancias en unidades de scale
        self.register_buffer("W", torch.randn(D, d) / scale)
        # b: un desfase aleatorio entre 0 y 2*pi para cada coseno
        self.register_buffer("b", 2 * math.pi * torch.rand(D))
        # beta es lo único que se entrena: una regresión logística sobre las D features (D pesos -> 1 logit de mora)
        self.beta = nn.Linear(D, 1, bias=False)
        self.D, self.ridge = D, ridge            # guardamos D y el ridge para usarlos en ajustar_covarianza

    def phi(self, h):
        """Las random Fourier features de una tanda h (filas = solicitantes, columnas = d variables)."""
        # h @ W.T + b: D proyecciones aleatorias por solicitante; cos(...): las features
        # raíz(2/D): la normalización que hace que phi(x)·phi(x') aproxime al kernel gaussiano
        return math.sqrt(2.0 / self.D) * torch.cos(h @ self.W.T + self.b)


class Identidad(nn.Module):
    """Cuerpo vacío: devuelve las variables tal cual. En la SNGP, en este lugar va la red."""

    def features(self, x):
        return x                                 # no transforma nada: el GP mira las variables directo


class GPVariables(nn.Module):
    """El GP del 03: random features directo sobre las variables que le pasemos (las k elegidas por Lasso).
    d_in: cuántas variables entran.
    scale: largo de escala del kernel (None -> raíz(d_in))."""

    def __init__(self, d_in, scale=None):
        super().__init__()
        self.body = Identidad()                  # cuerpo: en la SNGP será la red
        self.head = GPHead(d_in, scale=scale)    # cabeza: la capa GP de arriba

    def forward(self, x):
        # x -> cuerpo -> random features (phi) -> beta -> logit de mora; squeeze(-1) saca la dimensión de tamaño 1
        return self.head.beta(self.head.phi(self.body.features(x))).squeeze(-1)


@torch.no_grad()                                 # no calcula gradientes: más rápido y sin gastar memoria
def perdida_promedio(model, loader):
    """Pérdida (entropía cruzada binaria) promedio sobre todo un loader, sin entrenar.
    model: la red o el GP.
    loader: el DataLoader con las tandas (x, y) a evaluar."""
    model.eval()                                 # modo evaluación
    perdida_fn = nn.BCEWithLogitsLoss(reduction="sum")  # suma (no promedio) para poder promediar bien al final
    total, n = 0.0, 0                            # acumuladores: pérdida total y cantidad de filas
    for x, y in loader:                          # una tanda (batch) a la vez
        total += perdida_fn(model(x), y).item()  # .item() pasa el tensor a un número de Python
        n += len(y)
    return total / n                             # promedio exacto, aunque la última tanda sea más chica


def entrenar(model, train_loader, val_loader, lr=1e-3, max_epocas=50, paciencia=5, verbose=True):
    """Loop de entrenamiento con early stopping sobre la pérdida de validación.
    model: lo que se entrena (GPVariables o SNGP).
    train_loader / val_loader: las tandas de train y de validación.
    lr: tamaño del paso de Adam.
    max_epocas: tope de pasadas completas por train.
    paciencia: cuántas épocas seguidas sin mejorar en validación toleramos antes de cortar.
    verbose: si imprime la pérdida de cada época.
    Devuelve el historial [(época, pérdida train, pérdida val), ...] y deja cargados los pesos de la MEJOR época
    (no los de la última)."""
    perdida_fn = nn.BCEWithLogitsLoss()          # entropía cruzada para clasificación binaria, a partir del logit
    optimizador = torch.optim.Adam(model.parameters(), lr=lr)  # Adam mueve los pesos entrenables; lr = tamaño del paso
    mejor_val, mejor_estado, sin_mejora = float("inf"), None, 0  # mejor pérdida hasta ahora, su copia y un contador
    historial = []
    for epoca in range(1, max_epocas + 1):      # una época = una pasada completa por train
        model.train()                            # modo entrenamiento
        suma, n = 0.0, 0
        for x, y in train_loader:
            optimizador.zero_grad()              # 1. borrar los gradientes del paso anterior
            perdida = perdida_fn(model(x), y)    # 2. predecir y medir el error
            perdida.backward()                   # 3. calcular hacia dónde mover cada peso (backpropagation)
            optimizador.step()                   # 4. mover los pesos un poquito en esa dirección
            suma += perdida.item() * len(y)      # acumula la pérdida ponderada por el tamaño de la tanda
            n += len(y)
        perdida_train = suma / n                 # pérdida promedio de la época en train
        perdida_val = perdida_promedio(model, val_loader)  # pérdida en datos que el modelo NO usó para aprender
        historial.append((epoca, perdida_train, perdida_val))
        if verbose:
            print(f"época {epoca:2d} | pérdida train {perdida_train:.4f} "
                  f"| pérdida val {perdida_val:.4f}")
        if perdida_val < mejor_val - 1e-5:       # ¿mejoró de verdad (más que 1e-5)?
            mejor_val, sin_mejora = perdida_val, 0
            mejor_estado = copy.deepcopy(model.state_dict())  # guardamos una COPIA de los pesos actuales
        else:
            sin_mejora += 1                      # una época más sin mejorar
            if sin_mejora >= paciencia:          # early stopping: si pasan `paciencia` épocas sin mejorar, cortamos
                break
    model.load_state_dict(mejor_estado)          # volvemos al mejor modelo, no al último (así no sobreajustamos)
    return historial


@torch.no_grad()
def ajustar_covarianza(model, loader):
    """Una pasada por train, ya con el modelo entrenado: acumula la matriz de precisión P y guarda su Cholesky.
    model: el modelo YA entrenado.
    loader: todo train en tandas (no hace falta mezclar).
    Teoría: es la aproximación de Laplace de la logística sobre las features. P = I + suma de p(1-p) phi phi^T sobre
    train (la ecuación 4 del informe). Pensalo como un mapa de por dónde pasaron los datos: donde hubo muchos
    solicitantes parecidos, P es "grande" en esa dirección (hay información); donde no hubo nadie, P se queda en la
    prior. Cada solicitante pesa p(1-p): cuenta más aquel sobre el que el modelo todavía duda."""
    model.eval()
    head = model.head                            # atajo: la cabeza GP, donde viven phi, beta y la matriz P
    # P arranca en ridge x identidad: la prior (sin datos, la incertidumbre es la máxima)
    P = head.ridge * torch.eye(head.D, dtype=torch.float64)  # float64: más precisión numérica
    for x, _ in loader:                          # el "_" es y, que acá no hace falta
        phi = head.phi(model.body.features(x)).double()      # features de la tanda, en doble precisión
        p = torch.sigmoid(head.beta(phi.float())).squeeze(-1).double()  # prob. de mora que predice el modelo
        # cada solicitante suma a P su phi*phi^T, pesado por p(1-p)
        P += (phi * (p * (1 - p)).unsqueeze(1)).T @ phi
    head.L = torch.linalg.cholesky(P)            # P = L L^T; guardamos L (nunca invertimos P directamente)


@torch.no_grad()
def predecir(model, X, batch=4096):
    """Devuelve (logit de mora, varianza) para cada fila de X, por tandas.
    model: el modelo entrenado y con la covarianza ya ajustada.
    X: tensor [n, d_in].
    batch: cuántas filas por tanda (para no llenar la memoria).
    La varianza es phi(x)^T P^-1 phi(x): chica si el solicitante cae en una zona muy visitada en train (el modelo
    sabe lo que sabe) y grande si cae en una zona casi vacía (el modelo "sabe que no sabe"). Es la incertidumbre
    sensible a la distancia de todo el trabajo."""
    model.eval()
    logits, varianzas = [], []                   # una lista por tanda; al final se pegan
    for i in range(0, len(X), batch):            # recorre X de a `batch` filas
        phi = model.head.phi(model.body.features(X[i:i + batch]))   # features de la tanda
        logits.append(model.head.beta(phi).squeeze(-1))             # el logit de mora de cada fila
        v = torch.cholesky_solve(phi.double().T, model.head.L)      # resuelve P v = phi, o sea v = P^-1 phi
        varianzas.append((phi.double().T * v).sum(0).float())       # phi^T P^-1 phi, fila por fila: la varianza
    return torch.cat(logits).numpy(), torch.cat(varianzas).numpy()  # une las tandas y pasa a numpy


# ---------------------------------------------------------------------------------------------------------------------
# PARTE 2. La SNGP: red con normalización espectral + la misma capa GP
# ---------------------------------------------------------------------------------------------------------------------
# La red transforma las 200 variables en una representación h(x) de d_oculta = 128 números, y la cabeza GP trabaja
# sobre h(x) en vez de sobre las variables. Para que la incertidumbre siga siendo sensible a la distancia, la red no
# puede deformar las distancias: si dos solicitantes están lejos en las variables, tienen que seguir lejos en h(x).
# Eso se logra con dos cosas juntas:
#   - la normalización espectral (Miyato et al., 2018) acota la norma de cada matriz de pesos (||W||_2 <= 1), así que
#     ninguna capa puede estirar las distancias más allá de un factor fijo;
#   - los bloques residuales (salida = x + coef * algo) conservan la entrada, así que dos puntos distintos no pueden
#     colapsar en uno (con coef < 1, cada bloque estira o comprime las distancias como mucho entre (1 - coef) y (1 + coef)).
# Ojo: la primera capa (la que pasa de 200 a 128) no es residual, así que esa garantía no vale para ella.

class BloqueResidual(nn.Module):
    """Un bloque de la red: salida = x + coef * ReLU(W x + b), con W de norma espectral <= 1.
    d: ancho del bloque (entra y sale con d números).
    coef: cuánto pesa la corrección (0.9, menor a 1); es fijo, no se aprende."""

    def __init__(self, d, coef=0.9):
        super().__init__()
        self.lineal = spectral_norm(nn.Linear(d, d))   # capa lineal d -> d (con sesgo), con su norma topada en 1
        self.act = nn.ReLU()                  # max(0, z): el "codo" que permite curvas (sin él, apilar capas sería lineal)
        self.coef = coef

    def forward(self, x):
        return x + self.coef * self.act(self.lineal(x))   # la entrada se conserva y se le suma una corrección


class CuerpoSN(nn.Module):
    """El cuerpo de la SNGP: una capa de entrada y después n_bloques bloques residuales.
    d_in: variables de entrada (200 en Home Credit).
    d_oculta: ancho de la representación (128).
    n_bloques: cuántos bloques residuales (2).
    coef: cuánto corrige cada bloque (0.9)."""

    def __init__(self, d_in, d_oculta=128, n_bloques=2, coef=0.9):
        super().__init__()
        self.entrada = spectral_norm(nn.Linear(d_in, d_oculta))   # de 200 variables a 128 números
        self.act = nn.ReLU()                                      # el codo, después de la capa de entrada
        bloques = []
        for _ in range(n_bloques):                                # n_bloques bloques nuevos, cada uno con sus pesos
            bloques.append(BloqueResidual(d_oculta, coef))
        self.bloques = nn.ModuleList(bloques)                     # ModuleList: para que PyTorch los vea y los entrene

    def features(self, x):
        """La representación h(x). Se llama features igual que en Identidad, así entrenar, ajustar_covarianza y
        predecir funcionan sin cambios."""
        h = self.act(self.entrada(x))       # [batch, 200] -> [batch, 128], con ReLU
        for bloque in self.bloques:         # pasa por cada bloque, uno atrás del otro
            h = bloque(h)
        return h                            # [batch, 128]


class SNGP(nn.Module):
    """La SNGP: red con normalización espectral (cuerpo) + capa GP con random features (cabeza).
    d_in: variables de entrada (200 en Home Credit, 10 en la simulación, 2 en las medias lunas).
    d_oculta: ancho de la red. n_bloques: cuántos bloques residuales. coef: cuánto corrige cada bloque.
    scale: largo de escala del kernel de la cabeza GP (None -> raíz(d_oculta))."""

    def __init__(self, d_in, d_oculta=128, n_bloques=2, coef=0.9, scale=None):
        super().__init__()
        self.body = CuerpoSN(d_in, d_oculta, n_bloques, coef)   # en el GP del 03 acá iba Identidad()
        self.head = GPHead(d_oculta, scale=scale)               # la cabeza ve los d_oculta números del cuerpo

    def forward(self, x):
        # igual que GPVariables: cuerpo (ahora la red) -> random features -> beta -> logit de mora
        return self.head.beta(self.head.phi(self.body.features(x))).squeeze(-1)
