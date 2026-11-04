# Reproducibilidad

La consigna del TP pide un archivo de R para replicar los análisis. Este trabajo está hecho en **Python** (PyTorch, XGBoost, scikit-learn), que es el lenguaje que figuraba en la propuesta y que acordamos con el profesor (Gabriel Martos Venturini). Acá explico cómo replicar todo con ese lenguaje.

## Qué implementamos y qué es una librería

- **Implementado desde cero en este proyecto:** Conformal Selection (p-valores conformales y Benjamini-Hochberg, en `conformal.py`); la capa GP con random Fourier features, la red con normalización espectral y bloques residuales, la SNGP, el entrenamiento con early stopping, la matriz de precisión y la varianza predictiva (`redes.py`); el ensamble, la regla ingenua, la simulación y todo el protocolo experimental. Las pruebas de que Conformal Selection controla la FDR están en `tests/test_conformal.py`.
- **Librerías que usamos como piezas:** PyTorch (gradientes y optimizador), XGBoost (el modelo de referencia, no el método que estudiamos), scikit-learn (Lasso, particiones, vecinos más cercanos, métricas) y pandas / numpy / scipy / matplotlib.

## Entorno

- Python 3.14 (probado en Windows). Las versiones exactas de las librerías están en `requirements.txt`: `pip install -r requirements.txt`.
- Para abrir los notebooks hace falta Jupyter o VS Code con la extensión de Jupyter (`pip install ipykernel` si falta).
- Todo corre en CPU, sin GPU.

## Datos

En la carpeta que entregamos ya están en `codigo/data/`. En el repositorio de GitHub no están (pesan demasiado y la licencia de Kaggle no deja redistribuirlos), así que ahí hay que bajarlos de Kaggle (competencia *Home Credit Default Risk*) y dejarlos en `codigo/data/`:

| Archivo | Se usa en | Tamaño aprox. |
|---|---|---|
| `application_train.csv` | 01 y 02 | 166 MB |
| `bureau.csv` | 01 y 02 (para marcar quién es thin-file) | 170 MB |

El resto de los resultados se genera corriendo los notebooks. La simulación (06), las medias lunas (07) y los tests no necesitan datos externos.

## Cómo replicar

Correr los notebooks en este orden: **01, 02, 03, 05, 08, 04**, y después **06, 07, 09 y 10** en cualquier orden. Cada uno lee lo que guardó el anterior en `codigo/outputs/`. En total son unas dos horas, casi todo por las redes del 05 y el 08 y por la simulación del 06.

Atajos:

- Para replicar el experimento principal (tablas 1 y 2, figuras 2 y 3) sin reentrenar nada, alcanza con tener `codigo/outputs/preds/` e `codigo/outputs/ids_test.csv` y correr el 04.
- Para replicar las figuras 3 y 4 (y las de apoyo) alcanza con lo mismo, más `datos_procesados.npz`, y correr el 10.
- Para replicar la simulación y las medias lunas (figuras 1, 5 y 6) no hace falta ningún dato: se corren el 06 y el 07.
- `python -m pytest tests/` corre las pruebas de Conformal Selection (desde la carpeta `codigo/`, para que encuentre `conformal.py`).

## Semillas y qué esperar

Todo tiene semilla fija: 42 para la partición de los datos y los modelos de Home Credit, `r` = 0, ..., 19 para cada partición del pool en calibración y prueba, `1000 + k` para cada red del ensamble, y la semilla de cada escenario en la simulación. Con las mismas versiones de las librerías se obtienen los mismos números. En otra máquina o con otra versión de PyTorch, los resultados de las redes pueden moverse en el tercer decimal por cuestiones numéricas; las conclusiones no cambian.

La regla del protocolo: todo lo que se elige (las variables y la escala del GP, el λ de la SNGP, la arquitectura de la red, el λ de XGBoost + distancia) se elige en validación, y el pool se evalúa una sola vez.

## Qué notebook produce cada resultado del informe

| Resultado | Notebook | Archivo o bloque |
|---|---|---|
| 14% de thin-file, mora de 10,1% contra 7,7% | 01 | Bloque 2 |
| Figura 1 (medias lunas) | 07 | `codigo/outputs/F1_medias_lunas.pdf` |
| Tabla 1 (AUC, log-loss y ECE) | 04 | Bloque 13, `codigo/outputs/tabla_T1.csv` |
| Tabla 2 (aprobación y FDR, con la regla ingenua) | 04 | Bloques 9 y 14, `codigo/outputs/tabla_T2.csv`, `codigo/outputs/tabla_regla_ingenua.csv` |
| Figura 2 (aprobación contra α) | 04 | `codigo/outputs/F2_aprobacion.pdf` |
| Figura 3 (cajas) y los cinco cocientes de medianas | 10 (Bloque 6) | `codigo/outputs/F3_cajas.pdf` |
| El cociente 1,12 de la varianza de la SNGP (figura de apoyo) | 04 | `codigo/outputs/F3_varianza.pdf` |
| Diferencia de AUC ensamble − XGBoost con su intervalo bootstrap | 04 (Bloque 12) y 08 (Bloque 3) | |
| Cocientes 0,79 (GP), 1,01 y 1,19 (ensamble) y 0,998 (kNN) | 03, 08 y 09 | |
| λ = 0 para la SNGP | 05 | Bloque 6 |
| λ = 0 para XGBoost + distancia | 09 | Bloque 2 |
| Terciles de discrepancia y de varianza GP | 08 | Bloque 5 |
| Figura 4 (con el ρ = 0,94) | 10 | `codigo/outputs/extra_rangos_xgb_ensamble.pdf` |
| Figuras de apoyo: densidades con el corte conformal y mapa de la simulación (con el 92% / 9,2%) | 10 | `codigo/outputs/extra_densidades_conformal.pdf`, `codigo/outputs/extra_mapa_simulacion.pdf` |
| Figura 5 y la ventaja según distancia | 06 | `codigo/outputs/F4_simulacion.pdf`, `codigo/outputs/sim_resumen.csv` |
| Figura 6 (FDR en la simulación) | 06 | `codigo/outputs/F5_fdr_thin_sim.pdf` |
| Cómo ordena cada modelo a los thin-file de la trampa | 06 | Bloque 6 |
