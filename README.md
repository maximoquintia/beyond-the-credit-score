# Beyond the Credit Score

Trabajo final de Machine Learning para Economistas (UTDT, 2026), de Máximo Quintiá.

**El informe completo está en [`informe/beyond-the-credit-score_Quintia_Maximo.pdf`](informe/beyond-the-credit-score_Quintia_Maximo.pdf).**

**La pregunta:** los solicitantes de crédito sin historial de bureau ("thin-file") son justamente aquellos sobre los que un modelo de scoring sabe menos. ¿Sirve un modelo cuya incertidumbre crece con la distancia a los datos de entrenamiento (la SNGP, que combina una red neuronal con un Proceso Gaussiano) para aprobar a más thin-file, cuando la decisión se toma con Conformal Selection, que controla la proporción de morosos entre los aprobados? Y si sirve, ¿cuándo? De paso, ¿qué aporta Conformal Selection frente a la regla ingenua de aprobar según la probabilidad predicha?

**La respuesta corta:**

- En Home Credit ningún modelo con incertidumbre le gana a XGBoost (AUC en thin-file de 0,709 para la SNGP, 0,713 para un ensamble de cinco redes y 0,720 para XGBoost). La razón es que los thin-file no están más lejos de los datos de entrenamiento que el resto: con cinco medidas distintas, el cociente de distancias thin-file / resto da cerca de 1.
- En una simulación donde controlamos la distancia, cuando los thin-file están lejos la SNGP aprueba unos 19 puntos más que XGBoost con la misma garantía, y la ventaja desaparece cuando están cerca.
- Conformal Selection cumple su promesa con los cuatro modelos, en Home Credit y en la simulación, siempre que se calibre por segmento. Y, a diferencia de la regla ingenua, su garantía no depende de que el modelo esté bien calibrado.

## Los modelos

| Modelo | Qué es | Dónde |
|---|---|---|
| XGBoost | Árboles con boosting: el estándar de la industria y nuestra vara | `02` |
| GP | Proceso Gaussiano aproximado con random Fourier features, sobre las 20 variables que elige un Lasso | `03`, `redes.py` |
| SNGP | Red residual con normalización espectral + capa GP de salida, sobre las 200 variables | `05`, `redes.py` |
| Ensamble | Promedio de 5 SNGP con semillas distintas; su incertidumbre es la discrepancia entre las redes | `08` |

`conformal.py` tiene Conformal Selection (p-valores conformales y Benjamini-Hochberg) y `redes.py` todas las piezas de PyTorch. Las dos cosas están implementadas desde cero en este proyecto y comentadas línea por línea.

## Cómo correrlo

1. Instalar las dependencias: `pip install -r requirements.txt` (más `ipykernel` si tu editor lo pide para abrir notebooks).
2. Tener en `data/` los archivos `application_train.csv` y `bureau.csv`. En la carpeta entregada ya están; si clonaste el repo de GitHub, hay que bajarlos de Kaggle (competencia *Home Credit Default Risk*).
3. Correr los notebooks en este orden: **01, 02, 03, 05, 08, 04**, y después **06, 07, 09 y 10** en cualquier orden. El 04 y el 10 necesitan las predicciones que guardan el 05 y el 08.
4. Opcional: `python -m pytest tests/` chequea por simulación que Conformal Selection controla la FDR.

Todo corre en CPU. `REPRODUCIBILIDAD.md` tiene el detalle (versiones, semillas, atajos para no reentrenar).

| Notebook | Qué hace | Qué guarda en `outputs/` | Tiempo aprox. |
|---|---|---|---|
| `01_datos` | Limpia y procesa los datos, define thin-file y parte en train / calib / test | `datos_procesados.npz`, `ids_*.csv`, `preproc.joblib`, `nombres_columnas.csv` | 2 min |
| `02_xgboost` | Entrena XGBoost y mira cómo le va con los thin-file | `preds/xgboost_*.csv` | 5 min |
| `03_gp` | El GP: Lasso, grilla de hiperparámetros en validación y modelo final | `preds/gp_vars_*.csv` | 10 min |
| `05_sngp` | La SNGP: entrenamiento y elección de λ y de la arquitectura en validación | `preds/sngp_vars_*.csv` | 30 min |
| `08_ensamble` | Las 5 SNGP, la comparación con XGBoost y el diagnóstico por terciles de incertidumbre | `preds/sngp_ensamble.npz`, `ensamble_resultados.csv` | 15 min |
| `04_experimento` | El experimento principal: 4 modelos, 20 particiones del pool, por segmento y agrupado, y la regla ingenua | `tabla_T1.csv`, `tabla_T2.csv`, `tabla_regla_ingenua.csv`, `resultados_experimento.csv`, F2, F3 | 5 min |
| `06_simulacion` | Thin-file a distancia controlada (XGBoost, SNGP y ensamble) | `sim_*.csv`, F4, F5 de la simulación | 1 hora |
| `07_medias_lunas` | El ejemplo en 2D del mecanismo | F1 | 5 min |
| `09_xgb_distancia` | XGBoost corregido por distancia (kNN y varianza de la SNGP) | `xgb_distancia_resultados.csv` | 5 min |
| `10_figuras_extra` | Las figuras 4, 5 y 6 del informe | `extra_*` | 5 min |

Los tiempos son en una notebook sin GPU y varían bastante.

**Un detalle de nombres:** en el 01 y el 02 las tres partes se llaman train (60%), calib (10%) y test (30%). Desde el 03 cambian de rol: el 10% "calib" pasa a ser **validación** (para elegir hiperparámetros y para el early stopping) y el 30% "test" pasa a ser el **pool**, donde corre el experimento conformal (que lo parte 20 veces en calibración y prueba). Los archivos conservaron los nombres de la primera etapa.

## Dónde sale cada figura y tabla del informe

| En el informe | Qué muestra | Archivo | Notebook |
|---|---|---|---|
| Figura 1 | Medias lunas: probabilidad e incertidumbre de los 4 modelos en 2D | `outputs/F1_medias_lunas.pdf` | 07 |
| Figura 2 | Tasa de aprobación contra α, thin-file y resto | `outputs/F2_aprobacion.pdf` | 04 |
| Figura 3 | Varianza de la SNGP, thin-file contra resto | `outputs/F3_varianza.pdf` | 04 |
| Figura 4 | Percentil de riesgo de cada thin-file según XGBoost y según el ensamble | `outputs/extra_rangos_xgb_ensamble.pdf` | 10 |
| Figura 5 | Densidades del riesgo predicho con el corte de Conformal Selection | `outputs/extra_densidades_conformal.pdf` | 10 |
| Figura 6 | Mapa de la simulación en 2D con la varianza de la SNGP | `outputs/extra_mapa_simulacion.pdf` | 10 |
| Figura 7 | Simulación: incertidumbre contra distancia, aprobación y ventaja según distancia | `outputs/F4_simulacion.pdf` | 06 |
| Figura 8 | FDR dentro de thin-file en la simulación, por segmento contra agrupada | `outputs/F5_fdr_thin_sim.pdf` | 06 |
| Tabla 1 | AUC, log-loss y ECE por modelo y segmento | `outputs/tabla_T1.csv` | 04 |
| Tabla 2 | Aprobación y FDR por segmento, agrupada e ingenua | `outputs/tabla_T2.csv` y `outputs/tabla_regla_ingenua.csv` (el Bloque 14 del 04 la imprime armada) | 04 |

Los nombres de los archivos no siguen la numeración del informe porque las figuras se fueron armando antes que el texto; esta tabla es la que manda.

## Estructura

```
beyond-credit-score/
├── 01_datos.ipynb ... 10_figuras_extra.ipynb
├── conformal.py, redes.py
├── tests/test_conformal.py
├── requirements.txt
├── README.md, REPRODUCIBILIDAD.md, notas.md
├── informe/      (el PDF del informe)
├── data/          (en la entrega están los dos CSV; en GitHub no, se bajan de Kaggle)
└── outputs/       (en git solo las figuras y las tablas; el resto se regenera corriendo los notebooks)
    └── preds/     (las predicciones de cada modelo)
```

## Referencias principales

- Jin, Y. y Candès, E. J. (2023). Selection by prediction with conformal p-values. *Journal of Machine Learning Research*, 24(244).
- Liu, J. Z., Padhy, S., Ren, J., Lin, Z., Wen, Y., Jerfel, G., Nado, Z., Snoek, J., Tran, D. y Lakshminarayanan, B. (2023). A simple approach to improve single-model deep uncertainty via distance-awareness. *Journal of Machine Learning Research*, 24(42).
- Benjamini, Y. y Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1).
- Lakshminarayanan, B., Pritzel, A. y Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Rahimi, A. y Recht, B. (2007). Random features for large-scale kernel machines. *NeurIPS*.
- Miyato, T., Kataoka, T., Koyama, M. y Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. *ICLR*.
- Chen, T. y Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD*.
