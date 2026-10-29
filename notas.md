# Notas del proyecto

Mis notas para tener a mano en la defensa: qué encontramos, con qué números y de dónde sale cada uno. El detalle de cómo correr todo está en el `README.md` y en `REPRODUCIBILIDAD.md`.

## La idea en 30 segundos

- **Pregunta:** ¿un modelo cuya incertidumbre crece con la distancia a los datos de entrenamiento (la SNGP) aprueba a más solicitantes sin historial de bureau ("thin-file"), cuando se decide con Conformal Selection? Si sí, ¿cuándo? Y ¿qué aporta Conformal Selection frente a creerle a la probabilidad del modelo?
- **Datos:** Home Credit Default Risk (Kaggle), 307.511 solicitantes. `TARGET = 1` es moroso. Thin-file = no tener ningún crédito en `bureau.csv`: son 44.020 (14%), con 10,1% de mora contra 7,7% del resto.
- **Herramienta de decisión:** Conformal Selection (Jin y Candès, 2023). Con un score por solicitante y un nivel α, elige a quién aprobar de modo que, en promedio, la proporción de morosos entre los aprobados (la FDR) no pase de α. La garantía vale con cualquier score; un score que ordena mejor solo hace que se apruebe a más gente.
- **Modelos:** XGBoost (la vara), GP con random features sobre 20 variables, SNGP sobre las 200 variables y un ensamble de 5 SNGP.
- **Hipótesis de partida:** los thin-file se parecen menos a los datos de entrenamiento; un modelo que lo nota debería dudar de ellos, limpiar el extremo seguro del ranking y aprobar a más con el mismo control de riesgo.

## Qué encontramos

1. **En Home Credit ningún modelo con incertidumbre le gana a XGBoost.** AUC thin-file en el pool: XGBoost 0,7203, ensamble 0,7126, SNGP 0,7091, GP 0,6996 (Tabla 1, notebook 04). El ensamble cierra la mitad de la brecha de una red sola, pero sigue abajo de forma significativa: ensamble − XGBoost = −0,0077, IC 95% [−0,013; −0,002] (04 y 08). Aprobación thin-file a α 3 / 5 / 7 / 9%: XGBoost 28,9 / 54,4 / 80,4 / 95,6%; ensamble 26,7 / 54,2 / 78,6 / 95,4%; SNGP 22,6 / 52,1 / 78,3 / 95,4%; GP 16,1 / 49,7 / 78,4 / 94,5% (Tabla 2 y Figura 2).
2. **El motivo: los thin-file no están lejos.** Con cinco medidas, el cociente de medianas thin-file / resto da cerca de 1: varianza de la SNGP 1,12 (04), varianza GP del ensamble 1,01 y discrepancia entre redes 1,19 (08), varianza del GP sobre 20 variables 0,79 (03) y distancia a los vecinos más cercanos 0,998 (09). Corregir XGBoost por distancia elige λ = 0 con las dos medidas que probamos (09), y a la SNGP la validación también le elige λ = 0 (05). La forma de decirlo: un thin-file **sí es distinto** del resto (le faltan las variables de bureau), pero **no está lejos** de los datos de entrenamiento, porque en train hay miles como él. La varianza de la SNGP mide lo segundo, no si el solicitante es más difícil de predecir. Y como XGBoost y el ensamble ordenan a los thin-file casi igual (ρ = 0,94, Figura 4), aprueban casi a los mismos.
3. **Cuando sí están lejos, funciona.** En la simulación (06), con thin-file a distancia máxima 6, la SNGP aprueba 19,1 puntos más que XGBoost a α = 6% y el ensamble 18,3; hasta distancia 2 la diferencia está entre −0,5 y +1,6 puntos (cero, en la práctica); a distancia 4 es puro ruido. La varianza de la SNGP sigue a la distancia (ρ = 0,86 en la semilla del panel a); la discrepancia del ensamble mucho menos (0,28 en esa semilla, ~0,39 en promedio). El mecanismo (06, Bloque 6): dentro de la trampa, XGBoost les da a todos los lejanos casi el mismo riesgo (~7-8%, cuando su mora real va de 6,5% a 40%) porque se queda con lo último que aprendió en el borde de los datos; la SNGP también los subestima, pero su predicción vuelve hacia el promedio al alejarse y los ordena bien entre sí (AUC dentro de los lejanos 0,78 contra 0,56). Con el mismo α, XGBoost tiene que ser más estricto con todos.
4. **Conformal Selection cumple siempre, pero calibrando por segmento.** Por segmento, la FDR thin-file a α = 5% queda entre 4,9% y 5,1% con los cuatro modelos. Calibrando todo junto, dentro de thin-file se pasa: 6,4% XGBoost, 6,5% ensamble, 6,7% SNGP y 7,2% GP (04). En la simulación pasa lo mismo: a α = 10%, 12,7% con XGBoost y 11,8% con la SNGP agrupando (06).
5. **Conformal Selection no necesita que el modelo esté bien calibrado.** La regla ingenua (aprobar mientras el riesgo predicho promedio de los aprobados no pase de α) casi cumple con XGBoost, que está bien calibrado (ECE thin-file 0,0039; 5,3% de morosos a α = 5%), pero con el GP, que subestima la mora de los thin-file (predice 8,2% contra 10,1% real), aprueba al 73% y termina con 6,8% de morosos, un 35% más de lo prometido (04, Bloque 14).
6. **No todas las medidas de incertidumbre dicen lo mismo.** Por terciles dentro de thin-file, el AUC de XGBoost baja cuando sube la discrepancia entre redes (0,741 → 0,694) y sube con la varianza GP (0,685 → 0,760): la discrepancia marca dónde se equivoca el modelo y la varianza no. Es descriptivo (~2 errores estándar) (08).

## Las cuatro lecciones

1. **Diagnosticar antes de modelar.** El cociente de distancias thin-file / resto ya anticipaba que la SNGP no iba a ayudar en Home Credit. Una medida de incertidumbre solo vale si hay distancia que medir.
2. **La garantía conformal viaja con cualquier score, pero hay que calibrar por segmento** para que se cumpla dentro de thin-file. Y no depende de la calibración del modelo.
3. **No todas las medidas de incertidumbre son iguales:** la discrepancia entre redes marca dónde falla el modelo; la varianza GP no.
4. **Para un banco:** XGBoost + Conformal Selection por segmento, mientras los clientes nuevos no estén lejos de los conocidos. Si se alejan, ahí conviene un modelo que sepa cuándo no sabe.

## Limitaciones (para tenerlas claras)

- El mecanismo de la distancia está mostrado solo en una simulación construida para que funcione.
- La garantía de Conformal Selection es en promedio (una partición suelta puede pasarse de α) y supone que los solicitantes nuevos son intercambiables con los de calibración; un corrimiento en el tiempo puede romper eso.
- XGBoost (200 variables) y el GP (20) no son comparables uno a uno.
- Línea de investigación: buscar un dataset real con clientes genuinamente fuera del soporte de los datos.

## Lo que probamos y no entró al informe

- Un "GP ajustado" por su varianza (04): casi igual al GP. Sigue en el código porque sacarlo cambiaría los sorteos de los modelos que vienen después.
- Mezclas 50/50 de XGBoost con la SNGP o con el ensamble (08): empatan con XGBoost.
- Grilla de arquitecturas de la SNGP (05): ninguna mejora a la red base; quedó la misma.

## Regla del proyecto

Todo lo que se elige (variables y escala del GP, λ, arquitectura) se elige en validación; el pool se evalúa una sola vez. Lo que se probó después de ver resultados se presenta como análisis posterior.
