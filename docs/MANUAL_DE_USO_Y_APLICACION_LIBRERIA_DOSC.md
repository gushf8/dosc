# Manual Oficial de Arquitectura, Instalación y Aplicación: Librería `dosc`

**Autor y Creador:** Gustavo Fernando Huaman Fernandez  
**Filiación:** Escuela de Posgrado, Universidad Tecnológica del Perú  
**Repositorio Oficial de la Librería:** [https://github.com/gushf8/dosc](https://github.com/gushf8/dosc)  
**Versión:** 1.0.0 (Production / Stable)  
**Licencia de Código:** MIT License  
**Licencia Metodológica / Paper:** Creative Commons Attribution 4.0 International (CC-BY 4.0)

---

## 1. Visión General: ¿Qué es `dosc` y qué problema resuelve?

En econometría aplicada, ciencias sociales y salud pública, los paneles longitudinales puros (seguir a la misma persona año tras año) son escasos, costosos de recolectar y sufren de alta deserción muestral (*attrition bias*). En su lugar, los países disponen de **cortes transversales repetidos** (como la ENAHO en Perú, CASEN en Chile, CPS en Estados Unidos o Encuesta Continua de Hogares).

Desde hace 40 años, la metodología estándar de **Deaton (1985)** agrupaba observaciones en rangos demográficos arbitrarios (ej. "hombres nacidos entre 1980 y 1984"). Sin embargo, este enfoque clásico tiene **cuatro limitaciones severas**:
1. **Pérdida de heterogeneidad interna:** Mezcla individuos heterogéneos dentro del mismo cajón.
2. **Cardinalidad arbitraria:** No existe un criterio matemático para decidir si deben ser 5, 10 o 30 cohortes.
3. **Sesgo por error de medición (EIV):** Genera atenuación asintótica (subestima el efecto real de la política o educación entre 40% y 80%).
4. **Fronteras rígidas:** Los individuos en los límites de edad caen de forma binaria en cohortes distintas.

### La Solución de DOSC (*Dynamic Optimal Synthetic Cohorts*):
`dosc` transforma cortes transversales repetidos en **pseudo-paneles sintéticos óptimos** mediante:
* **Aprendizaje Estadístico No Supervisado:** Descubre perfiles latentes invariantes a partir de múltiples covariables estructurales.
* **Proyección Suave de Mahalanobis:** Asigna pertenencias continuas $p_{igt}$ escaladas por temperatura para evitar fronteras rígidas.
* **Control de Muestreo Complejo:** Incorpora factores de expansión muestral y tamaño de muestra efectivo de **Kish** ($n_{gt}^{\text{eff}}$).
* **Optimización Endógena de $G^*$:** Encuentra automáticamente el número óptimo de cohortes minimizando una función de pérdida multi-objetivo (balanceando sesgo distributivo vs. varianza).
* **Estimador WLS con Efectos Fijos:** Estima modelos de Two-Way Fixed Effects con errores estándar clusterizados y robustos.
* **Prueba de Equivalencia TOST:** Demuestra matemáticamente que el parámetro recuperado es equivalente al del panel real.

---

## 2. Instalación de la Librería en Python

La librería `dosc` está completamente empaquetada bajo los estándares modernos de Python (`pyproject.toml` y `setup.py`, compatible con Python 3.9, 3.10, 3.11 y 3.12).

### Opción A: Instalación directa desde GitHub (Recomendada para usuarios)
Cualquier usuario en el mundo puede instalarla ejecutando en su terminal:

```bash
pip install git+https://github.com/gushf8/dosc.git
```

### Opción B: Instalación local en Modo Desarrollo (Editable)
Si tienes el repositorio clonado en tu máquina local:

```bash
cd dosc_public
pip install -e .
```
*(El flag `-e` permite modificar el código fuente en `src/dosc` y ver los cambios reflejados al instante).*

### Dependencias Principales:
* `numpy >= 1.22.0`
* `pandas >= 1.4.0`
* `scipy >= 1.8.0`
* `scikit-learn >= 1.0.0`

---

## 3. Estructura Modular del Paquete (`src/dosc/`)

El núcleo de la librería está diseñado de forma modular, desacoplada y tipada:

| Módulo | Componente Principal | Función y Responsabilidad |
| :--- | :--- | :--- |
| `dosc.config` | `DOSCConfig` | Dataclass declarativa de hiperparámetros (rango de cohortes, umbral de estabilidad, temperatura Mahalanobis). |
| `dosc.engine` | `DOSCEngine` | Motor central que optimiza el número de cohortes $G^*$ y genera la matriz pseudo-panel ponderada. |
| `dosc.estimators` | `fit_synthetic_wls` | Estimador de Mínimos Cuadrados Ponderados (WLS) con Efectos Fijos Bidireccionales (Two-Way FE) y varianza clusterizada. |
| `dosc.metrics` | `two_one_sided_tost` | Contraste de hipótesis de equivalencia de Schuirmann (TOST) para validar la recuperación del parámetro real. |
| `dosc.metrics` | `calculate_ier` | Ratio de Eficiencia de Inferencia (IER) para comparar DOSC frente a Deaton y microdatos. |
| `dosc.dgp` | `generate_synthetic_panel_dgp` | Generador calibrado de procesos generadores de datos (DGP) para benchmarks y simulaciones de Monte Carlo. |

---

## 4. Guía de Uso Paso a Paso: Aplicación Práctica

### Ejemplo 1: Aplicación con Microdatos Reales (Caso Tipo ENAHO / Encuestas de Hogares)

Supongamos que dispones de una base de datos de encuestas repetidas `encuesta_anual.csv` con columnas:
* `year`: Año de la encuesta (corte temporal $t = 1, \dots, T$).
* `birth_year`, `sex`, `education_base`: Covariables invariantes en el tiempo para formar cohortes ($Z$).
* `experiencia`, `horas_trabajadas`: Covariables variables en el tiempo ($X$).
* `log_ingreso`: Variable dependiente de resultado ($Y$).
* `factor_expansion`: Ponderador muestral de la encuesta ($w_i$).

```python
import pandas as pd
from dosc import DOSCEngine, DOSCConfig, fit_synthetic_wls

# 1. Cargar los datos
df = pd.read_csv("encuesta_anual.csv")

# 2. Configurar el motor de optimización
config = DOSCConfig(
    G_min=6,                # Mínimo de cohortes a evaluar
    G_max=30,               # Máximo de cohortes a evaluar
    step_size=2,            # Paso de búsqueda en la grilla
    gamma_temperature=1.0,  # Parámetro de temperatura (suavizado Mahalanobis)
    tau_s=0.70,             # Umbral mínimo de estabilidad longitudinal (70%)
    n_min=20.0,             # Tamaño muestral efectivo mínimo de Kish por celda
    random_state=42         # Semilla determinista reproducible
)

engine = DOSCEngine(config=config)

# 3. Ajustar DOSC sobre el DataFrame
resultado_dosc = engine.fit_dataframe(
    df=df,
    time_col="year",
    invariant_cols=["birth_year", "sex", "education_base"],
    varying_cols=["experiencia", "horas_trabajadas"],
    outcome_col="log_ingreso",
    weight_col="factor_expansion"
)

# 4. Inspeccionar la cardinalidad óptima encontrada
print(f"Cohortes óptimas seleccionadas (G*): {resultado_dosc.optimal_G}")
print(f"Estabilidad temporal lograda: {resultado_dosc.temporal_stability:.4f}")
print(f"Pérdida multi-objetivo mínima: {resultado_dosc.multi_loss:.4f}")

# 5. Obtener el Pseudo-Panel Agregado
pseudo_panel_df = resultado_dosc.pseudo_panel
print(pseudo_panel_df.head())

# 6. Estimar el Modelo Econométrico Two-Way Fixed Effects
regresion = fit_synthetic_wls(
    panel_df=pseudo_panel_df,
    outcome_col="log_ingreso",
    regressor_cols=["experiencia", "horas_trabajadas"],
    cohort_col="cohort_id",
    time_col="year",
    weight_col="effective_n"
)

# 7. Resumen de resultados estilo Stata / R
print(regresion.summary())
```

---

### Ejemplo 2: Simulación y Prueba de Equivalencia Estadística (TOST)

Para verificar si el estimador recupera el "Ground Truth" poblacional $\beta^* = 0.50$:

```python
from dosc import generate_synthetic_panel_dgp, DOSCEngine, DOSCConfig, fit_synthetic_wls, two_one_sided_tost

# 1. Generar datos sintéticos con heterogeneidad latente no observada
df_sim, beta_true = generate_synthetic_panel_dgp(
    T=6,             # 6 oleadas temporales
    N_per_wave=2500, # 2,500 observaciones por año
    true_beta=0.50,  # Efecto estructural real
    seed=42
)

# 2. Ejecutar DOSC
config = DOSCConfig(G_min=4, G_max=16, step_size=2)
engine = DOSCEngine(config=config)
res = engine.fit_dataframe(
    df=df_sim,
    time_col="wave",
    invariant_cols=["Z_1", "Z_2", "Z_3"],
    varying_cols=["X_1"],
    outcome_col="Y"
)

# 3. Estimar con WLS
modelo = fit_synthetic_wls(
    panel_df=res.pseudo_panel,
    outcome_col="Y",
    regressor_cols=["X_1"],
    cohort_col="cohort_id",
    time_col="wave",
    weight_col="effective_n"
)

beta_estimado = modelo.params["X_1"]
se_estimado = modelo.bse["X_1"]

# 4. Prueba TOST de Schuirmann (+/- 10% margen de equivalencia)
tost_res = two_one_sided_tost(
    estimate=beta_estimado,
    benchmark=beta_true,
    se=se_estimado,
    df=modelo.df_resid,
    delta=0.05
)

print(tost_res.summary())
# Si p_value < 0.05: Se rechaza la hipótesis nula de no-equivalencia.
# Conclusión: DOSC recupera el parámetro poblacional con equivalencia garantizada.
```

---

## 5. Batería de Pruebas Unitarias (`pytest`)

Para garantizar la estabilidad y reproducibilidad del paquete:

```bash
cd dosc_public
pytest tests/ -v
```
Las pruebas validan:
* Invarianza de escala en las proyecciones de Mahalanobis.
* Restricciones de no-negatividad de la fórmula de Kish.
* Convergencia monótona de la función de pérdida multi-objetivo.
* Exactitud de los grados de libertad en el estimador Two-Way FE.

---

## 6. Cita Académica

Si utilizas la librería `dosc` en artículos de investigación, tesis o reportes de consultoría, debes citarla según el formato oficial:

### Formato APA 7ma Edición:
```text
HUAMAN FERNANDEZ, G. F. (2026). Can Repeated Cross-Sections Replicate Panels? Optimal Synthetic Cohorts for Complex Survey Econometrics. https://doi.org/10.5281/zenodo.24326322
```

### Formato BibTeX:
```bibtex
@article{huaman2026dosc,
  author       = {Huaman Fernandez, Gustavo Fernando},
  title        = {Can Repeated Cross-Sections Replicate Panels? Optimal Synthetic Cohorts for Complex Survey Econometrics},
  year         = {2026},
  doi          = {10.5281/zenodo.24326322},
  url          = {https://doi.org/10.5281/zenodo.24326322},
  journal      = {Zenodo Preprint}
}
```
