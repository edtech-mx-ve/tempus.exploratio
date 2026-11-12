<div align="center">

<img src="assets/tempus-logo.png" width="150" alt="Logotipo TEMPUS EXPLORATIO">

# TEMPUS DECISIO — TEMPUS EXPLORATIO

### Módulo I · Exploración y pronóstico estadístico de series temporales bursátiles

**Edición local de 10 años · Python 3.12 · Streamlit · pandas · statsmodels · Plotly · PDF**

[Aplicación pública (edición anterior)](https://tempus-exploratio.streamlit.app/) ·
[Repositorio público (edición anterior)](https://github.com/edtech-mx-ve/tempus.exploratio)

</div>

> **Estado de publicación:** esta documentación describe la **edición local de diez años**. La versión pública de GitHub/Streamlit Cloud **no se ha actualizado** con los cambios locales recientes. No debe suponerse equivalencia entre ambas.

## 1. Alcance y datos verificados

TEMPUS EXPLORATIO estudia el comportamiento histórico de cinco acciones y compara modelos **estadísticos** (no modelos de aprendizaje automático) con una evaluación temporal reproducible. La aplicación local funciona **solo con diez años**; ya no ofrece la selección de cinco años.

| Elemento | Valor |
|---|---|
| Activos | JNJ, MSFT, JPM, WMT, XOM |
| Fuente | Yahoo Finance, mediante `yfinance`; históricos guardados como snapshots Parquet |
| Corte | 30 de octubre de 2026 |
| Primera sesión observada | 31 de octubre de 2016 |
| Frecuencia | Una observación por **sesión bursátil diaria** |
| Registros verificados | **2.514 por activo · 12.570 en total** |
| Mercados/calendario de extrapolación | Sesiones NYSE, excluyendo jornadas no negociadas |
| Métodos | Ingenuo, suavizamiento exponencial simple (**SES**) y **ARIMA(1,1,1)** |
| Métricas | MAE, RMSE y sMAPE |
| Evaluación retrospectiva | **51 orígenes por activo y modelo**; paso de diez sesiones |
| Predicciones retrospectivas | **765** en total: 5 activos × 51 orígenes × 3 modelos |
| Estimaciones posteriores al corte | **300**: 5 activos × 20 sesiones × 3 modelos |
| Horizontes consultables | **1, 5, 10 y 20 sesiones bursátiles** |
| Incertidumbre | Intervalos condicionales al 95 % para ARIMA |
| Tema | Diurno (claro) por defecto |

**Diez años naturales no equivalen a 3.650 registros diarios**: se contabilizan únicamente sesiones con actividad bursátil. Las 765 y las 300 estimaciones son **salidas de los modelos**, no registros históricos adicionales.

Los resultados corresponden al archivo terminado el **30/10/2026**. «Pronóstico futuro» significa posterior al último cierre observado; **no necesariamente posterior a la fecha actual**. El proyecto tiene fines académicos y no constituye asesoramiento financiero.

## 2. Funcionalidades

- **Control de datos**: selección de empresa y fuente; snapshots locales versionados y descarga de históricos de diez años.
- **Precio e indicadores**: cierre y medias móviles MA20/MA50 (20 y 50 sesiones).
- **Rendimientos**: rendimientos diarios y distribución.
- **Datos y exportación**: tabla y CSV de los datos preparados.
- **Calidad**: fechas, duplicados, faltantes, rangos y coherencia de las observaciones.
- **Estadística temporal**: ADF, KPSS, ACF, PACF y otros diagnósticos.
- **Comparación multiserie**: índice base 100, correlaciones y comparación de activos.
- **Pronósticos estadísticos**: evaluación *walk-forward* y exportación de resultados.
- **Pronóstico futuro**: extrapolaciones para 1/5/10/20 sesiones y descargas.
- **Informe final**: consulta y descarga del PDF integral de diez años cuando existe una ejecución verificada.
- **Prueba integral**: auditoría de los snapshots, los cálculos y las evidencias, con tabla de resultados y visualizaciones. **No equivale** a automatizar clics de todos los controles Streamlit.
- **Páginas informativas**: Acerca de, Dataset, Fundamentos, Trivia, Ayuda e Institucional; la Ayuda incluye un ejemplo de comprobación con JNJ.

### Convenciones de visualización

| Tipo | Etiqueta recomendada del eje Y |
|---|---|
| Precio y pronósticos | `Precio (USD/acción)` |
| Rendimiento | `Rendimiento (%)` |
| Volatilidad | `Volatilidad anualizada (%)` |
| Volumen | `Acciones negociadas` |
| Comparación relativa | `Índice (base 100)` |
| Correlación | `Coeficiente de Pearson` |
| Eje temporal | `Fecha (sesiones bursátiles diarias)` |

Las unidades se aplican según corresponda; un gráfico de distribución o ACF/PACF tiene ejes propios. Los rezagos se expresan en **sesiones bursátiles**.

## 3. Requisitos e instalación local

- **Python 3.12** y conexión a internet **solo si se vuelven a adquirir cotizaciones**.
- Windows PowerShell; los comandos se ejecutan desde la raíz del proyecto.
- Dependencias fijadas o declaradas en `requirements.txt`.
- Los archivos `data/snapshots/` se mantienen locales, versionados y sin sobrescrituras silenciosas.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abrir **http://localhost:8501**. Para iniciar por defecto con tema diurno, mantener `.streamlit/config.toml` con:

```toml
[theme]
base = "light"
primaryColor = "#1767E8"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F3F6FB"
textColor = "#172033"
```

Si el navegador conserva una preferencia personal, seleccionar **Settings → Theme → Light**.

## 4. Datos históricos: adquisición y validación

Para actualizar los cinco activos con el corte académico indicado:

```powershell
.\.venv\Scripts\python.exe refresh_10y.py --until 2026-10-30
```

El flujo comprueba cobertura, coherencia OHLC y procedencia; produce snapshots Parquet y metadatos. **No hace falta repetir esta descarga** para regenerar un informe que use los snapshots ya validados.

La ejecución local confirmada de los cinco históricos fue:

```text
JNJ   2514 observaciones    2016-10-31 → 2026-10-30
MSFT  2514 observaciones    2016-10-31 → 2026-10-30
JPM   2514 observaciones    2016-10-31 → 2026-10-30
WMT   2514 observaciones    2016-10-31 → 2026-10-30
XOM   2514 observaciones    2016-10-31 → 2026-10-30
TOTAL 12570 observaciones
```

**Integridad**: verificar hashes SHA-256, rango, número de sesiones y coherencia entre símbolos antes de aceptar cualquier resultado. No presentar demostraciones sintéticas como datos reales.

## 5. Generación automática de las evidencias y el informe

El procesamiento por lotes **no necesita pulsar cinco veces los botones de la aplicación**. Separa adquisición, validación, indicadores, evaluación, extrapolación y construcción del informe.

```powershell
.\.venv\Scripts\python.exe run_10y_pipeline.py --snapshots data/snapshots --output-root reports --until 2026-10-30
```

El programa:

1. Lee los cinco Parquet verificados, **sin descargarlos de nuevo**.
2. Calcula indicadores y comparaciones normalizadas.
3. Ejecuta el *walk-forward* cronológico con modelos Ingenuo, SES y ARIMA(1,1,1).
4. Exporta 765 predicciones retrospectivas y sus métricas agregadas.
5. Estima 20 sesiones bursátiles para cada modelo y activo: 300 valores.
6. Produce CSV/JSON, contrato y PDF integral.
7. Comprueba archivos y hashes; escribe el `manifest.json` al cerrar correctamente.

**Ejecución real registrada el 12 de noviembre de 2026:**

```text
FINALIZADO: reports/integral_10y_20261112T042339296752Z
PDF: reports/integral_10y_20261112T042339296752Z/TEMPUS_EXPLORATIO_Informe_Integral_10_Anos.pdf
Total: 12570 registros; 765 predicciones retrospectivas
```

La línea `INFORME VERIFICADO` indica que el programa completó sus comprobaciones automáticas. **No sustituye la revisión académica o visual del documento**.

### 5.1. Análisis comparativo anterior a «PRONÓSTICOS FUTUROS»

La actualización local del generador incorpora la sección **«ANÁLISIS COMPARATIVO DE LAS CINCO ACCIONES»** antes de **«PRONÓSTICOS FUTUROS»**, con rendimiento histórico, volatilidad, caída máxima (*drawdown*) y comparación de modelos según las métricas existentes. Evita conclusiones especulativas.

Para reconstruir el PDF **sin recalcular ARIMA ni las 765 predicciones**, reutilizando el directorio original verificado:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_comparative_report.py -q
.\.venv\Scripts\python.exe rebuild_comparative_report.py --run-dir "reports\integral_10y_20261112T042339296752Z"
```

**Estado confirmado:** las cuatro pruebas del análisis comparativo pasaron. **No se ha confirmado aún la salida `FINALIZADO` de la reconstrucción en tu Windows**; por tanto, no se afirma que el PDF ampliado ya esté generado. El comando crea una nueva carpeta `integral_10y_<timestamp>`, conserva el informe original y actualiza el manifiesto de la nueva ejecución.

## 6. Prueba integral desde la app

En **Prueba integral**, ejecutar **«Ejecutar prueba integral»**. La sección comprueba los históricos, indicadores, diagnósticos y evidencias disponibles; presenta tabla de estado y gráficas con unidades explícitas.

**Ejemplo reproducible con JNJ:**

1. Cargar el snapshot local de JNJ; revisar **2.514 registros**.
2. Abrir precios, medias móviles, rendimientos, calidad y estadística temporal.
3. Ejecutar **Prueba integral** y verificar las cinco series (**12.570 registros**).
4. Revisar las visualizaciones y el estado de las evidencias.
5. En **Informe final**, abrir el PDF de diez años y comprobar el análisis comparativo antes de los pronósticos, una vez reconstruido.

**Alcance y límites:** esta prueba valida componentes de cálculo y archivos; **no realiza clics automáticos** sobre cada botón, descarga ni selector de la interfaz. Esos elementos aún requieren pruebas funcionales interactivas.

## 7. Pruebas y criterios de aceptación

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_integral_audit_ui.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_comparative_report.py -q
```

Los archivos de prueba específicos estarán disponibles si se instalaron sus respectivas ampliaciones.

**Para dar por aceptada la edición local:**

- Cinco históricos reales íntegros, con igual periodo y sesiones.
- 2.514 observaciones por empresa; 12.570 en total.
- Evidencias retrospectivas y futuras verificadas por el pipeline.
- PDF generado en un directorio versionado con `manifest.json`.
- Sección comparativa anterior a «PRONÓSTICOS FUTUROS», una vez reconstruido el PDF.
- Etiquetas y unidades claras, tema diurno, navegación y descargas comprobadas.
- Sin fallos en las pruebas automatizadas ni en los controles manuales exigidos.

Si falla una comprobación, **no debe presentarse la ejecución como íntegra**.

## 8. Organización principal

```text
TEMPUS EXPLORATIO/
├── app.py
├── README.md
├── requirements.txt
├── refresh_10y.py
├── run_10y_pipeline.py
├── rebuild_comparative_report.py
├── .streamlit/config.toml
├── assets/tempus-logo.png
├── tempus/
│   ├── core.py
│   ├── forecasting.py
│   ├── future.py
│   ├── history_period.py
│   ├── integral_audit.py
│   ├── integral_audit_ui.py
│   ├── comparative_report.py
│   ├── integrated_report.py
│   ├── info_pages.py
│   └── ... otros módulos
├── data/snapshots/            # archivos locales versionados
├── reports/
│   └── integral_10y_<timestamp>/
│       ├── evidence/
│       ├── future/
│       ├── contrato/
│       ├── manifest.json
│       └── TEMPUS_EXPLORATIO_Informe_Integral_10_Anos.pdf
└── tests/
```

La estructura exacta puede incluir más módulos, respaldos e informes previos. **No subir** entornos virtuales, secretos, respaldos, temporales ni snapshots sin revisar licencias y permisos de redistribución.

## 9. Reproducibilidad, supuestos y limitaciones

- Las métricas se refieren a **errores retrospectivos** en el periodo de evaluación, no a errores conocidos de los pronósticos futuros.
- El *walk-forward* toma **51 orígenes**, no todas las fechas posibles.
- Comparar sMAPE entre empresas no sustituye análisis de riesgo, cambios estructurales ni significancia estadística.
- Los intervalos condicionales de ARIMA **no garantizan cobertura real del 95 %** fuera de sus supuestos.
- Los cierres Yahoo Finance pueden incorporar ajustes o revisiones según la columna/fuente; conservar procedencia y fecha de adquisición.
- Datos bursátiles diarios y evaluación local **no constituyen computación distribuida Big Data**.
- Las versiones de cinco años se conservan únicamente como **historial archivado**, no como una opción de selección de la aplicación.
- No insertar métricas de cinco años en un informe de diez años.
- El análisis es **académico y descriptivo**, no recomendación de inversión.

## 10. Publicación

**No publicar todavía los cambios locales**. El orden acordado es ejecutar pruebas, inspeccionar gráficas e informe comparativo y **solo después** aprobar una actualización de GitHub/Streamlit Community Cloud.

- Repositorio público: https://github.com/edtech-mx-ve/tempus.exploratio
- Aplicación pública: https://tempus-exploratio.streamlit.app/

Estos enlaces se mantienen como referencia de la **versión previamente desplegada**.
