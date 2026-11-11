<div align="center">

# 📈 TEMPUS DECISIO — TEMPUS EXPLORATIO

### 🕰️ Módulo I · Análisis y pronóstico estadístico de series temporales financieras

**Adquisición web · Exploración temporal · Evaluación walk-forward · Pronósticos · Informe académico**

**🐍 Python 3.12 · 📊 Streamlit y Plotly · 🧮 pandas y statsmodels · 📄 PDF**

**[🌐 Aplicación web — pendiente de desplegar](https://share.streamlit.io/) · [📁 Repositorio GitHub](https://github.com/edtech-mx-ve/tempus.exploratio)**

</div>

---

## 1. 📍 Estado del proyecto

**TEMPUS EXPLORATIO** es el Módulo I de **TEMPUS DECISIO**. Es una aplicación académica de Streamlit que explora cinco series temporales bursátiles y compara tres métodos estadísticos para obtener estimaciones posteriores al último cierre.

| Elemento | Configuración o resultado |
|---|---|
| Versión preparada para la web | **0.5.6 · paquete mínimo** |
| Repositorio | `edtech-mx-ve/tempus.exploratio` |
| Empresa / símbolo | **JNJ, MSFT, JPM, WMT y XOM** |
| Fuente histórica declarada | Yahoo Finance, mediante `yfinance` |
| Corte de la evidencia incluida | **30 de octubre de 2026** |
| Observaciones | **1.255 por empresa · 6.275 en total** |
| Métodos estadísticos | Ingenuo, SES y ARIMA(1,1,1) |
| Evaluación retrospectiva | *Walk-forward* · 26 orígenes por empresa y método |
| Métricas | MAE, RMSE y sMAPE |
| Horizontes de pronóstico | 1, 5, 10 y 20 sesiones de NYSE |
| Incertidumbre | Intervalo condicional del 95 % para ARIMA |
| Informe | PDF integral generado **desde la aplicación** |
| Despliegue | **Pendiente de configurar y verificar** |

> Los pronósticos son posteriores al último registro **del conjunto seleccionado**, no necesariamente posteriores a la fecha de consulta. No son recomendaciones de inversión.

## 2. 🎯 Objetivo y alcance

Analizar cinco series financieras obtenidas de la web, validar sus datos, estudiar su evolución, contrastar métodos estadísticos mediante evaluación cronológica y comunicar estimaciones e incertidumbre con visualizaciones y un informe reproducible.

Se trabaja con un subconjunto histórico de tamaño manejable. **6.275 registros no constituyen por sí mismos una implementación distribuida de Big Data.**

## 3. 🖥️ Interfaz de la aplicación

La barra lateral permite seleccionar el activo y el origen de los datos, y acceder a **Acerca de, Dataset, Fundamentos, Trivia, Ayuda e Institucional**.

El análisis financiero cuenta con nueve secciones:

| Sección | Función |
|---|---|
| Precio e indicadores | Cierre y medias móviles |
| Rendimientos | Distribución y evolución de retornos |
| Datos y exportación | Tabla y descarga de datos preparados |
| Calidad | Verificación de la serie |
| Estadística temporal | Estadísticos, ADF/KPSS, ACF/PACF |
| Comparación multiserie | Normalización y correlaciones |
| Pronósticos estadísticos | Evaluación *walk-forward* |
| Pronóstico futuro | Extrapolación a sesiones bursátiles posteriores |
| **Informe final** | Evaluación y PDF académico integral |

Para iniciar una demostración sin descargar cotizaciones, elige **Histórico incluido** y pulsa **Cargar / adquirir datos**.

## 4. 🔁 Flujo metodológico

```text
Web / histórico archivado / snapshot
                 │
                 ▼
     Validación y normalización
                 │
                 ▼
       Estadística temporal
                 │
                 ▼
         Ingenuo · SES · ARIMA
                 │
                 ▼
    Evaluación cronológica walk-forward
                 │
                 ▼
         MAE · RMSE · sMAPE
                 │
                 ▼
      Pronóstico posterior al corte
                 │
                 ▼
     Gráficas · CSV/JSON · PDF
```

La evaluación retrospectiva compara estimaciones con precios ya observados. Los pronósticos posteriores al corte son otra operación: su error solo puede evaluarse cuando se conocen los cierres correspondientes.

## 5. 📚 Datos financieros

| Símbolo | Empresa | Sector |
|---|---|---|
| JNJ | Johnson & Johnson | Salud |
| MSFT | Microsoft | Tecnología |
| JPM | JPMorgan Chase | Banca |
| WMT | Walmart | Comercio minorista |
| XOM | ExxonMobil | Energía |

La evidencia incluida contiene cinco CSV históricos y un manifiesto que registra **SHA-256, cantidad de filas y fecha final**. Los archivos de evaluación y de pronósticos futuros son necesarios para la sección **Informe final**.

**Fuentes técnicas:** [Yahoo Finance](https://finance.yahoo.com/) · [yfinance](https://github.com/ranaroussi/yfinance). Los hashes confirman integridad de los archivos archivados, no sustituyen los registros independientes de adquisición ni acreditan derechos de redistribución.

## 6. 🧮 Métodos de pronóstico

| Método | Principio |
|---|---|
| **Ingenuo** | Repite el último cierre observado |
| **SES** | Suavizamiento exponencial simple |
| **ARIMA(1,1,1)** | Modelo autorregresivo integrado de media móvil |

La evaluación utiliza **26 fechas comunes por empresa y método**, con un conjunto de prueba cronológico equivalente al 20 % final del histórico y una separación de diez sesiones entre orígenes.

| Empresa | Menor MAE en la evaluación | MAE | RMSE | sMAPE |
|---|---|---:|---:|---:|
| JNJ | ARIMA(1,1,1) | 2,2388 | 2,8101 | 0,9299 % |
| MSFT | Ingenuo | 8,0918 | 11,9832 | 1,8202 % |
| JPM | Ingenuo | 3,4415 | 4,2202 | 1,0923 % |
| WMT | ARIMA(1,1,1) | 1,6399 | 2,6464 | 1,3953 % |
| XOM | Ingenuo | 2,4942 | 3,0042 | 1,6753 % |

Estas cifras se refieren a la evaluación histórica incluida; no demuestran superioridad estadística universal.

## 7. 🔮 Pronósticos y PDF integral

El módulo genera estimaciones para **1, 5, 10 y 20 sesiones bursátiles** después del último registro. Los límites condicionales del 95 % se calculan únicamente para ARIMA.

La página **Informe final** consume los CSV/JSON de evidencia ya incorporados y permite **generar y descargar** un PDF con el análisis histórico, la evaluación y las cinco secciones de pronóstico. **No se almacena un PDF pre-generado en este repositorio mínimo.**

## 8. 🏗️ Estructura mínima para Streamlit Cloud

```text
tempus.exploratio/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── tempus/
│   ├── __init__.py
│   ├── assets.py
│   ├── core.py
│   ├── demo.py
│   ├── forecasting.py
│   ├── future.py
│   ├── future_report.py
│   ├── info_pages.py
│   ├── integrated_report.py
│   ├── io.py
│   ├── public_data.py
│   ├── report_pdf.py
│   ├── reporting.py
│   ├── sidebar_info.py
│   └── statistics.py
├── evidence/
│   └── sprint03/
│       ├── public_manifest.json
│       └── ... CSV/JSON históricos y de evaluación
└── reports/
    └── sprint05_futuro/
        └── ... CSV/JSON de pronósticos
```

**Quedan fuera del repositorio web:** `.venv/`, `data/snapshots/`, PDF pre-generados, reportes locales, pruebas, respaldos, scripts CLI, guías de sprints y archivos temporales. La edición completa del proyecto se conserva en el entorno local.

## 9. 🐍 Ejecución local

Requiere Python **3.12**. Desde la raíz de este paquete en Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre `http://localhost:8501`. Selecciona **Histórico incluido** y carga una empresa para comprobar el funcionamiento sin conexión al proveedor de datos.

## 10. ☁️ Publicación en GitHub y Streamlit Community Cloud

El repositorio de destino es:

**https://github.com/edtech-mx-ve/tempus.exploratio**

Desde la carpeta donde se haya descomprimido **este paquete mínimo**:

```powershell
git init
git branch -M main
git add .
git status --short
git commit -m "deploy: Streamlit TEMPUS EXPLORATIO minimo"
git remote add origin https://github.com/edtech-mx-ve/tempus.exploratio.git
git push -u origin main
```

En [Streamlit Community Cloud](https://share.streamlit.io/), selecciona:

| Campo | Valor |
|---|---|
| Repositorio | `edtech-mx-ve/tempus.exploratio` |
| Rama | `main` |
| Archivo de ejecución | `app.py` |
| Versión Python | `3.12` |
| Dependencias | `requirements.txt` |

La URL pública **se incorporará después del despliegue y su comprobación**.

## 11. ✅ Comprobaciones y reproducibilidad

Antes de empaquetar esta edición mínima se verificaron, mediante el código de proyecto:

- Los **cinco históricos** y sus huellas SHA-256.
- Las **15 filas de métricas** de evaluación y sus evidencias.
- Los **cinco pronósticos de 20 sesiones** con los respectivos metadatos.
- La **generación del informe integral de 12 páginas** a partir únicamente de los archivos incluidos.

La visualización y el funcionamiento del servicio desplegado deberán comprobarse en Streamlit Community Cloud; esas verificaciones no se sustituyen por la prueba local.

## 12. 🔐 Seguridad y limitaciones

- No publicar credenciales, tokens, variables de entorno, `.streamlit/secrets.toml` ni archivos personales.
- La descarga de Yahoo Finance depende de la disponibilidad de un proveedor externo.
- El almacenamiento temporal del servidor de Streamlit **no debe tratarse como persistente**.
- Los pronósticos no incorporan eventos futuros, noticias ni cambios estructurales.
- Los intervalos ARIMA son condicionales al modelo.
- Antes de hacer público el repositorio, **verificar las condiciones de reutilización y redistribución de los datos financieros archivados**.

## 13. 🎓 Contexto académico y referencias

| Campo | Información |
|---|---|
| Institución | **Instituto Internacional de Aguascalientes** |
| Programa | **Maestría en Inteligencia Artificial para la Transformación Digital** |
| Asignatura | **Tópicos de IA para la Transformación Digital** |
| Alumno | **Antonio Nicolás Toro González** |
| Tutor | **Dr. Francisco Javier Luna Rosas** |
| Proyecto | **TEMPUS DECISIO — TEMPUS EXPLORATIO, Módulo I** |

**Herramientas:** [Python](https://www.python.org/) · [Streamlit](https://streamlit.io/) · [pandas](https://pandas.pydata.org/) · [statsmodels](https://www.statsmodels.org/) · [Plotly](https://plotly.com/python/) · [GitHub](https://github.com/).

---

<div align="center">

### 📈 TEMPUS DECISIO — TEMPUS EXPLORATIO

**📊 Cinco series → 🧮 Ingenuo / SES / ARIMA → 📈 Evaluación temporal → 🔮 Pronóstico → 📄 Informe**

*Práctica 1 · Módulo I · Instituto Internacional de Aguascalientes*

</div>
