# 🧮 Aplicación Completa de Cálculo Vectorial Interactiva

## 🎯 Descripción del Proyecto

Aplicación web interactiva desarrollada en Python con **Streamlit** para resolver, visualizar y analizar problemas completos de **Cálculo Vectorial en 3D**. Incluye resolución paso a paso, visualizaciones interactivas 3D y exportación de resultados.

### ✨ Funcionalidades Principales

#### 1️⃣ **Campo Vectorial (∇·F, ∇×F)**
- Cálculo de Divergencia y Rotacional
- Visualización 3D de campos vectoriales
- Análisis combinado de campo, divergencia y rotacional

#### 2️⃣ **Gradiente en Campo Escalar (∇φ)**
- Gradiente simbólico y numérico
- Superficies de nivel interactivas
- Visualización del gradiente en 2D y 3D

#### 3️⃣ **Integral de Línea (∫ F·dr)**
- Cálculo simbólico paso a paso
- Gráfica del integrando F·(dr/dt) con precisión absoluta
- Visualización 3D de la curva con el campo vectorial
- Detección automática del caso rotacional clásico (-2π)

#### 4️⃣ **Flujo de Superficie (∬ F·n dS)**
- Cálculo de flujo con pasos detallados
- Visualización de superficie con vectores normales
- Superficies paramétricas personalizadas

#### 5️⃣ **Teorema de Stokes**
- Verificación ∮ F·dr = ∬ (∇×F)·n dS
- Visualización completa: superficie + frontera + campos
- Comparación lado a lado de ambas integrales

### 🎨 Características Técnicas

- **Visualización 3D interactiva** con Plotly (rotar, zoom, hover)
- **Cálculo simbólico** exacto con SymPy
- **Resolución paso a paso** en formato LaTeX
- **Exportación** de resultados en Markdown
- **Interfaz persistente**: Controles no reinician las gráficas
- **Casos de prueba** desde básicos hasta avanzados

---

## 🚀 Cómo Ejecutar la Aplicación

### Paso 1: Clonar o Descargar el Proyecto

```bash
# Si usas Git:
git clone <url-del-repositorio>
cd "Calculo vectorial/Version.1"

# O simplemente descarga la carpeta con los archivos
```

### Paso 2: Crear un Entorno Virtual (Recomendado)

**En Windows (PowerShell):**
```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\Activate.ps1
```

**En Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**En Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- `streamlit` - Framework web
- `plotly` - Gráficos 3D interactivos
- `numpy` - Cálculos numéricos
- `sympy` - Álgebra simbólica
- `scipy` - Integración numérica

### Paso 4: Ejecutar la Aplicación

```bash
streamlit run app_vectorial.py
```

La aplicación se abrirá automáticamente en tu navegador en:
```
http://localhost:8501
```

---

## 📖 Guía de Uso

### 1. Seleccionar una Curva

En la **barra lateral izquierda**, elige entre:
- Curvas predefinidas (Hélice, Lissajous, etc.)
- "Curva Personalizada" para definir tus propias ecuaciones

### 2. Ajustar Parámetros

Usa los **sliders** para modificar:
- Parámetros de la curva (A, B, C, a, b, c, δ)
- Rango de t (t₀ y t₁)
- Número de muestras N (calidad de la curva)
- Valor actual de t (punto donde calcular tangente/curvatura)

### 3. Visualizar Resultados

**Panel izquierdo (Gráfica 3D):**
- Curva completa en azul
- Punto actual en rojo
- Vectores tangente (verde), normal (naranja), binormal (púrpura)
- Usa el mouse para rotar la vista 3D

**Panel derecho (Datos):**
- Coordenadas del punto r(t)
- Vector tangente unitario T(t)
- Curvatura κ(t)
- Longitud de arco L(t)
- Velocidad ||r'(t)||

### 4. Opciones Adicionales

- **Animación**: Activa para ver el punto recorrer la curva
- **Proyecciones 2D**: Cambia a vista XY, XZ o YZ
- **Mostrar/Ocultar vectores**: Toggle para T, N, B

---

## 🧮 Conceptos Matemáticos Implementados

### Vector Tangente Unitario
```
T(t) = r'(t) / ||r'(t)||
```
Indica la dirección de movimiento en cada punto.

### Curvatura
```
κ(t) = ||r'(t) × r''(t)|| / ||r'(t)||³
```
Mide qué tan pronunciado es el giro de la curva.

### Longitud de Arco
```
L(t) = ∫[t₀ hasta t] ||r'(u)|| du
```
Distancia recorrida a lo largo de la curva.

### Triedro de Frenet
Sistema de coordenadas móvil:
- **T**: Tangente (dirección de movimiento)
- **N**: Normal (apunta hacia el centro de curvatura)
- **B**: Binormal (B = T × N)

---

## 🎓 Uso en Clase

Esta aplicación es ideal para:

1. **Visualizar conceptos abstractos** del cálculo vectorial
2. **Experimentar con diferentes curvas** y ver cómo cambian sus propiedades
3. **Verificar cálculos a mano** comparando con los resultados de la app
4. **Presentaciones** (pantalla completa, gráficos interactivos)
5. **Proyectos finales** de cursos de Cálculo III o Geometría Diferencial

---

## 🛠️ Estructura del Código

### `app_vectorial.py`
Aplicación principal de Streamlit con interfaz completa:
- Interfaz de usuario profesional
- Integración con calc_vectorial.py
- Visualizaciones 3D interactivas (Plotly)
- Exportación de informes PDF

### `calc_vectorial.py`
Módulo de cálculo vectorial seguro y vectorizado:
- Gradiente, divergencia, rotacional
- Integrales de línea y flujo de superficie
- Generador de ejercicios
- Parsing seguro (NO usa eval)

**Cada función está documentada** con docstrings completas y type hints.

### `requirements.txt`
Lista de dependencias con versiones compatibles.

### `README.md`
Este archivo con instrucciones completas.

---

## 🐛 Solución de Problemas

### Error: "streamlit: command not found"
```bash
# Asegúrate de activar el entorno virtual
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

### Error al parsear ecuaciones
- Usa sintaxis de Python: `**` para potencias, `exp()` para exponencial
- Ejemplos válidos: `cos(t)`, `t**2`, `exp(0.1*t)`, `A*sin(b*t)`

### La gráfica no se muestra
- Verifica que todas las dependencias estén instaladas
- Prueba con una curva predefinida primero
- Revisa la consola por errores+

---

## 📝 Personalización y Extensión

El código está diseñado para ser **fácil de extender**:

1. **Agregar nuevas funcionalidades**: Implementa funciones en `calc_vectorial.py` y agrégalas a `__all__`
2. **Cambiar visualizaciones**: Modifica las secciones de `plotly` en `app_vectorial.py`
3. **Agregar más ejercicios**: Extiende `generate_exercises()` con nuevos tipos
4. **Personalizar exportación PDF**: Modifica `export_report_pdf()` para nuevos formatos

---

## 👨‍💻 Tecnologías Utilizadas

- **Python 3.8+**
- **Streamlit** - Framework web interactivo
- **Plotly** - Gráficos 3D
- **Sympy** - Álgebra simbólica
- **NumPy** - Cálculo numérico
- **SciPy** - Integración numérica

---

## 📧 Contacto y Créditos

Proyecto desarrollado para el curso de **Cálculo Vectorial/Multivariable**.

**Generado con**: GitHub Copilot + Claude Sonnet 4.5  
**Fecha**: Noviembre 2025

---

## 📄 Licencia

Este proyecto es de uso educativo. Siéntete libre de modificarlo y adaptarlo a tus necesidades.

---

### 🎉 ¡Disfruta explorando el mundo del Cálculo Vectorial!

Si tienes preguntas o sugerencias, no dudes en contactarnos.
