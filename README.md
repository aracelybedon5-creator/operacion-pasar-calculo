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

#### 6️⃣ **📊 OPTIMIZACIÓN (Máximos/Mínimos)** ✨ NUEVO
- **Gradiente y Derivada Direccional**: Cálculo en un punto con dirección
  - Detección automática de dirección de máximo/mínimo crecimiento
  - Visualización con vectores y superficies
  - Valores exactos (√2, fracciones) cuando es posible
  
- **Puntos Críticos y Clasificación**: Encuentra y clasifica todos los puntos donde ∇φ = 0
  - Matriz Hessiana y valores propios
  - Clasificación automática: mínimo local, máximo local, punto silla
  - Visualización 3D con marcadores diferenciados por tipo
  
- **Multiplicadores de Lagrange**: Optimización con restricciones
  - Soporte para múltiples restricciones
  - Construcción automática del Lagrangiano
  - Resolución simbólica y numérica con fallback
  
- **Optimización en Regiones**: Análisis completo sobre regiones acotadas
  - Triángulos (vértices personalizables)
  - Rectángulos (límites configurables)
  - Elipses (semi-ejes y centro)
  - Procedimiento completo: interior + bordes + vértices
  - Tabla comparativa de todos los candidatos
  
- **Casos Especiales Pre-Configurados**:
  - Rectángulo inscrito en elipse (solución analítica: x = a/√2, y = b/√2)
  - Cobb-Douglas con restricción presupuestaria
  - Integral de línea F·dr (círculo unitario → 2π)

**Visualizaciones estilo GeoGebra**:
- Ejes con ticks numerados
- Curvas de nivel con etiquetas
- Campo de gradiente con flechas (go.Cone)
- Puntos críticos marcados con colores (azul=mínimo, rojo=máximo, amarillo=silla)
- Controles interactivos (rotación, zoom, pan)
- Tooltips informativos

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

### Módulos Principales

#### `app_vectorial.py` (2900+ líneas)
Aplicación principal de Streamlit con interfaz completa:
- 6 módulos principales (Campo Vectorial, Gradiente, Integral de Línea, Flujo, Stokes, **Optimización**)
- Interfaz de usuario profesional con session_state
- Integración con todos los módulos de cálculo
- Visualizaciones 3D interactivas (Plotly)
- Exportación de informes

#### `calc_vectorial.py` (650+ líneas)
Módulo de cálculo vectorial seguro y vectorizado:
- Gradiente, divergencia, rotacional
- Integrales de línea y flujo de superficie
- Generador de ejercicios
- Parsing seguro (NO usa eval, solo whitelist de funciones)

#### `optimizacion.py` ✨ NUEVO (1800+ líneas)
Módulo completo de optimización multivariable:
- `compute_gradient()`: Gradiente simbólico y función numpy
- `directional_derivative()`: Derivada direccional con análisis
- `hessian_and_eig()`: Hessiana y valores propios
- `classify_critical_point()`: Clasificación automática de puntos críticos
- `optimize_unconstrained()`: Resolución de ∇φ = 0
- `solve_lagrange()`: Multiplicadores de Lagrange
- `optimize_on_region()`: Optimización en regiones (triángulos, rectángulos, elipses)
- `visualize_optimization_3d()`: Visualizaciones estilo GeoGebra
- `visualize_contour_2d()`: Contornos con gradiente y región
- Casos especiales: rectángulo en elipse, Cobb-Douglas

#### `viz_vectorial.py`, `viz_superficies.py`, `viz_curvas.py`
Módulos de visualización especializados:
- Campos vectoriales 3D con flechas
- Superficies y curvas de nivel
- Integrando con áreas positivas/negativas
- Helper `ensure_array()` para compatibilidad

**Cada función está documentada** con docstrings completas en español y type hints.

### Tests

#### `tests/test_optimizacion.py` ✨ NUEVO (377 líneas)
Suite completa de tests pytest para optimización:
- 25+ tests cubriendo todas las funciones
- Tests de casos extremos y edge cases
- Tests de integración (workflows completos)
- Verificación de soluciones analíticas conocidas

**Ejecutar tests:**
```bash
# Todos los tests
pytest tests/ -v

# Solo optimización
pytest tests/test_optimizacion.py -v

# Con coverage
pytest tests/ --cov=optimizacion --cov-report=html
```

Ejemplo de salida esperada:
```
test_compute_gradient_simple ✓
test_classify_minimum ✓
test_classify_saddle ✓
test_optimize_triangle ✓
test_cobb_douglas ✓
test_max_rectangle_in_ellipse ✓
... (25+ tests)
========================= 25 passed in 2.5s =========================
```

### Archivos de Documentación

#### `requirements.txt`
Lista de dependencias con versiones compatibles.

#### `README.md`
Este archivo con instrucciones completas.

#### `CHANGELOG.md` ✨ NUEVO
Registro detallado de todos los cambios del proyecto.

#### `CASOS_DE_PRUEBA.md`
70+ casos de prueba organizados por dificultad para todas las funcionalidades.

#### `INSTRUCCIONES_GITHUB.md`
Guía completa de Git/GitHub para colaboración.

---

## 🧪 Testing y Calidad

### Cobertura de Tests

**Módulo de Optimización:**
- Gradiente: 4 tests
- Clasificación de puntos: 3 tests  
- Optimización sin restricciones: 2 tests
- Lagrange: 3 tests
- Regiones: 3 tests
- Casos especiales: 2 tests
- Formato exacto: 3 tests
- Visualización: 2 tests
- Integración: 3 tests

**Resultados esperados:**
- ✅ Todos los tests pasan
- ✅ Sin warnings críticos
- ✅ Cobertura >80% en optimizacion.py

### Validación Manual

Casos especiales con soluciones conocidas:
1. **Punto silla en x² - y²**: eigenvalues = [2, -2]
2. **Rectángulo en elipse**: x = a/√2, y = b/√2
3. **Cobb-Douglas α=0.5, β=0.5, px=150, py=250, M=50000**: x* ≈ 166.67, y* ≈ 100
4. **Círculo unitario rotacional**: ∫ F·dr = 2π

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
