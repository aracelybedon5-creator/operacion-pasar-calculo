# CHANGELOG

Registro de cambios significativos del proyecto Cálculo Vectorial 3D.

## [2.0.0] - Noviembre 17, 2025

### ✨ Nueva Funcionalidad Principal: Módulo de Optimización

**Gran actualización:** Integración completa de optimización multivariable con visualizaciones 3D/2D estilo GeoGebra.

#### Agregado
- **`optimizacion.py`**: Módulo completo de optimización multivariable (1800+ líneas)
  - `compute_gradient()`: Cálculo de gradiente simbólico y función numpy vectorizada
  - `directional_derivative()`: Derivada direccional con detección automática de direcciones de máximo/mínimo
  - `hessian_and_eig()`: Matriz Hessiana y valores propios para clasificación
  - `classify_critical_point()`: Clasificación automática (mínimo local, máximo local, punto silla, indeterminado)
  - `optimize_unconstrained()`: Resolución de ∇φ = 0 (simbólico y numérico con fallback)
  - `solve_lagrange()`: Multiplicadores de Lagrange para optimización con restricciones
  - `optimize_on_region()`: Optimización sobre regiones (triángulos, rectángulos, elipses)
  - `visualize_optimization_3d()`: Visualización 3D con superficie, gradiente y puntos críticos
  - `visualize_contour_2d()`: Visualización 2D con contornos, gradiente y región factible
  - `max_rectangle_in_ellipse()`: Caso especial pre-configurado
  - `cobb_douglas_optimization()`: Optimización Cobb-Douglas con restricción presupuestaria
  - `format_number_prefer_exact()`: Formateador que prefiere representaciones exactas (√2, ½, etc.)

- **Nueva pestaña en UI**: "📊 Optimización (Máximos/Mínimos)" con 6 sub-tabs:
  1. **Gradiente y Derivada Direccional**: Cálculo paso a paso con visualización
  2. **Puntos Críticos**: Encuentra y clasifica todos los puntos donde ∇φ = 0
  3. **Optimización Libre**: Redirección a Puntos Críticos
  4. **Multiplicadores de Lagrange**: Optimización con restricciones
  5. **Optimización en Regiones**: Análisis completo (interior + bordes + vértices)
  6. **Casos Especiales**: Problemas clásicos pre-configurados

- **`tests/test_optimizacion.py`**: Suite completa de tests pytest (300+ líneas)
  - 25+ tests cubriendo todas las funciones principales
  - Tests de casos extremos (función constante, función lineal, dirección cero)
  - Tests de integración (workflow completo)
  - Tests específicos:
    - `test_classify_saddle()`: Verifica punto silla en x² - y²
    - `test_optimize_triangle()`: Caso del quiz (triángulo con vértices)
    - `test_cobb_douglas()`: Verifica solución analítica
    - `test_max_rectangle_in_ellipse()`: x = a/√2, y = b/√2

#### Características Principales

**1. Representación Exacta de Números**
- Muestra √2, √10, fracciones (½, ⅓) en lugar de decimales cuando es posible
- Opción para toggle entre forma exacta y decimal
- Precisión de 8 decimales cuando se usa forma numérica

**2. Visualizaciones Estilo GeoGebra**
- Ejes con ticks numerados (no solo líneas)
- Rejilla y fondo claro
- Barra de color para magnitud del gradiente
- Marcadores con clasificación (azul=mínimo, rojo=máximo, amarillo=silla)
- Tooltips informativos con coordenadas y valores
- Cámara configurable con controles de rotación/zoom

**3. Pasos LaTeX Detallados**
- Cada función devuelve `latex_steps` con derivación completa
- Explicaciones textuales de clasificaciones
- Tablas comparativas de candidatos
- Sistema de ecuaciones mostrado paso a paso

**4. Manejo Robusto de Errores**
- Fallback numérico cuando falla resolución simbólica
- Múltiples puntos iniciales para métodos numéricos
- Validación de entrada (dirección no puede ser cero)
- Logging detallado para debugging

**5. Optimización en Regiones**
Procedimiento completo:
1. Buscar críticos interiores (∇φ = 0 dentro de región)
2. Parametrizar cada borde y resolver problemas 1D
3. Evaluar en todos los vértices
4. Comparar todos los candidatos
5. Determinar máximo/mínimo global

Soporta:
- Triángulos (con detección de pertenencia por coordenadas baricéntricas)
- Rectángulos (4 bordes lineales)
- Elipses (parametrización trigonométrica)
- Regiones implícitas (g(x,y) ≤ 0)

#### Integración con Código Existente

- **Reutiliza** `calc_vectorial.parse_expr_safe()` para parsing seguro
- **Mantiene** compatibilidad total con módulos existentes
- **No modifica** ninguna funcionalidad previa (solo agrega)
- **Sigue** los mismos patrones de session_state que otros módulos

#### Ejemplos Pre-Configurados

1. **Rectángulo inscrito en elipse**: Solución analítica x = a/√2, y = b/√2, A = 2ab
2. **Cobb-Douglas**: Maximizar x^α y^β sujeto a px·x + py·y = M
3. **Integral de línea**: F = (-y, x, 0) sobre círculo → 2π (con derivación completa)

#### Documentación

- Docstrings completos en español para todas las funciones
- Ejemplos de uso en cada docstring
- Descripción de parámetros y valores de retorno
- Explicación de algoritmos utilizados

#### Tests

Todos los tests pasan (pytest -v):
```
test_compute_gradient_simple ✓
test_classify_minimum ✓
test_classify_saddle ✓
test_optimize_unconstrained_simple ✓
test_lagrange_simple ✓
test_optimize_triangle ✓
test_max_rectangle_in_ellipse ✓
... (25+ tests)
```

### 🔧 Mejoras Técnicas

- **Caching**: Uso de `@lru_cache` para lambdify (evita recomputación)
- **Vectorización**: Todas las funciones numpy aceptan arrays y escalares
- **Tolerancias**: Configurables para detección de raíces/fracciones exactas
- **Performance**: Malla adaptativa (resolución configurable)

### 📝 Archivos Modificados

```
✓ optimizacion.py                 (NUEVO, 1827 líneas)
✓ tests/test_optimizacion.py      (NUEVO, 377 líneas)
✓ app_vectorial.py                (MODIFICADO, +800 líneas aprox)
✓ requirements.txt                (SIN CAMBIOS, ya tenía todas las deps)
✓ CHANGELOG.md                    (NUEVO, este archivo)
```

### 🎯 Casos de Uso Principales

**Caso 1: Estudiante necesita clasificar punto crítico**
```python
x, y = sp.symbols('x y')
result = classify_critical_point(x**2 - y**2, (x, y), (0, 0))
# → 'punto silla' con eigenvalues = [2, -2]
```

**Caso 2: Profesor verifica ejercicio del quiz**
```python
region = {'type': 'triangle', 'vertices': [(0,0), (0,8), (4,0)]}
result = optimize_on_region(f(x,y), (x,y), region)
# → máximo/mínimo global con tabla comparativa completa
```

**Caso 3: Investigador necesita optimizar con restricciones**
```python
result = solve_lagrange(x*y, (x,y), [x + y - 10])
# → solución (5, 5) con pasos LaTeX del Lagrangiano
```

### 🚀 Próximos Pasos Sugeridos

- [ ] Optimización con restricciones de desigualdad (Karush-Kuhn-Tucker)
- [ ] Métodos de descenso (gradiente conjugado, Newton)
- [ ] Optimización con múltiples objetivos (Pareto)
- [ ] Exportación de visualizaciones a PDF/PNG de alta calidad
- [ ] Animaciones de convergencia de algoritmos numéricos

---

## [1.0.0] - Noviembre 2025

### Versión Inicial

#### Agregado
- Módulo `calc_vectorial.py` con funciones core
- Módulo `viz_vectorial.py` para visualizaciones de campos vectoriales
- Módulo `viz_superficies.py` para visualizaciones de campos escalares
- Módulo `viz_curvas.py` para curvas y superficies paramétricas
- Aplicación Streamlit `app_vectorial.py` con 5 módulos:
  1. Campo Vectorial (∇·F, ∇×F)
  2. Gradiente de Campo Escalar (∇φ)
  3. Integral de Línea (∮ F·dr)
  4. Flujo de Superficie (∬ F·n dS)
  5. Verificación del Teorema de Stokes
- Sistema de session_state para persistencia de visualizaciones
- Helper `ensure_array()` para manejar resultados escalares de lambdify
- Visualización mejorada del integrando (áreas positivas/negativas separadas)
- Suite de tests pytest (23 tests)
- Documentación completa (README, CASOS_DE_PRUEBA, INSTRUCCIONES_GITHUB)

#### Corregido
- Error `'int' object has no attribute 'flatten'` (21+ aplicaciones de ensure_array)
- Visualizaciones que desaparecían al mover sliders (patrón session_state)
- Precisión del integrando con detección de signos y áreas

#### Documentado
- README.md con descripción del proyecto
- CASOS_DE_PRUEBA.md con 70+ casos organizados por dificultad
- INSTRUCCIONES_GITHUB.md con workflow completo de Git
- Docstrings en español en todos los módulos

---

## Formato del Changelog

Este changelog sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y el proyecto adhiere a [Versionado Semántico](https://semver.org/lang/es/).

### Tipos de Cambios
- **Agregado**: Nuevas funcionalidades
- **Modificado**: Cambios en funcionalidad existente
- **Obsoleto**: Funcionalidad que será removida
- **Eliminado**: Funcionalidad removida
- **Corregido**: Corrección de bugs
- **Seguridad**: Vulnerabilidades corregidas
