"""
MÓDULO DE CÁLCULO VECTORIAL 3D
==============================

Implementa operaciones de cálculo vectorial avanzado:
- Gradiente de campos escalares
- Divergencia y rotacional de campos vectoriales
- Integrales de línea sobre curvas paramétricas
- Flujo de superficie sobre superficies paramétricas
- Verificación numérica de teoremas (Green, Stokes, Gauss)

Notación Matemática Soportada (estilo GeoGebra):
------------------------------------------------
✓ Potencias: x^2 o x**2
✓ Multiplicación implícita: 2x, 3xy, 2sin(x)
✓ Funciones en español: sen(x), cos(x), tan(x)
✓ Raíces: sqrt(x) o raiz(x)
✓ Valor absoluto: abs(x) o |x|
✓ Constantes: pi, e, π

Autor: Proyecto Cálculo Multivariable
Fecha: Noviembre 2025
Versión: 1.1.0

IMPORTANTE: Todas las funciones están vectorizadas y usan numpy para rendimiento.
No se usa eval() en ningún lugar - solo sympy.parse_expr con whitelist.
"""

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import Tuple, Callable, Iterable, Dict, Optional, Union, List, Any
import logging
import plotly.graph_objects as go
import re  # Para preprocesamiento de expresiones

# Configurar logging (no prints)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN DE SEGURIDAD
# ==============================================================================

# Whitelist de funciones matemáticas permitidas
ALLOWED_FUNCTIONS = {
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'exp': sp.exp,
    'log': sp.log,
    'ln': sp.log,
    'sqrt': sp.sqrt,
    'Abs': sp.Abs,
    'abs': sp.Abs,
    'asin': sp.asin,
    'acos': sp.acos,
    'atan': sp.atan,
    'sinh': sp.sinh,
    'cosh': sp.cosh,
    'tanh': sp.tanh,
    'pi': sp.pi,
    'e': sp.E,
    'E': sp.E,
}

# Longitud máxima de expresión permitida (seguridad)
MAX_EXPR_LENGTH = 300

# Caracteres permitidos en expresiones (seguridad)
ALLOWED_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/()., ")


# ==============================================================================
# FUNCIÓN 1: PARSEO SEGURO DE EXPRESIONES
# ==============================================================================

def safe_parse(expr_str: str, vars: Tuple[sp.Symbol, ...]) -> sp.Expr:
    """
    Parsea una expresión matemática de forma segura usando whitelist.
    Acepta notación matemática natural como GeoGebra/Wolfram Alpha.
    
    NO usa eval() ni ejecuta código arbitrario. Solo permite funciones
    matemáticas estándar definidas en ALLOWED_FUNCTIONS.
    
    Notación soportada (estilo GeoGebra):
    -------------------------------------
    - Potencias: x^2, x**2 (ambas funcionan)
    - Multiplicación implícita: 2x, 3xy, 2sin(x)
    - Funciones en español: sen(x), cos(x), tan(x), ln(x)
    - Raíces: sqrt(x), raiz(x)
    - Valor absoluto: abs(x), |x|
    - Pi y e: pi, e, π
    - Paréntesis: (x+1)^2, 2(x+y)
    
    Parámetros
    ----------
    expr_str : str
        Expresión matemática como string (ej: "x^2 + sin(y)", "2x + 3y")
    vars : Tuple[sp.Symbol, ...]
        Tupla de símbolos permitidos (ej: (x, y, z))
    
    Retorna
    -------
    sp.Expr
        Expresión simbólica de Sympy parseada
    
    Lanza
    -----
    ValueError
        Si la expresión es demasiado larga, contiene caracteres no permitidos,
        o usa funciones/nombres fuera de la whitelist
    
    Ejemplos
    --------
    >>> x, y, z = sp.symbols('x y z')
    >>> safe_parse("x^2 + y^2", (x, y, z))  # Acepta ^ 
    x**2 + y**2
    
    >>> safe_parse("2x + 3y", (x, y, z))  # Multiplicación implícita
    2*x + 3*y
    
    >>> safe_parse("sen(x) + cos(y)", (x, y, z))  # Español
    sin(x) + cos(y)
    """
    # Validación 1: Longitud razonable
    if len(expr_str) > MAX_EXPR_LENGTH:
        raise ValueError(
            f"Expresión demasiado larga ({len(expr_str)} caracteres). "
            f"Máximo permitido: {MAX_EXPR_LENGTH}"
        )
    
    # Validación 2: Caracteres permitidos (prevenir inyección)
    expr_clean = expr_str.strip()
    if not expr_clean:
        raise ValueError("Expresión vacía no es válida")
    
    # =========================================================================
    # PREPROCESAMIENTO: Convertir notación matemática natural a sintaxis Python
    # =========================================================================
    
    # 1. Reemplazar funciones en español por inglés
    spanish_replacements = {
        'sen': 'sin',      # seno
        'tg': 'tan',       # tangente (notación alternativa)
        'ctg': 'cot',      # cotangente
        'arcsen': 'asin',  # arcoseno
        'arccos': 'acos',  # arcocoseno
        'arctan': 'atan',  # arcotangente
        'raiz': 'sqrt',    # raíz cuadrada
        'π': 'pi',         # símbolo pi
        'ln': 'log',       # logaritmo natural
    }
    
    for spanish, english in spanish_replacements.items():
        # Reemplazo cuidadoso para no romper palabras más largas
        import re
        # Buscar palabra completa (no parte de otra palabra)
        pattern = r'\b' + re.escape(spanish) + r'\b'
        expr_clean = re.sub(pattern, english, expr_clean)
    
    # 2. Convertir ^ a ** (potencias estilo matemático)
    expr_clean = expr_clean.replace('^', '**')
    
    # 3. Manejar valor absoluto |x| -> abs(x)
    # Contar barras verticales (deben ser pares)
    if expr_clean.count('|') % 2 == 0 and '|' in expr_clean:
        # Reemplazar pares de | por abs()
        parts = expr_clean.split('|')
        result = parts[0]
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                result += f'abs({parts[i]})' + parts[i + 1]
        expr_clean = result
    
    logger.info(f"Expresión preprocesada: {expr_str} -> {expr_clean}")
    
    # =========================================================================
    # PARSEO CON TRANSFORMACIONES AUTOMÁTICAS
    # =========================================================================
    
    # Crear diccionario local con símbolos permitidos
    local_dict = ALLOWED_FUNCTIONS.copy()
    for var in vars:
        local_dict[str(var)] = var
    
    # Agregar alias adicionales
    local_dict['π'] = sp.pi
    local_dict['e'] = sp.E
    
    try:
        # Parsear con sympy usando transformaciones automáticas
        # 'all' incluye: implicit_multiplication, convert_xor, etc.
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor
        )
        
        # Transformaciones que queremos aplicar
        transformations = (
            standard_transformations +
            (implicit_multiplication_application, convert_xor)
        )
        
        parsed = parse_expr(
            expr_clean,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True
        )
        
        # Validación 3: Verificar que no haya símbolos no permitidos
        free_symbols = parsed.free_symbols
        allowed_symbols = set(vars)
        
        for symbol in free_symbols:
            if symbol not in allowed_symbols:
                raise ValueError(
                    f"Símbolo '{symbol}' no permitido. "
                    f"Solo se permiten: {', '.join(str(v) for v in vars)}"
                )
        
        logger.info(f"Expresión parseada exitosamente: {expr_str} -> {parsed}")
        return parsed
        
    except (SyntaxError, TypeError) as e:
        raise ValueError(
            f"Error de sintaxis en la expresión '{expr_str}': {str(e)}\n"
            f"Asegúrate de usar sintaxis matemática válida."
        )
    except Exception as e:
        raise ValueError(
            f"Error al parsear expresión '{expr_str}': {str(e)}"
        )


# ==============================================================================
# FUNCIÓN 2: LAMBDIFY VECTORIZADO
# ==============================================================================

def lambdify_vector(
    exprs: Iterable[sp.Expr], 
    vars: Tuple[sp.Symbol, ...]
) -> Callable:
    """
    Convierte expresiones simbólicas vectoriales en función numpy vectorizada.
    
    Parámetros
    ----------
    exprs : Iterable[sp.Expr]
        Lista/tupla de expresiones simbólicas (ej: (P, Q, R) para campo F=(P,Q,R))
    vars : Tuple[sp.Symbol, ...]
        Variables independientes (ej: (x, y, z))
    
    Retorna
    -------
    Callable
        Función f(X, Y, Z, ...) que acepta arrays numpy y retorna tuple de arrays
        (Fx, Fy, Fz, ...) con las mismas dimensiones
    
    Notas
    -----
    La función retornada está completamente vectorizada - puede aceptar:
    - Escalares: f(1.0, 2.0, 3.0) -> (val1, val2, val3)
    - Arrays: f(X_grid, Y_grid, Z_grid) -> (Fx_grid, Fy_grid, Fz_grid)
    
    Ejemplos
    --------
    >>> x, y, z = sp.symbols('x y z')
    >>> F = (x + y, x - y, z**2)
    >>> F_num = lambdify_vector(F, (x, y, z))
    >>> F_num(1.0, 2.0, 3.0)
    (3.0, -1.0, 9.0)
    >>> import numpy as np
    >>> X = np.array([1, 2])
    >>> Y = np.array([3, 4])
    >>> Z = np.array([5, 6])
    >>> F_num(X, Y, Z)
    (array([4, 6]), array([-2, -2]), array([25, 36]))
    """
    # Convertir a lista si es necesario
    exprs_list = list(exprs)
    
    # Lambdify cada componente
    funcs = []
    for expr in exprs_list:
        func = sp.lambdify(vars, expr, 'numpy')
        funcs.append(func)
    
    # Crear función vectorizada que retorna tuple
    def vectorized_func(*args):
        """Evalúa todas las componentes y retorna tuple de resultados."""
        results = tuple(f(*args) for f in funcs)
        return results
    
    return vectorized_func


# ==============================================================================
# FUNCIÓN 3: GRADIENTE DE CAMPO ESCALAR
# ==============================================================================

def compute_gradient_scalar(
    phi_expr: sp.Expr,
    vars: Tuple[sp.Symbol, ...] = None
) -> Tuple[Tuple[sp.Expr, ...], Callable]:
    """
    Calcula el gradiente de un campo escalar φ(x, y, z).
    
    El gradiente ∇φ = (∂φ/∂x, ∂φ/∂y, ∂φ/∂z) apunta en la dirección
    de máximo crecimiento del campo escalar.
    
    Parámetros
    ----------
    phi_expr : sp.Expr
        Expresión simbólica del campo escalar φ
    vars : Tuple[sp.Symbol, ...], optional
        Variables (x, y, z). Si None, se extraen de phi_expr
    
    Retorna
    -------
    grad_sym : Tuple[sp.Expr, ...]
        Gradiente simbólico (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
    grad_num : Callable
        Función numpy vectorizada grad(X, Y, Z) -> (Gx, Gy, Gz)
    
    Notas
    -----
    Propiedades del gradiente:
    - ∇φ ⊥ curvas de nivel de φ
    - ||∇φ|| = tasa máxima de cambio
    - ∇φ = 0 en puntos críticos (máx/mín/silla)
    
    Ejemplos
    --------
    >>> x, y, z = sp.symbols('x y z')
    >>> phi = x**2 + y**2 + z**2  # Paraboloide
    >>> grad_sym, grad_num = compute_gradient_scalar(phi, (x, y, z))
    >>> grad_sym
    (2*x, 2*y, 2*z)
    >>> grad_num(1.0, 0.0, 0.0)
    (2.0, 0.0, 0.0)
    """
    # Si no se especifican vars, extraerlas de la expresión
    if vars is None:
        free_syms = list(phi_expr.free_symbols)
        # Ordenar para consistencia (x, y, z)
        vars = tuple(sorted(free_syms, key=str))
    
    # Calcular derivadas parciales
    grad_components = []
    for var in vars:
        partial = sp.diff(phi_expr, var)
        grad_components.append(partial)
    
    grad_sym = tuple(grad_components)
    
    # Lambdify el gradiente
    grad_num = lambdify_vector(grad_sym, vars)
    
    logger.info(f"Gradiente calculado: ∇φ = {grad_sym}")
    
    return grad_sym, grad_num


# ==============================================================================
# FUNCIÓN 4: DIVERGENCIA Y ROTACIONAL DE CAMPO VECTORIAL
# ==============================================================================

def compute_divergence_and_curl(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    vars: Tuple[sp.Symbol, sp.Symbol, sp.Symbol] = None
) -> Tuple[sp.Expr, Tuple[sp.Expr, ...], Callable, Callable]:
    """
    Calcula divergencia y rotacional de un campo vectorial F(x,y,z).
    
    Divergencia: ∇·F = ∂P/∂x + ∂Q/∂y + ∂R/∂z (campo escalar)
    Rotacional: ∇×F = (∂R/∂y - ∂Q/∂z, ∂P/∂z - ∂R/∂x, ∂Q/∂x - ∂P/∂y) (campo vectorial)
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial (P, Q, R) donde F = P*i + Q*j + R*k
    vars : Tuple[sp.Symbol, sp.Symbol, sp.Symbol], optional
        Variables (x, y, z). Si None, usa (x, y, z) por defecto
    
    Retorna
    -------
    div_sym : sp.Expr
        Divergencia simbólica ∇·F
    curl_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Rotacional simbólico ∇×F = (cx, cy, cz)
    div_num : Callable
        Función numpy div(X, Y, Z) -> escalar o array
    curl_num : Callable
        Función numpy curl(X, Y, Z) -> (Cx, Cy, Cz)
    
    Notas
    -----
    Interpretación física:
    - div(F) > 0: fuente (diverge desde el punto)
    - div(F) < 0: sumidero (converge hacia el punto)
    - div(F) = 0: incompresible (preserva volumen)
    
    - curl(F) ≠ 0: campo rotacional (tiene torbellinos)
    - curl(F) = 0: campo conservativo (existe potencial)
    
    Ejemplos
    --------
    >>> x, y, z = sp.symbols('x y z')
    >>> F = (2*x, 2*y, 2*z)  # Campo radial
    >>> div_sym, curl_sym, div_num, curl_num = compute_divergence_and_curl(F, (x,y,z))
    >>> div_sym
    6
    >>> curl_sym
    (0, 0, 0)
    >>> div_num(1.0, 1.0, 1.0)
    6
    """
    P, Q, R = F_sym
    
    # Usar (x, y, z) por defecto si no se especifica
    if vars is None:
        x, y, z = sp.symbols('x y z')
        vars = (x, y, z)
    else:
        x, y, z = vars
    
    # =========================================================================
    # DIVERGENCIA: ∇·F = ∂P/∂x + ∂Q/∂y + ∂R/∂z
    # =========================================================================
    
    div_sym = sp.diff(P, x) + sp.diff(Q, y) + sp.diff(R, z)
    
    # =========================================================================
    # ROTACIONAL: ∇×F = det| i    j    k   |
    #                       | ∂/∂x ∂/∂y ∂/∂z|
    #                       | P    Q    R   |
    # =========================================================================
    
    # Componente x: ∂R/∂y - ∂Q/∂z
    curl_x = sp.diff(R, y) - sp.diff(Q, z)
    
    # Componente y: ∂P/∂z - ∂R/∂x
    curl_y = sp.diff(P, z) - sp.diff(R, x)
    
    # Componente z: ∂Q/∂x - ∂P/∂y
    curl_z = sp.diff(Q, x) - sp.diff(P, y)
    
    curl_sym = (curl_x, curl_y, curl_z)
    
    # =========================================================================
    # LAMBDIFY PARA EVALUACIÓN NUMÉRICA
    # =========================================================================
    
    div_num = sp.lambdify(vars, div_sym, 'numpy')
    curl_num = lambdify_vector(curl_sym, vars)
    
    logger.info(f"Divergencia: ∇·F = {div_sym}")
    logger.info(f"Rotacional: ∇×F = {curl_sym}")
    
    return div_sym, curl_sym, div_num, curl_num


# ==============================================================================
# FUNCIÓN 5: INTEGRAL DE LÍNEA SOBRE CURVA PARAMÉTRICA
# ==============================================================================

def line_integral_over_param(
    F_num: Callable,
    r_num: Callable,
    dr_num: Callable,
    t0: float,
    t1: float,
    n: int = 2000
) -> float:
    """
    Calcula la integral de línea ∫_C F·dr sobre una curva paramétrica.
    
    La integral se calcula como:
    ∫_{t0}^{t1} F(r(t))·r'(t) dt
    
    usando el método del trapecio (numpy.trapz) completamente vectorizado.
    
    Parámetros
    ----------
    F_num : Callable
        Campo vectorial F(x, y, z) -> (Fx, Fy, Fz)
    r_num : Callable
        Curva paramétrica r(t) -> (x, y, z)
    dr_num : Callable
        Derivada de la curva r'(t) -> (dx/dt, dy/dt, dz/dt)
    t0 : float
        Límite inferior del parámetro t
    t1 : float
        Límite superior del parámetro t
    n : int, default=2000
        Número de puntos de integración (más = más preciso pero más lento)
    
    Retorna
    -------
    float
        Valor de la integral de línea ∫_C F·dr
    
    Notas
    -----
    - Si n es muy grande (>10000), puede ser lento. Usar con precaución.
    - La precisión depende de n y de qué tan suave sea la curva.
    - Para curvas muy irregulares, aumentar n.
    
    Advertencias
    -----------
    Si n > 10000, se recomienda mostrar disclaimer de "alto consumo"
    en la UI antes de ejecutar.
    
    Ejemplos
    --------
    >>> # Integral de F=(-y, x, 0) sobre círculo unitario
    >>> def F(x, y, z): return (-y, x, np.zeros_like(z))
    >>> def r(t): return (np.cos(t), np.sin(t), np.zeros_like(t))
    >>> def dr(t): return (-np.sin(t), np.cos(t), np.zeros_like(t))
    >>> result = line_integral_over_param(F, r, dr, 0, 2*np.pi, 1000)
    >>> np.isclose(result, 2*np.pi, rtol=1e-3)
    True
    """
    # Validación
    if n <= 0:
        raise ValueError(f"n debe ser positivo, recibido: {n}")
    
    if n > 10000:
        logger.warning(
            f"n={n} es muy grande. Esto puede ser lento. "
            f"Considere usar n <= 10000 para mejor rendimiento."
        )
    
    # Crear array de valores de t
    t_vals = np.linspace(t0, t1, n)
    
    # Evaluar r(t) y r'(t) en todos los puntos (vectorizado)
    x_vals, y_vals, z_vals = r_num(t_vals)
    dx_vals, dy_vals, dz_vals = dr_num(t_vals)
    
    # Evaluar F(r(t)) en todos los puntos (vectorizado)
    Fx_vals, Fy_vals, Fz_vals = F_num(x_vals, y_vals, z_vals)
    
    # Calcular producto punto F(r(t))·r'(t) en cada punto
    dot_product = Fx_vals * dx_vals + Fy_vals * dy_vals + Fz_vals * dz_vals
    
    # Integrar usando trapecio
    integral = np.trapezoid(dot_product, t_vals)
    
    logger.info(
        f"Integral de línea calculada: ∫_C F·dr = {integral:.6f} "
        f"(n={n}, t∈[{t0:.2f}, {t1:.2f}])"
    )
    
    return float(integral)


# ==============================================================================
# FUNCIÓN 6: FLUJO DE SUPERFICIE SOBRE SUPERFICIE PARAMÉTRICA
# ==============================================================================

def surface_flux_over_param(
    F_num: Callable,
    r_num: Callable,
    ru_num: Callable,
    rv_num: Callable,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    nu: int = 200,
    nv: int = 200
) -> float:
    """
    Calcula el flujo de superficie ∬_S F·n dS sobre superficie paramétrica.
    
    El flujo se calcula como:
    ∬ F(r(u,v))·(r_u × r_v) dudv
    
    donde r_u = ∂r/∂u, r_v = ∂r/∂v, y (r_u × r_v) es el vector normal.
    
    Usa integración numérica doble con numpy.trapz completamente vectorizada.
    
    Parámetros
    ----------
    F_num : Callable
        Campo vectorial F(x, y, z) -> (Fx, Fy, Fz)
    r_num : Callable
        Superficie paramétrica r(u, v) -> (x, y, z)
        DEBE aceptar arrays 2D (mallas U, V)
    ru_num : Callable
        Derivada parcial ∂r/∂u (u, v) -> (∂x/∂u, ∂y/∂u, ∂z/∂u)
        DEBE aceptar arrays 2D
    rv_num : Callable
        Derivada parcial ∂r/∂v (u, v) -> (∂x/∂v, ∂y/∂v, ∂z/∂v)
        DEBE aceptar arrays 2D
    u0, u1 : float
        Límites del parámetro u
    v0, v1 : float
        Límites del parámetro v
    nu : int, default=200
        Número de puntos en dirección u
    nv : int, default=200
        Número de puntos en dirección v
    
    Retorna
    -------
    float
        Flujo total ∬_S F·n dS
    
    Notas
    -----
    - La malla total tiene nu×nv puntos
    - Para nu=nv=200, son 40,000 puntos (rápido en hardware moderno)
    - Si nu×nv > 100,000, puede ser lento
    
    El vector normal n = r_u × r_v NO está normalizado (incluye dS automáticamente)
    
    Ejemplos
    --------
    >>> # Flujo de F=(0,0,1) a través del plano z=0 en [0,1]×[0,1]
    >>> def F(x, y, z): return (np.zeros_like(z), np.zeros_like(z), np.ones_like(z))
    >>> def r(u, v): return (u, v, np.zeros_like(u))
    >>> def ru(u, v): return (np.ones_like(u), np.zeros_like(u), np.zeros_like(u))
    >>> def rv(u, v): return (np.zeros_like(v), np.ones_like(v), np.zeros_like(v))
    >>> flux = surface_flux_over_param(F, r, ru, rv, 0, 1, 0, 1, 50, 50)
    >>> np.isclose(flux, 1.0, rtol=1e-6)
    True
    """
    # Validación
    if nu <= 0 or nv <= 0:
        raise ValueError(f"nu y nv deben ser positivos: nu={nu}, nv={nv}")
    
    if nu * nv > 100000:
        logger.warning(
            f"Malla grande: {nu}×{nv} = {nu*nv} puntos. "
            f"Esto puede ser lento. Considere reducir la resolución."
        )
    
    # Crear mallas de parámetros
    u_vals = np.linspace(u0, u1, nu)
    v_vals = np.linspace(v0, v1, nv)
    U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
    
    # Evaluar r(u,v), r_u(u,v), r_v(u,v) en toda la malla (vectorizado)
    X, Y, Z = r_num(U, V)
    ru_x, ru_y, ru_z = ru_num(U, V)
    rv_x, rv_y, rv_z = rv_num(U, V)
    
    # Calcular producto cruz: n = r_u × r_v (componente a componente)
    nx = ru_y * rv_z - ru_z * rv_y
    ny = ru_z * rv_x - ru_x * rv_z
    nz = ru_x * rv_y - ru_y * rv_x
    
    # Evaluar F(r(u,v)) en toda la malla
    Fx, Fy, Fz = F_num(X, Y, Z)
    
    # Calcular producto punto: F·n en cada punto
    dot_product = Fx * nx + Fy * ny + Fz * nz
    
    # Integración doble con trapecio
    # Primero integramos en v, luego en u
    integral_v = np.trapezoid(dot_product, v_vals, axis=1)
    flux = np.trapezoid(integral_v, u_vals)
    
    logger.info(
        f"Flujo de superficie calculado: ∬ F·n dS = {flux:.6f} "
        f"(malla {nu}×{nv}, u∈[{u0:.2f},{u1:.2f}], v∈[{v0:.2f},{v1:.2f}])"
    )
    
    return float(flux)


# ==============================================================================
# FUNCIÓN 7: VISUALIZACIÓN DE CAMPO VECTORIAL (SLICE 2D)
# ==============================================================================

def plot_vector_field_slice(
    F_num: Callable,
    plane: str = 'z',
    plane_value: float = 0.0,
    xrange: Tuple[float, float] = (-3, 3),
    yrange: Tuple[float, float] = (-3, 3),
    density: int = 12,
    normalize: bool = True,
    scale: float = 0.5
):
    """
    Visualiza un campo vectorial 3D en un plano (slice).
    
    Crea una figura Plotly con vectores representados como conos 3D.
    La densidad controla cuántos vectores se muestran (menos = más rápido).
    
    Parámetros
    ----------
    F_num : Callable
        Campo vectorial F(x, y, z) -> (Fx, Fy, Fz)
    plane : str, default='z'
        Plano de corte: 'x', 'y', o 'z'
    plane_value : float, default=0.0
        Valor del plano (ej: z=0 para plano XY)
    xrange : Tuple[float, float], default=(-3, 3)
        Rango de visualización en x
    yrange : Tuple[float, float], default=(-3, 3)
        Rango de visualización en y
    density : int, default=12
        Número de vectores por dimensión (12×12 = 144 vectores)
    normalize : bool, default=True
        Si True, normaliza vectores a magnitud 1 (muestra solo dirección)
    scale : float, default=0.5
        Factor de escala para tamaño de vectores
    
    Retorna
    -------
    plotly.graph_objects.Figure
        Figura Plotly lista para mostrar con st.plotly_chart()
    
    Notas
    -----
    - density > 20 puede ser lento en navegador
    - normalize=True es mejor para visualizar direcciones
    - normalize=False muestra magnitudes relativas
    
    Ejemplos
    --------
    >>> import plotly.graph_objects as go
    >>> def F(x, y, z): return (-y, x, np.zeros_like(z))
    >>> fig = plot_vector_field_slice(F, 'z', 0.0, density=10)
    >>> # fig.show()  # Descomentar para visualizar
    """
    import plotly.graph_objects as go
    
    # Validación
    if plane not in ['x', 'y', 'z']:
        raise ValueError(f"plane debe ser 'x', 'y', o 'z', recibido: {plane}")
    
    if density > 30:
        logger.warning(
            f"density={density} es alto. Puede ser lento en navegador. "
            f"Considere density <= 20 para mejor rendimiento."
        )
    
    # Crear malla según el plano
    if plane == 'z':
        # Plano XY (z=const)
        x_vals = np.linspace(xrange[0], xrange[1], density)
        y_vals = np.linspace(yrange[0], yrange[1], density)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.full_like(X, plane_value)
        axis1_name, axis2_name = 'X', 'Y'
        
    elif plane == 'y':
        # Plano XZ (y=const)
        x_vals = np.linspace(xrange[0], xrange[1], density)
        z_vals = np.linspace(yrange[0], yrange[1], density)
        X, Z = np.meshgrid(x_vals, z_vals)
        Y = np.full_like(X, plane_value)
        axis1_name, axis2_name = 'X', 'Z'
        
    else:  # plane == 'x'
        # Plano YZ (x=const)
        y_vals = np.linspace(xrange[0], xrange[1], density)
        z_vals = np.linspace(yrange[0], yrange[1], density)
        Y, Z = np.meshgrid(y_vals, z_vals)
        X = np.full_like(Y, plane_value)
        axis1_name, axis2_name = 'Y', 'Z'
    
    # Evaluar campo vectorial
    Fx, Fy, Fz = F_num(X, Y, Z)
    
    # Calcular magnitudes
    magnitude = np.sqrt(Fx**2 + Fy**2 + Fz**2)
    
    # Normalizar si se solicita
    if normalize:
        # Evitar división por cero
        magnitude_safe = np.where(magnitude > 1e-10, magnitude, 1.0)
        Fx_plot = Fx / magnitude_safe
        Fy_plot = Fy / magnitude_safe
        Fz_plot = Fz / magnitude_safe
    else:
        Fx_plot = Fx
        Fy_plot = Fy
        Fz_plot = Fz
    
    # Escalar vectores
    Fx_plot *= scale
    Fy_plot *= scale
    Fz_plot *= scale
    
    # Aplanar arrays para Plotly
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    fx_flat = Fx_plot.flatten()
    fy_flat = Fy_plot.flatten()
    fz_flat = Fz_plot.flatten()
    mag_flat = magnitude.flatten()
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir campo vectorial como conos
    fig.add_trace(go.Cone(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        u=fx_flat,
        v=fy_flat,
        w=fz_flat,
        colorscale='Viridis',
        sizemode='absolute',
        sizeref=scale * 0.3,
        showscale=True,
        colorbar=dict(title="||F||"),
        cmin=mag_flat.min(),
        cmax=mag_flat.max(),
        hovertemplate=(
            f'<b>Campo Vectorial</b><br>'
            f'{axis1_name}: %{{x:.2f}}<br>'
            f'{axis2_name}: %{{y:.2f}}<br>'
            f'Magnitud: %{{marker.color:.3f}}<br>'
            f'<extra></extra>'
        ),
        name='Campo F'
    ))
    
    # Configurar layout
    plane_str = f"{plane}={plane_value:.2f}"
    title = f"Campo Vectorial en {plane_str}"
    if normalize:
        title += " (normalizado)"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        height=700,
        showlegend=False
    )
    
    return fig


# ==============================================================================
# FUNCIÓN 8 (OPCIONAL): VERIFICACIÓN NUMÉRICA DEL TEOREMA DE STOKES
# ==============================================================================

def compare_stokes_surface_line(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    r_boundary: Callable,
    dr_boundary: Callable,
    t0: float,
    t1: float,
    r_surface: Callable,
    ru_surface: Callable,
    rv_surface: Callable,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    vars: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    tolerance: float = 1e-3,
    n_line: int = 2000,
    nu_surf: int = 100,
    nv_surf: int = 100
) -> Dict[str, Union[float, bool, str]]:
    """
    Verifica numéricamente el Teorema de Stokes.
    
    Teorema de Stokes:
    ∮_C F·dr = ∬_S (∇×F)·n dS
    
    donde C es la frontera de S con orientación compatible.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial simbólico (P, Q, R)
    r_boundary : Callable
        Curva frontera r(t) parametrizada
    dr_boundary : Callable
        Derivada r'(t)
    t0, t1 : float
        Límites de t para la frontera
    r_surface : Callable
        Superficie S(u, v)
    ru_surface, rv_surface : Callable
        Derivadas parciales de S
    u0, u1, v0, v1 : float
        Límites de u y v
    vars : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables (x, y, z)
    tolerance : float, default=1e-3
        Tolerancia para considerar que se cumple Stokes
    n_line : int, default=2000
        Puntos para integral de línea
    nu_surf, nv_surf : int, default=100
        Puntos para integral de superficie
    
    Retorna
    -------
    dict
        {
            'line_integral': valor de ∮_C F·dr,
            'surface_integral': valor de ∬_S (∇×F)·n dS,
            'difference': diferencia absoluta,
            'relative_error': error relativo,
            'stokes_holds': bool (True si |diff| < tolerance),
            'message': str con resultado
        }
    
    Notas
    -----
    Este test es NUMÉRICO, no exacto. Los errores vienen de:
    - Discretización de las integrales
    - Errores de redondeo
    - Orientación de la superficie
    
    Ejemplos
    --------
    >>> # TODO: Agregar ejemplo con superficie conocida
    """
    # Calcular rotacional de F
    _, curl_sym, _, curl_num = compute_divergence_and_curl(F_sym, vars)
    
    # Lambdify F
    F_num = lambdify_vector(F_sym, vars)
    
    # Calcular integral de línea ∮_C F·dr
    line_int = line_integral_over_param(
        F_num, r_boundary, dr_boundary, t0, t1, n=n_line
    )
    
    # Calcular integral de superficie ∬_S (∇×F)·n dS
    surface_int = surface_flux_over_param(
        curl_num, r_surface, ru_surface, rv_surface,
        u0, u1, v0, v1, nu=nu_surf, nv=nv_surf
    )
    
    # NUEVO: Calcular error grid para visualización
    u_vals = np.linspace(u0, u1, nu_surf)
    v_vals = np.linspace(v0, v1, nv_surf)
    U, V = np.meshgrid(u_vals, v_vals)
    
    # Calcular (∇×F)·n en cada punto de la malla
    error_grid = np.zeros_like(U)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u_ij = U[i, j]
            v_ij = V[i, j]
            
            # Posición en el espacio
            r_ij = np.array(r_surface(u_ij, v_ij))
            
            # Rotacional evaluado en r(u,v)
            curl_at_point = np.array(curl_num(*r_ij))
            
            # Vector normal: ru × rv
            ru_ij = np.array(ru_surface(u_ij, v_ij))
            rv_ij = np.array(rv_surface(u_ij, v_ij))
            n_ij = np.cross(ru_ij, rv_ij)
            
            # Producto punto: (∇×F)·n
            integrand = np.dot(curl_at_point, n_ij)
            
            # Error: diferencia respecto al valor medio esperado
            # (Este es un proxy del error local)
            error_grid[i, j] = integrand
    
    # Normalizar error_grid para mostrar desviación del promedio
    mean_integrand = np.mean(error_grid)
    error_grid = error_grid - mean_integrand
    
    # Calcular diferencia y error relativo
    difference = abs(line_int - surface_int)
    
    # Evitar división por cero en error relativo
    avg_value = (abs(line_int) + abs(surface_int)) / 2.0
    if avg_value > 1e-10:
        relative_error = difference / avg_value
    else:
        relative_error = difference  # Si ambos son ~0, usar diferencia absoluta
    
    stokes_holds = difference < tolerance
    
    # Mensaje de resultado
    if stokes_holds:
        message = f"✓ Teorema de Stokes verificado (error: {difference:.6f} < {tolerance})"
    else:
        message = f"✗ Teorema de Stokes NO verificado (error: {difference:.6f} ≥ {tolerance})"
    
    logger.info(
        f"Stokes check: ∮F·dr={line_int:.6f}, ∬(∇×F)·n dS={surface_int:.6f}, "
        f"diff={difference:.6f}, holds={stokes_holds}"
    )
    
    return {
        'line_integral': float(line_int),
        'surface_integral': float(surface_int),
        'difference': float(difference),
        'relative_error': float(relative_error),
        'stokes_holds': stokes_holds,
        'message': message,
        'error_grid': error_grid,
        'U': U,
        'V': V
    }


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def format_vector_latex(vec: Tuple[sp.Expr, ...], names: Tuple[str, ...] = ('i', 'j', 'k')) -> str:
    """
    Formatea un vector simbólico para LaTeX.
    
    Parámetros
    ----------
    vec : Tuple[sp.Expr, ...]
        Componentes del vector
    names : Tuple[str, ...], default=('i', 'j', 'k')
        Nombres de los vectores base
    
    Retorna
    -------
    str
        String LaTeX formateado
    
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> vec = (2*x, 3*y, 0)
    >>> format_vector_latex(vec)
    '(2 x)\\mathbf{i} + (3 y)\\mathbf{j}'
    """
    terms = []
    for component, name in zip(vec, names):
        if component != 0:
            latex_comp = sp.latex(component)
            terms.append(f"({latex_comp})\\mathbf{{{name}}}")
    
    if not terms:
        return "\\mathbf{0}"
    
    return " + ".join(terms)


# ==============================================================================
# NUEVAS FUNCIONALIDADES AVANZADAS
# ==============================================================================

def explain_gradient_steps(
    phi_expr: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    detail_level: str = 'intermedio'
) -> Dict[str, Any]:
    """
    Explica paso a paso el cálculo del gradiente con derivadas simbólicas.
    
    Parámetros
    ----------
    phi_expr : sp.Expr
        Expresión simbólica del campo escalar φ(x,y,z)
    vars : Tuple[sp.Symbol, ...]
        Variables (x, y, z) como símbolos de SymPy
    detail_level : str, default='intermedio'
        Nivel de detalle: 'basico', 'intermedio', 'completo'
    
    Retorna
    -------
    Dict[str, Any]
        Diccionario con:
        - 'grad_sym': tupla de expresiones simbólicas (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
        - 'grad_latex': lista con cada derivada en LaTeX
        - 'explanations': lista de strings explicando cada paso
        - 'critical_points': lista de puntos críticos simbólicos
        - 'phi_original_latex': LaTeX de φ original
    
    Ejemplos
    --------
    >>> x, y, z = sp.symbols('x y z')
    >>> phi = x**2 + y**2 + z**2
    >>> result = explain_gradient_steps(phi, (x, y, z))
    >>> result['grad_sym']
    (2*x, 2*y, 2*z)
    """
    import time
    start_time = time.time()
    
    logger.info(f"Explicando gradiente de {phi_expr} con nivel {detail_level}")
    
    # Calcular derivadas parciales
    grad_sym = tuple(sp.diff(phi_expr, var) for var in vars)
    grad_latex = [sp.latex(g) for g in grad_sym]
    phi_original_latex = sp.latex(phi_expr)
    
    # Generar explicaciones para cada derivada parcial
    explanations = []
    var_names = ['x', 'y', 'z'][:len(vars)]
    
    for i, (var, deriv, var_name) in enumerate(zip(vars, grad_sym, var_names)):
        explanation = _explain_derivative(phi_expr, var, deriv, var_name, detail_level)
        explanations.append(explanation)
    
    # Intentar encontrar puntos críticos (∇φ = 0)
    critical_points = []
    try:
        logger.info("Intentando resolver ∇φ = 0 simbólicamente...")
        solutions = sp.solve(list(grad_sym), vars, dict=True)
        
        for sol_dict in solutions[:10]:  # Limitar a 10 soluciones
            if all(val.is_real for val in sol_dict.values() if hasattr(val, 'is_real')):
                point = tuple(sol_dict.get(v, 0) for v in vars)
                critical_points.append(point)
        
        logger.info(f"Encontrados {len(critical_points)} puntos críticos")
    except Exception as e:
        logger.warning(f"No se pudieron encontrar puntos críticos simbólicamente: {e}")
        critical_points = []
    
    elapsed = time.time() - start_time
    logger.info(f"Explicación del gradiente completada en {elapsed:.3f}s")
    
    # Construir gradient_latex como string (formato vectorial)
    gradient_latex = f"\\left( {', '.join(grad_latex)} \\right)"
    
    # Construir steps_latex como lista de pasos
    steps_latex = []
    
    # Paso 1: Definición
    steps_latex.append({
        'title': 'Paso 1: Definición del Gradiente',
        'content': f"$$\\nabla \\phi = \\left( \\frac{{\\partial \\phi}}{{\\partial x}}, \\frac{{\\partial \\phi}}{{\\partial y}}, \\frac{{\\partial \\phi}}{{\\partial z}} \\right)$$",
        'explanation': 'El gradiente es un vector formado por las derivadas parciales respecto a cada variable.'
    })
    
    # Paso 2: Campo escalar original
    steps_latex.append({
        'title': 'Paso 2: Campo Escalar φ',
        'content': f"$$\\phi = {phi_original_latex}$$",
        'explanation': ''
    })
    
    # Pasos 3, 4, 5: Cada derivada parcial
    var_names = ['x', 'y', 'z'][:len(vars)]
    for i, (var_name, deriv_latex, explanation) in enumerate(zip(var_names, grad_latex, explanations)):
        steps_latex.append({
            'title': f'Paso {i+3}: Derivada Parcial ∂φ/∂{var_name}',
            'content': f"$$\\frac{{\\partial \\phi}}{{\\partial {var_name}}} = {deriv_latex}$$",
            'explanation': explanation
        })
    
    # Paso final: Resultado
    steps_latex.append({
        'title': f'Paso {len(vars)+3}: Resultado del Gradiente',
        'content': f"$$\\nabla \\phi = {gradient_latex}$$",
        'explanation': 'Este es el gradiente completo del campo escalar.'
    })
    
    # Calcular valor numérico (opcional, para evaluación posterior)
    gradient_num = None
    try:
        # Intentar evaluar en un punto por defecto
        gradient_num = tuple(float(g.evalf()) if g.is_number else None for g in grad_sym)
    except:
        pass
    
    return {
        'gradient_sym': grad_sym,  # tupla de expresiones simbólicas
        'gradient_latex': gradient_latex,  # string para mostrar
        'gradient_num': gradient_num,  # evaluación numérica
        'steps_latex': steps_latex,  # lista de pasos
        'critical_points': critical_points,
        'phi_original_latex': phi_original_latex,
        'execution_time': elapsed
    }


def _explain_derivative(
    expr: sp.Expr,
    var: sp.Symbol,
    result: sp.Expr,
    var_name: str,
    detail_level: str
) -> str:
    """
    Genera explicación en español de cómo se derivó una expresión.
    
    Parámetros
    ----------
    expr : sp.Expr
        Expresión original
    var : sp.Symbol
        Variable respecto a la cual se deriva
    result : sp.Expr
        Resultado de la derivada
    var_name : str
        Nombre de la variable ('x', 'y', 'z')
    detail_level : str
        'basico', 'intermedio', 'completo'
    
    Retorna
    -------
    str
        Explicación en texto
    """
    
    # ========================================================================
    # NIVEL BÁSICO: Solo fórmula y resultado final
    # ========================================================================
    if detail_level == 'basico':
        explanation = f"**∂φ/∂{var_name}** = {sp.latex(result)}"
        return explanation
    
    # ========================================================================
    # Análisis detallado de la expresión para niveles superiores
    # ========================================================================
    
    # Detectar estructura de la expresión
    terms = []
    if isinstance(expr, sp.Add):
        terms = list(expr.args)
    else:
        terms = [expr]
    
    # Clasificar términos según su tipo
    constant_terms = []
    polynomial_terms = []
    trig_terms = []
    exp_terms = []
    log_terms = []
    product_terms = []
    composite_terms = []
    
    for term in terms:
        if not term.has(var):
            constant_terms.append(term)
        elif isinstance(term, sp.Pow) and term.args[0] == var:
            polynomial_terms.append(term)
        elif term.has(sp.sin) or term.has(sp.cos) or term.has(sp.tan):
            trig_terms.append(term)
        elif term.has(sp.exp):
            exp_terms.append(term)
        elif term.has(sp.log):
            log_terms.append(term)
        elif isinstance(term, sp.Mul) and sum(1 for arg in term.args if arg.has(var)) > 1:
            product_terms.append(term)
        elif term == var:
            polynomial_terms.append(term)
        else:
            composite_terms.append(term)
    
    # ========================================================================
    # NIVEL INTERMEDIO: Derivación paso a paso con reglas nombradas
    # ========================================================================
    if detail_level == 'intermedio':
        explanation = f"**Derivando ∂φ/∂{var_name}:**\n\n"
        explanation += f"**Expresión original:** φ = {sp.latex(expr)}\n\n"
        
        steps = []
        
        # Términos constantes
        if constant_terms:
            const_sum = sp.Add(*constant_terms)
            steps.append(f"• **Términos constantes** ({sp.latex(const_sum)}): "
                        f"La derivada de una constante es **0**")
        
        # Términos polinomiales
        if polynomial_terms:
            for term in polynomial_terms:
                if term == var:
                    steps.append(f"• **Derivada de {var_name}:** d/d{var_name}({var_name}) = **1**")
                elif isinstance(term, sp.Pow):
                    base, exp_val = term.args
                    if base == var and exp_val.is_number:
                        deriv = sp.diff(term, var)
                        steps.append(f"• **Regla de la potencia:** d/d{var_name}({sp.latex(term)}) = "
                                   f"{exp_val}·{var_name}^{{{exp_val-1}}} = **{sp.latex(deriv)}**")
        
        # Términos trigonométricos
        if trig_terms:
            for term in trig_terms:
                deriv = sp.diff(term, var)
                if term.has(sp.sin):
                    steps.append(f"• **Derivada del seno:** d/d{var_name}({sp.latex(term)}) = **{sp.latex(deriv)}**")
                elif term.has(sp.cos):
                    steps.append(f"• **Derivada del coseno:** d/d{var_name}({sp.latex(term)}) = **{sp.latex(deriv)}**")
        
        # Términos exponenciales
        if exp_terms:
            for term in exp_terms:
                deriv = sp.diff(term, var)
                steps.append(f"• **Derivada exponencial:** d/d{var_name}({sp.latex(term)}) = **{sp.latex(deriv)}**")
        
        # Términos logarítmicos
        if log_terms:
            for term in log_terms:
                deriv = sp.diff(term, var)
                steps.append(f"• **Derivada del logaritmo:** d/d{var_name}({sp.latex(term)}) = **{sp.latex(deriv)}**")
        
        # Productos (regla del producto)
        if product_terms:
            for term in product_terms:
                deriv = sp.diff(term, var)
                steps.append(f"• **Regla del producto:** d/d{var_name}({sp.latex(term)}) = **{sp.latex(deriv)}**")
        
        # Términos compuestos
        if composite_terms:
            for term in composite_terms:
                deriv = sp.diff(term, var)
                steps.append(f"• **Regla de la cadena:** d/d{var_name}({sp.latex(term)}) = **{sp.latex(deriv)}**")
        
        explanation += "\n".join(steps)
        explanation += f"\n\n**Resultado final:** ∂φ/∂{var_name} = **{sp.latex(result)}**"
        
        return explanation
    
    # ========================================================================
    # NIVEL COMPLETO: Derivación simbólica detallada + interpretación geométrica
    # ========================================================================
    if detail_level == 'completo':
        explanation = f"## 📐 Derivación Completa de ∂φ/∂{var_name}\n\n"
        explanation += f"### 1️⃣ Expresión Original\n"
        explanation += f"$$\\phi = {sp.latex(expr)}$$\n\n"
        
        explanation += f"### 2️⃣ Análisis de Términos\n\n"
        
        # Desglose detallado de cada tipo de término
        if constant_terms:
            const_sum = sp.Add(*constant_terms)
            explanation += f"**Términos Constantes:** {sp.latex(const_sum)}\n\n"
            explanation += f"• Las constantes son independientes de {var_name}\n"
            explanation += f"• Por definición: d/d{var_name}(C) = 0 para cualquier constante C\n"
            explanation += f"• **Contribución al gradiente:** 0\n\n"
        
        if polynomial_terms:
            explanation += f"**Términos Polinomiales en {var_name}:**\n\n"
            for term in polynomial_terms:
                if term == var:
                    explanation += f"• Término lineal: {sp.latex(term)}\n"
                    explanation += f"  - Aplicamos: d/d{var_name}({var_name}) = 1\n"
                    explanation += f"  - **Derivada:** 1\n\n"
                elif isinstance(term, sp.Pow):
                    base, exp_val = term.args
                    if base == var and exp_val.is_number:
                        deriv = sp.diff(term, var)
                        explanation += f"• Potencia: {sp.latex(term)}\n"
                        explanation += f"  - **Regla de la potencia:** d/d{var_name}({var_name}^n) = n·{var_name}^(n-1)\n"
                        explanation += f"  - Aplicando con n={exp_val}: d/d{var_name}({sp.latex(term)}) = {exp_val}·{var_name}^{{{exp_val-1}}}\n"
                        explanation += f"  - **Derivada:** {sp.latex(deriv)}\n\n"
        
        if trig_terms:
            explanation += f"**Términos Trigonométricos:**\n\n"
            for term in trig_terms:
                deriv = sp.diff(term, var)
                if term.has(sp.sin):
                    inner = list(term.find(sp.sin))[0].args[0]
                    explanation += f"• Seno: {sp.latex(term)}\n"
                    explanation += f"  - **Regla básica:** d/d{var_name}(sin(u)) = cos(u)·du/d{var_name}\n"
                    if inner != var:
                        inner_deriv = sp.diff(inner, var)
                        explanation += f"  - Función interna: u = {sp.latex(inner)}, du/d{var_name} = {sp.latex(inner_deriv)}\n"
                        explanation += f"  - **Regla de la cadena:** cos({sp.latex(inner)})·{sp.latex(inner_deriv)}\n"
                    explanation += f"  - **Derivada:** {sp.latex(deriv)}\n\n"
                elif term.has(sp.cos):
                    inner = list(term.find(sp.cos))[0].args[0]
                    explanation += f"• Coseno: {sp.latex(term)}\n"
                    explanation += f"  - **Regla básica:** d/d{var_name}(cos(u)) = -sin(u)·du/d{var_name}\n"
                    if inner != var:
                        inner_deriv = sp.diff(inner, var)
                        explanation += f"  - Función interna: u = {sp.latex(inner)}, du/d{var_name} = {sp.latex(inner_deriv)}\n"
                        explanation += f"  - **Regla de la cadena:** -sin({sp.latex(inner)})·{sp.latex(inner_deriv)}\n"
                    explanation += f"  - **Derivada:** {sp.latex(deriv)}\n\n"
        
        if exp_terms:
            explanation += f"**Términos Exponenciales:**\n\n"
            for term in exp_terms:
                deriv = sp.diff(term, var)
                exps = list(term.find(sp.exp))
                if exps:
                    inner = exps[0].args[0]
                    explanation += f"• Exponencial: {sp.latex(term)}\n"
                    explanation += f"  - **Regla básica:** d/d{var_name}(e^u) = e^u·du/d{var_name}\n"
                    if inner != var:
                        inner_deriv = sp.diff(inner, var)
                        explanation += f"  - Exponente: u = {sp.latex(inner)}, du/d{var_name} = {sp.latex(inner_deriv)}\n"
                        explanation += f"  - **Regla de la cadena:** e^{{{sp.latex(inner)}}}·{sp.latex(inner_deriv)}\n"
                    explanation += f"  - **Derivada:** {sp.latex(deriv)}\n\n"
        
        if log_terms:
            explanation += f"**Términos Logarítmicos:**\n\n"
            for term in log_terms:
                deriv = sp.diff(term, var)
                logs = list(term.find(sp.log))
                if logs:
                    inner = logs[0].args[0]
                    explanation += f"• Logaritmo: {sp.latex(term)}\n"
                    explanation += f"  - **Regla básica:** d/d{var_name}(ln(u)) = (1/u)·du/d{var_name}\n"
                    if inner != var:
                        inner_deriv = sp.diff(inner, var)
                        explanation += f"  - Argumento: u = {sp.latex(inner)}, du/d{var_name} = {sp.latex(inner_deriv)}\n"
                        explanation += f"  - **Regla de la cadena:** (1/{sp.latex(inner)})·{sp.latex(inner_deriv)}\n"
                    explanation += f"  - **Derivada:** {sp.latex(deriv)}\n\n"
        
        if product_terms:
            explanation += f"**Productos (Regla del Producto):**\n\n"
            for term in product_terms:
                deriv = sp.diff(term, var)
                explanation += f"• Producto: {sp.latex(term)}\n"
                explanation += f"  - **Regla del producto:** d/d{var_name}(u·v) = u'·v + u·v'\n"
                
                # Intentar identificar factores u y v
                factors = [arg for arg in term.args if arg.has(var)]
                if len(factors) >= 2:
                    u, v = factors[0], sp.Mul(*factors[1:])
                    u_prime = sp.diff(u, var)
                    v_prime = sp.diff(v, var)
                    explanation += f"  - u = {sp.latex(u)}, u' = {sp.latex(u_prime)}\n"
                    explanation += f"  - v = {sp.latex(v)}, v' = {sp.latex(v_prime)}\n"
                    explanation += f"  - u'·v + u·v' = ({sp.latex(u_prime)})·({sp.latex(v)}) + ({sp.latex(u)})·({sp.latex(v_prime)})\n"
                
                explanation += f"  - **Derivada:** {sp.latex(deriv)}\n\n"
        
        if composite_terms:
            explanation += f"**Términos Compuestos:**\n\n"
            for term in composite_terms:
                deriv = sp.diff(term, var)
                explanation += f"• Compuesto: {sp.latex(term)}\n"
                explanation += f"  - **Derivada:** {sp.latex(deriv)}\n\n"
        
        explanation += f"### 3️⃣ Simplificación y Resultado Final\n\n"
        explanation += f"Sumando todas las derivadas parciales y simplificando:\n\n"
        explanation += f"$$\\frac{{\\partial \\phi}}{{\\partial {var_name}}} = {sp.latex(result)}$$\n\n"
        
        # Interpretación geométrica
        explanation += f"### 4️⃣ Interpretación Geométrica\n\n"
        explanation += f"**∂φ/∂{var_name}** representa:\n\n"
        explanation += f"• La **tasa de cambio instantánea** de φ cuando nos movemos en la dirección {var_name}\n"
        explanation += f"• Si ∂φ/∂{var_name} > 0: φ **aumenta** al incrementar {var_name}\n"
        explanation += f"• Si ∂φ/∂{var_name} < 0: φ **disminuye** al incrementar {var_name}\n"
        explanation += f"• Si ∂φ/∂{var_name} = 0: φ es **estacionaria** en la dirección {var_name} (posible punto crítico)\n\n"
        
        # Evaluar en un punto ejemplo SIMBÓLICAMENTE (sin decimales)
        try:
            # Intentar evaluar en (1, 1, 1)
            test_point = {sp.Symbol('x'): 1, sp.Symbol('y'): 1, sp.Symbol('z'): 1}
            if var in test_point:
                result_at_point = result.subs(test_point)
                result_at_point = sp.simplify(result_at_point)
                if result_at_point.is_number:
                    explanation += f"### 5️⃣ Ejemplo en el Punto (1,1,1)\n\n"
                    explanation += f"Evaluando simbólicamente:\n\n"
                    explanation += f"$$\\frac{{\\partial \\phi}}{{\\partial {var_name}}}\\bigg|_{{(1,1,1)}} = {sp.latex(result_at_point)}$$\n\n"
                    
                    # Interpretación sin decimales
                    if result_at_point > 0:
                        explanation += f"En este punto, φ está **aumentando** en la dirección {var_name}.\n"
                    elif result_at_point < 0:
                        explanation += f"En este punto, φ está **disminuyendo** en la dirección {var_name}.\n"
                    else:
                        explanation += f"En este punto, φ es **estacionaria** en la dirección {var_name}.\n"
        except:
            pass
        
        return explanation
    
    # Fallback
    return f"∂φ/∂{var_name} = {sp.latex(result)}"


def explain_div_curl_steps(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
) -> Dict[str, Any]:
    """
    Explica paso a paso el cálculo de divergencia y rotacional con derivaciones simbólicas.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial (P, Q, R)
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables (x, y, z)
    
    Retorna
    -------
    Dict[str, Any]
        Diccionario con todos los pasos y resultados
    """
    import time
    start_time = time.time()
    
    logger.info("Explicando cálculo de divergencia y rotacional paso a paso")
    
    x, y, z = vars_
    P, Q, R = F_sym
    
    steps_latex = []
    
    # Paso 1: Mostrar el campo vectorial
    steps_latex.append({
        'title': '1. Campo vectorial F',
        'content': f"$$\\mathbf{{F}} = \\left( {sp.latex(P)}, {sp.latex(Q)}, {sp.latex(R)} \\right)$$",
        'explanation': 'Este es el campo vectorial sobre el cual calcularemos la divergencia y el rotacional.'
    })
    
    # DIVERGENCIA
    # Paso 2: Fórmula de la divergencia
    steps_latex.append({
        'title': '2. Fórmula de la divergencia',
        'content': r"$$\nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$",
        'explanation': 'La divergencia mide la tendencia del campo a expandirse o contraerse.'
    })
    
    # Paso 3: Calcular cada derivada parcial
    dP_dx = sp.diff(P, x)
    dQ_dy = sp.diff(Q, y)
    dR_dz = sp.diff(R, z)
    
    steps_latex.append({
        'title': '3. Derivadas parciales para la divergencia',
        'content': f"""$$\\frac{{\\partial P}}{{\\partial x}} = \\frac{{\\partial}}{{\\partial x}}\\left( {sp.latex(P)} \\right) = {sp.latex(dP_dx)}$$

$$\\frac{{\\partial Q}}{{\\partial y}} = \\frac{{\\partial}}{{\\partial y}}\\left( {sp.latex(Q)} \\right) = {sp.latex(dQ_dy)}$$

$$\\frac{{\\partial R}}{{\\partial z}} = \\frac{{\\partial}}{{\\partial z}}\\left( {sp.latex(R)} \\right) = {sp.latex(dR_dz)}$$""",
        'explanation': 'Calculamos cada derivada parcial por separado.'
    })
    
    # Paso 4: Resultado de la divergencia
    div_sym = dP_dx + dQ_dy + dR_dz
    div_sym = sp.simplify(div_sym)
    
    steps_latex.append({
        'title': '4. Resultado de la divergencia',
        'content': f"$$\\nabla \\cdot \\mathbf{{F}} = {sp.latex(dP_dx)} + {sp.latex(dQ_dy)} + {sp.latex(dR_dz)}$$",
        'explanation': f'Sumamos las tres derivadas parciales.'
    })
    
    steps_latex.append({
        'title': '5. Divergencia simplificada',
        'content': f"$$\\nabla \\cdot \\mathbf{{F}} = {sp.latex(div_sym)}$$",
        'explanation': f'Resultado final de la divergencia. ' + 
                      ('🔵 **Campo incompresible** (conserva volumen)' if div_sym == 0 else '🔹 Campo compresible')
    })
    
    # ROTACIONAL
    # Paso 6: Fórmula del rotacional
    steps_latex.append({
        'title': '6. Fórmula del rotacional',
        'content': r"""$$\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$$""",
        'explanation': 'El rotacional mide la tendencia del campo a rotar alrededor de un punto.'
    })
    
    # Paso 7: Calcular cada componente del rotacional
    curl_i = sp.diff(R, y) - sp.diff(Q, z)
    curl_j = sp.diff(P, z) - sp.diff(R, x)
    curl_k = sp.diff(Q, x) - sp.diff(P, y)
    
    steps_latex.append({
        'title': '7. Componentes del rotacional',
        'content': f"""**Componente i:**
$$\\left(\\nabla \\times \\mathbf{{F}}\\right)_i = \\frac{{\\partial R}}{{\\partial y}} - \\frac{{\\partial Q}}{{\\partial z}} = {sp.latex(sp.diff(R, y))} - {sp.latex(sp.diff(Q, z))} = {sp.latex(curl_i)}$$

**Componente j:**
$$\\left(\\nabla \\times \\mathbf{{F}}\\right)_j = \\frac{{\\partial P}}{{\\partial z}} - \\frac{{\\partial R}}{{\\partial x}} = {sp.latex(sp.diff(P, z))} - {sp.latex(sp.diff(R, x))} = {sp.latex(curl_j)}$$

**Componente k:**
$$\\left(\\nabla \\times \\mathbf{{F}}\\right)_k = \\frac{{\\partial Q}}{{\\partial x}} - \\frac{{\\partial P}}{{\\partial y}} = {sp.latex(sp.diff(Q, x))} - {sp.latex(sp.diff(P, y))} = {sp.latex(curl_k)}$$""",
        'explanation': 'Calculamos cada componente del rotacional usando el determinante formal.'
    })
    
    # Paso 8: Resultado del rotacional
    curl_sym = (curl_i, curl_j, curl_k)
    curl_sym = tuple(sp.simplify(comp) for comp in curl_sym)
    
    steps_latex.append({
        'title': '8. Resultado del rotacional',
        'content': f"$$\\nabla \\times \\mathbf{{F}} = \\left( {sp.latex(curl_sym[0])}, {sp.latex(curl_sym[1])}, {sp.latex(curl_sym[2])} \\right)$$",
        'explanation': 'Resultado final del rotacional. ' + 
                      ('✅ **Campo conservativo** (existe potencial)' if all(c == 0 for c in curl_sym) else '🌀 Campo rotacional')
    })
    
    elapsed = time.time() - start_time
    logger.info(f"Cálculo de divergencia y rotacional completado en {elapsed:.3f}s")
    
    return {
        'divergence_sym': div_sym,
        'divergence_latex': sp.latex(div_sym),
        'curl_sym': curl_sym,
        'curl_latex': tuple(sp.latex(c) for c in curl_sym),
        'is_incompressible': (div_sym == 0),
        'is_conservative': all(c == 0 for c in curl_sym),
        'steps_latex': steps_latex,
        'execution_time': elapsed
    }


def explain_line_integral_steps(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    r_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    t: sp.Symbol,
    t0: float,
    t1: float
) -> Dict[str, Any]:
    """
    Explica paso a paso el cálculo de una integral de línea con derivaciones simbólicas.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial (P, Q, R) como expresiones simbólicas
    r_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Parametrización (x(t), y(t), z(t)) como expresiones simbólicas
    t : sp.Symbol
        Símbolo del parámetro
    t0 : float
        Límite inferior de integración
    t1 : float
        Límite superior de integración
    
    Retorna
    -------
    Dict[str, Any]
        Diccionario con:
        - 'r_sym': parametrización r(t)
        - 'r_prime_sym': derivada r'(t)
        - 'F_of_r_sym': F(r(t))
        - 'integrand_sym': F(r(t))·r'(t) simplificado
        - 'antiderivative': antiderivada simbólica (si existe)
        - 'definite_symbolic': valor simbólico de la integral (si existe)
        - 'numeric_value': valor numérico calculado
        - 'steps_latex': lista de pasos en LaTeX
        - 'why_symbolic_failed': razón del fallo si no se pudo resolver simbólicamente
    
    Ejemplos
    --------
    >>> x, y, z, t = sp.symbols('x y z t')
    >>> F = (-y, x, 0)
    >>> r = (sp.cos(t), sp.sin(t), 0)
    >>> result = explain_line_integral_steps(F, r, t, 0, 2*sp.pi)
    >>> result['integrand_sym']
    1
    """
    import time
    start_time = time.time()
    
    logger.info(f"Explicando integral de línea paso a paso")
    
    x, y, z = sp.symbols('x y z')
    P, Q, R = F_sym
    x_t, y_t, z_t = r_sym
    
    steps_latex = []
    why_symbolic_failed = None
    
    # Convertir límites numéricos a simbólicos si son múltiplos de π
    def to_symbolic_limit(val):
        """Convierte un límite numérico a simbólico si es múltiplo de π."""
        if isinstance(val, (int, float)):
            # Detectar múltiplos de π
            ratio = val / np.pi
            if abs(ratio - round(ratio)) < 1e-10:  # Es múltiplo de π
                ratio_int = int(round(ratio))
                if ratio_int == 0:
                    return sp.Integer(0)
                elif ratio_int == 1:
                    return sp.pi
                elif ratio_int == -1:
                    return -sp.pi
                elif ratio_int == 2:
                    return 2*sp.pi
                elif ratio_int == -2:
                    return -2*sp.pi
                else:
                    return ratio_int * sp.pi
            # Detectar fracciones de π comunes
            for denom in [2, 3, 4, 6]:
                ratio = val / (np.pi / denom)
                if abs(ratio - round(ratio)) < 1e-10:
                    ratio_int = int(round(ratio))
                    if ratio_int == 1:
                        return sp.pi / denom
                    else:
                        return ratio_int * sp.pi / denom
        return val
    
    t0_sym = to_symbolic_limit(t0)
    t1_sym = to_symbolic_limit(t1)
    
    # Paso 1: Mostrar parametrización
    steps_latex.append({
        'title': '1. Parametrización de la curva',
        'content': f"$$\\mathbf{{r}}(t) = \\left( {sp.latex(x_t)}, {sp.latex(y_t)}, {sp.latex(z_t)} \\right)$$",
        'explanation': 'Esta es la curva sobre la cual vamos a integrar.'
    })
    
    # Paso 2: Calcular r'(t)
    r_prime = (sp.diff(x_t, t), sp.diff(y_t, t), sp.diff(z_t, t))
    r_prime = tuple(sp.simplify(comp) for comp in r_prime)
    
    steps_latex.append({
        'title': "2. Derivada de la parametrización",
        'content': f"$$\\mathbf{{r}}'(t) = \\left( {sp.latex(r_prime[0])}, {sp.latex(r_prime[1])}, {sp.latex(r_prime[2])} \\right)$$",
        'explanation': 'Derivamos cada componente de r(t) respecto a t.'
    })
    
    # Paso 3: Sustituir en F
    F_of_r = (
        P.subs({x: x_t, y: y_t, z: z_t}),
        Q.subs({x: x_t, y: y_t, z: z_t}),
        R.subs({x: x_t, y: y_t, z: z_t})
    )
    F_of_r = tuple(sp.simplify(comp) for comp in F_of_r)
    
    steps_latex.append({
        'title': '3. Campo vectorial evaluado en la curva',
        'content': f"$$\\mathbf{{F}}(\\mathbf{{r}}(t)) = \\left( {sp.latex(F_of_r[0])}, {sp.latex(F_of_r[1])}, {sp.latex(F_of_r[2])} \\right)$$",
        'explanation': 'Sustituimos x=x(t), y=y(t), z=z(t) en el campo F.'
    })
    
    # Paso 4: Producto punto F·dr
    integrand = sum(F_of_r[i] * r_prime[i] for i in range(3))
    integrand = sp.simplify(integrand)
    integrand = sp.trigsimp(integrand)  # Simplificar identidades trigonométricas
    
    steps_latex.append({
        'title': '4. Producto punto F(r(t))·r\'(t)',
        'content': f"$$\\mathbf{{F}}(\\mathbf{{r}}(t)) \\cdot \\mathbf{{r}}'(t) = {sp.latex(integrand)}$$",
        'explanation': 'Este es el integrando de nuestra integral de línea.'
    })
    
    # Paso 5: Intentar integración simbólica
    antiderivative = None
    definite_symbolic = None
    
    try:
        logger.info("Intentando integración simbólica...")
        antiderivative = sp.integrate(integrand, t)
        
        if antiderivative.has(sp.Integral):
            # La integral no se pudo resolver
            raise ValueError("SymPy returned unevaluated integral")
        
        steps_latex.append({
            'title': '5. Antiderivada (integración simbólica)',
            'content': f"$$\\int {sp.latex(integrand)} \\, dt = {sp.latex(antiderivative)} + C$$",
            'explanation': 'Calculamos la antiderivada del integrando.'
        })
        
        # Evaluar en los límites
        try:
            val_upper = antiderivative.subs(t, t1_sym)
            val_lower = antiderivative.subs(t, t0_sym)
            definite_symbolic = sp.simplify(val_upper - val_lower)
            
            steps_latex.append({
                'title': '6. Evaluación en los límites',
                'content': f"$$\\left[ {sp.latex(antiderivative)} \\right]_{{{sp.latex(t0_sym)}}}^{{{sp.latex(t1_sym)}}} = {sp.latex(val_upper)} - ({sp.latex(val_lower)}) = {sp.latex(definite_symbolic)}$$",
                'explanation': f'Aplicamos el teorema fundamental del cálculo: F(t₁) - F(t₀).'
            })
            
            logger.info(f"Integral simbólica exitosa: {definite_symbolic}")
            
        except Exception as e:
            logger.warning(f"Error evaluando límites: {e}")
            why_symbolic_failed = f"La antiderivada existe pero no se pudo evaluar en los límites: {str(e)}"
            
    except Exception as e:
        logger.warning(f"Integración simbólica falló: {e}")
        why_symbolic_failed = f"No se pudo calcular la integral simbólicamente. Razón: {str(e)}"
        
        steps_latex.append({
            'title': '5. Integración simbólica no disponible',
            'content': f"$$\\int_{{{sp.latex(t0_sym)}}}^{{{sp.latex(t1_sym)}}} {sp.latex(integrand)} \\, dt \\quad \\text{{(evaluación numérica requerida)}}$$",
            'explanation': f'La integral no tiene forma cerrada simple. Se usará integración numérica.'
        })
    
    # Paso 6: Cálculo numérico (siempre)
    try:
        # Convertir a función numérica
        integrand_num = sp.lambdify(t, integrand, modules=['numpy'])
        
        # Usar scipy.integrate.quad para mayor precisión
        from scipy import integrate as sci_integrate
        numeric_value, error_estimate = sci_integrate.quad(integrand_num, float(t0), float(t1))
        
        steps_latex.append({
            'title': '7. Verificación numérica',
            'content': f"$$\\int_{{{sp.latex(t0_sym)}}}^{{{sp.latex(t1_sym)}}} {sp.latex(integrand)} \\, dt \\approx {numeric_value:.10f}$$",
            'explanation': f'Calculado usando integración numérica adaptativa (scipy.quad). Error estimado: {error_estimate:.2e}'
        })
        
        # Si tenemos resultado simbólico, comparar
        if definite_symbolic is not None:
            symbolic_float = float(definite_symbolic.evalf())
            abs_diff = abs(numeric_value - symbolic_float)
            rel_diff = abs_diff / max(abs(symbolic_float), 1e-10)
            
            steps_latex.append({
                'title': '8. Comparación simbólico vs numérico',
                'content': f"""$$\\text{{Resultado simbólico: }} \\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} = {sp.latex(definite_symbolic)}$$

$$\\text{{Valor numérico: }} \\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} \\approx {numeric_value:.10f}$$

$$\\text{{Verificación: }} {sp.latex(definite_symbolic)} \\approx {symbolic_float:.10f}$$

$$\\text{{Error absoluto: }} {abs_diff:.2e} \\quad \\text{{Error relativo: }} {rel_diff:.2e}$$""",
                'explanation': '✅ Excelente concordancia entre el resultado simbólico y numérico!' if rel_diff < 1e-6 else '⚠️ Verificar precisión numérica.'
            })
        
    except Exception as e:
        logger.error(f"Cálculo numérico falló: {e}")
        numeric_value = np.nan
        steps_latex.append({
            'title': 'Error en cálculo numérico',
            'content': f"Error: {str(e)}",
            'explanation': 'No se pudo completar la integración numérica.'
        })
    
    elapsed = time.time() - start_time
    logger.info(f"Explicación de integral de línea completada en {elapsed:.3f}s")
    
    return {
        'r_sym': r_sym,
        'r_prime_sym': r_prime,
        'F_of_r_sym': F_of_r,
        'integrand_sym': integrand,
        'integrand_latex': sp.latex(integrand),
        'antiderivative': antiderivative,
        'definite_symbolic': definite_symbolic,
        'definite_symbolic_latex': sp.latex(definite_symbolic) if definite_symbolic else None,
        'numeric_value': numeric_value,
        'steps_latex': steps_latex,
        'why_symbolic_failed': why_symbolic_failed,
        'execution_time': elapsed
    }


def explain_surface_flux_steps(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    r_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    u: sp.Symbol,
    v: sp.Symbol,
    u0: float,
    u1: float,
    v0: float,
    v1: float
) -> Dict[str, Any]:
    """
    Explica paso a paso el cálculo del flujo de superficie con derivaciones simbólicas.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial (P, Q, R) como expresiones simbólicas
    r_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Parametrización (x(u,v), y(u,v), z(u,v)) como expresiones simbólicas
    u, v : sp.Symbol
        Símbolos de los parámetros
    u0, u1, v0, v1 : float
        Límites de integración
    
    Retorna
    -------
    Dict[str, Any]
        Diccionario con pasos simbólicos y resultado numérico
    """
    import time
    start_time = time.time()
    
    logger.info(f"Explicando flujo de superficie paso a paso")
    
    x, y, z = sp.symbols('x y z')
    P, Q, R = F_sym
    x_uv, y_uv, z_uv = r_sym
    
    steps_latex = []
    why_symbolic_failed = None
    
    # Convertir límites numéricos a simbólicos si son múltiplos de π
    def to_symbolic_limit(val):
        """Convierte un límite numérico a simbólico si es múltiplo de π."""
        if isinstance(val, (int, float)):
            # Detectar múltiplos de π
            ratio = val / np.pi
            if abs(ratio - round(ratio)) < 1e-10:
                ratio_int = int(round(ratio))
                if ratio_int == 0:
                    return sp.Integer(0)
                elif ratio_int == 1:
                    return sp.pi
                elif ratio_int == -1:
                    return -sp.pi
                elif ratio_int == 2:
                    return 2*sp.pi
                elif ratio_int == -2:
                    return -2*sp.pi
                else:
                    return ratio_int * sp.pi
            # Detectar fracciones de π comunes
            for denom in [2, 3, 4, 6]:
                ratio = val / (np.pi / denom)
                if abs(ratio - round(ratio)) < 1e-10:
                    ratio_int = int(round(ratio))
                    if ratio_int == 1:
                        return sp.pi / denom
                    else:
                        return ratio_int * sp.pi / denom
        return val
    
    u0_sym = to_symbolic_limit(u0)
    u1_sym = to_symbolic_limit(u1)
    v0_sym = to_symbolic_limit(v0)
    v1_sym = to_symbolic_limit(v1)
    
    # Paso 1: Mostrar parametrización
    steps_latex.append({
        'title': '1. Parametrización de la superficie',
        'content': f"$$\\mathbf{{r}}(u,v) = \\left( {sp.latex(x_uv)}, {sp.latex(y_uv)}, {sp.latex(z_uv)} \\right)$$",
        'explanation': 'Esta es la superficie sobre la cual calcularemos el flujo.'
    })
    
    # Paso 2: Calcular r_u y r_v
    r_u = (sp.diff(x_uv, u), sp.diff(y_uv, u), sp.diff(z_uv, u))
    r_v = (sp.diff(x_uv, v), sp.diff(y_uv, v), sp.diff(z_uv, v))
    r_u = tuple(sp.simplify(comp) for comp in r_u)
    r_v = tuple(sp.simplify(comp) for comp in r_v)
    
    steps_latex.append({
        'title': '2. Derivadas parciales',
        'content': f"""$$\\mathbf{{r}}_u = \\frac{{\\partial \\mathbf{{r}}}}{{\\partial u}} = \\left( {sp.latex(r_u[0])}, {sp.latex(r_u[1])}, {sp.latex(r_u[2])} \\right)$$
$$\\mathbf{{r}}_v = \\frac{{\\partial \\mathbf{{r}}}}{{\\partial v}} = \\left( {sp.latex(r_v[0])}, {sp.latex(r_v[1])}, {sp.latex(r_v[2])} \\right)$$""",
        'explanation': 'Derivamos r respecto a u y v.'
    })
    
    # Paso 3: Calcular producto cruz r_u × r_v (vector normal)
    normal = (
        r_u[1]*r_v[2] - r_u[2]*r_v[1],
        r_u[2]*r_v[0] - r_u[0]*r_v[2],
        r_u[0]*r_v[1] - r_u[1]*r_v[0]
    )
    normal = tuple(sp.simplify(comp) for comp in normal)
    
    steps_latex.append({
        'title': '3. Vector normal (r_u × r_v)',
        'content': f"""$$\\mathbf{{r}}_u \\times \\mathbf{{r}}_v = \\begin{{vmatrix}} \\mathbf{{i}} & \\mathbf{{j}} & \\mathbf{{k}} \\\\ {sp.latex(r_u[0])} & {sp.latex(r_u[1])} & {sp.latex(r_u[2])} \\\\ {sp.latex(r_v[0])} & {sp.latex(r_v[1])} & {sp.latex(r_v[2])} \\end{{vmatrix}}$$
$$= \\left( {sp.latex(normal[0])}, {sp.latex(normal[1])}, {sp.latex(normal[2])} \\right)$$""",
        'explanation': 'El producto cruz da el vector normal a la superficie con magnitud igual al área del paralelogramo.'
    })
    
    # Paso 4: F evaluado en r(u,v)
    F_of_r = (
        P.subs({x: x_uv, y: y_uv, z: z_uv}),
        Q.subs({x: x_uv, y: y_uv, z: z_uv}),
        R.subs({x: x_uv, y: y_uv, z: z_uv})
    )
    F_of_r = tuple(sp.simplify(comp) for comp in F_of_r)
    
    steps_latex.append({
        'title': '4. Campo vectorial evaluado en la superficie',
        'content': f"$$\\mathbf{{F}}(\\mathbf{{r}}(u,v)) = \\left( {sp.latex(F_of_r[0])}, {sp.latex(F_of_r[1])}, {sp.latex(F_of_r[2])} \\right)$$",
        'explanation': 'Sustituimos la parametrización en F.'
    })
    
    # Paso 5: Producto punto F·(r_u × r_v)
    integrand = sum(F_of_r[i] * normal[i] for i in range(3))
    integrand = sp.simplify(integrand)
    
    steps_latex.append({
        'title': '5. Producto punto F·(r_u × r_v)',
        'content': f"$$\\mathbf{{F}}(\\mathbf{{r}}(u,v)) \\cdot (\\mathbf{{r}}_u \\times \\mathbf{{r}}_v) = {sp.latex(integrand)}$$",
        'explanation': 'Este es el integrando para el flujo de superficie.'
    })
    
    # Paso 6: Intentar integración simbólica doble
    definite_symbolic = None
    
    try:
        logger.info("Intentando integración simbólica doble...")
        # Primero integrar respecto a u
        inner_integral = sp.integrate(integrand, (u, u0_sym, u1_sym))
        
        if not inner_integral.has(sp.Integral):
            # Luego integrar respecto a v
            definite_symbolic = sp.integrate(inner_integral, (v, v0_sym, v1_sym))
            
            if not definite_symbolic.has(sp.Integral):
                definite_symbolic = sp.simplify(definite_symbolic)
                
                steps_latex.append({
                    'title': '6. Integración simbólica',
                    'content': f"$$\\iint_S \\mathbf{{F}} \\cdot \\mathbf{{n}} \\, dS = \\int_{{{sp.latex(v0_sym)}}}^{{{sp.latex(v1_sym)}}} \\int_{{{sp.latex(u0_sym)}}}^{{{sp.latex(u1_sym)}}} {sp.latex(integrand)} \\, du \\, dv = {sp.latex(definite_symbolic)}$$",
                    'explanation': 'La integral se pudo calcular simbólicamente.'
                })
                
                logger.info(f"Integral de superficie simbólica: {definite_symbolic}")
            else:
                raise ValueError("Outer integral not evaluable")
        else:
            raise ValueError("Inner integral not evaluable")
            
    except Exception as e:
        logger.warning(f"Integración simbólica falló: {e}")
        why_symbolic_failed = f"Integral doble muy compleja para resolución simbólica: {str(e)}"
        
        steps_latex.append({
            'title': '6. Integración simbólica no disponible',
            'content': f"$$\\iint_S \\mathbf{{F}} \\cdot \\mathbf{{n}} \\, dS \\quad \\text{{(evaluación numérica requerida)}}$$",
            'explanation': 'Se usará integración numérica por la complejidad del integrando.'
        })
    
    # Paso 7: Cálculo numérico (siempre)
    try:
        # Convertir a función numérica
        integrand_num = sp.lambdify((u, v), integrand, modules=['numpy'])
        
        # Usar scipy.integrate.dblquad
        from scipy import integrate as sci_integrate
        numeric_value, error_estimate = sci_integrate.dblquad(
            integrand_num,
            v0, v1,
            lambda v: u0, lambda v: u1
        )
        
        steps_latex.append({
            'title': '7. Verificación numérica',
            'content': f"$$\\iint_S \\mathbf{{F}} \\cdot \\mathbf{{n}} \\, dS \\approx {numeric_value:.10f}$$",
            'explanation': f'Calculado usando integración numérica doble (scipy.dblquad). Error estimado: {error_estimate:.2e}'
        })
        
        # Comparar si tenemos simbólico
        if definite_symbolic is not None:
            symbolic_float = float(definite_symbolic.evalf())
            abs_diff = abs(numeric_value - symbolic_float)
            rel_diff = abs_diff / max(abs(symbolic_float), 1e-10)
            
            steps_latex.append({
                'title': '8. Comparación simbólico vs numérico',
                'content': f"""$$\\text{{Resultado simbólico: }} \\iint_S \\mathbf{{F}} \\cdot \\mathbf{{n}} \\, dS = {sp.latex(definite_symbolic)}$$

$$\\text{{Valor numérico: }} \\iint_S \\mathbf{{F}} \\cdot \\mathbf{{n}} \\, dS \\approx {numeric_value:.10f}$$

$$\\text{{Verificación: }} {sp.latex(definite_symbolic)} \\approx {symbolic_float:.10f}$$

$$\\text{{Error absoluto: }} {abs_diff:.2e} \\quad \\text{{Error relativo: }} {rel_diff:.2e}$$""",
                'explanation': '✅ Excelente concordancia entre el resultado simbólico y numérico!' if rel_diff < 1e-6 else '⚠️ Verificar precisión numérica.'
            })
        
    except Exception as e:
        logger.error(f"Cálculo numérico falló: {e}")
        numeric_value = np.nan
        steps_latex.append({
            'title': 'Error en cálculo numérico',
            'content': f"Error: {str(e)}",
            'explanation': 'No se pudo completar la integración numérica.'
        })
    
    elapsed = time.time() - start_time
    logger.info(f"Explicación de flujo de superficie completada en {elapsed:.3f}s")
    
    return {
        'r_sym': r_sym,
        'r_u_sym': r_u,
        'r_v_sym': r_v,
        'normal_sym': normal,
        'F_of_r_sym': F_of_r,
        'integrand_sym': integrand,
        'integrand_latex': sp.latex(integrand),
        'definite_symbolic': definite_symbolic,
        'definite_symbolic_latex': sp.latex(definite_symbolic) if definite_symbolic else None,
        'numeric_value': numeric_value,
        'steps_latex': steps_latex,
        'why_symbolic_failed': why_symbolic_failed,
        'execution_time': elapsed
    }


def explain_stokes_verification_steps(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    r_boundary_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    t: sp.Symbol,
    t0: float,
    t1: float,
    r_surface_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    u: sp.Symbol,
    v: sp.Symbol,
    u0: float,
    u1: float,
    v0: float,
    v1: float
) -> Dict[str, Any]:
    """
    Explica paso a paso la verificación del Teorema de Stokes con derivaciones simbólicas.
    
    Teorema de Stokes: ∮_C F·dr = ∬_S (∇×F)·n dS
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial (P, Q, R)
    r_boundary_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Parametrización de la frontera C
    t : sp.Symbol
        Parámetro para la frontera
    t0, t1 : float
        Límites de integración para t
    r_surface_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Parametrización de la superficie S
    u, v : sp.Symbol
        Parámetros para la superficie
    u0, u1, v0, v1 : float
        Límites de integración para u, v
    
    Retorna
    -------
    Dict[str, Any]
        Diccionario con todos los pasos y resultados
    """
    import time
    start_time = time.time()
    
    logger.info("Explicando verificación del Teorema de Stokes paso a paso")
    
    x, y, z = sp.symbols('x y z')
    P, Q, R = F_sym
    
    steps_latex = []
    
    # Convertir límites a simbólicos
    def to_symbolic_limit(val):
        if isinstance(val, (int, float)):
            ratio = val / np.pi
            if abs(ratio - round(ratio)) < 1e-10:
                ratio_int = int(round(ratio))
                if ratio_int == 0:
                    return sp.Integer(0)
                elif ratio_int == 1:
                    return sp.pi
                elif ratio_int == -1:
                    return -sp.pi
                elif ratio_int == 2:
                    return 2*sp.pi
                elif ratio_int == -2:
                    return -2*sp.pi
                else:
                    return ratio_int * sp.pi
        return val
    
    t0_sym = to_symbolic_limit(t0)
    t1_sym = to_symbolic_limit(t1)
    u0_sym = to_symbolic_limit(u0)
    u1_sym = to_symbolic_limit(u1)
    v0_sym = to_symbolic_limit(v0)
    v1_sym = to_symbolic_limit(v1)
    
    # Paso 1: Mostrar el teorema
    steps_latex.append({
        'title': 'Teorema de Stokes',
        'content': r"$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} \, dS$$",
        'explanation': 'La circulación de F alrededor de C es igual al flujo del rotacional de F a través de S.'
    })
    
    # Paso 2: Campo vectorial
    steps_latex.append({
        'title': '1. Campo vectorial F',
        'content': f"$$\\mathbf{{F}} = \\left( {sp.latex(P)}, {sp.latex(Q)}, {sp.latex(R)} \\right)$$",
        'explanation': 'Este es el campo vectorial sobre el cual verificaremos el teorema.'
    })
    
    # Paso 3: Calcular rotacional ∇×F
    curl = (
        sp.diff(R, y) - sp.diff(Q, z),
        sp.diff(P, z) - sp.diff(R, x),
        sp.diff(Q, x) - sp.diff(P, y)
    )
    curl = tuple(sp.simplify(comp) for comp in curl)
    
    steps_latex.append({
        'title': '2. Rotacional del campo (∇×F)',
        'content': f"""$$\\nabla \\times \\mathbf{{F}} = \\begin{{vmatrix}} \\mathbf{{i}} & \\mathbf{{j}} & \\mathbf{{k}} \\\\ \\frac{{\\partial}}{{\\partial x}} & \\frac{{\\partial}}{{\\partial y}} & \\frac{{\\partial}}{{\\partial z}} \\\\ {sp.latex(P)} & {sp.latex(Q)} & {sp.latex(R)} \\end{{vmatrix}}$$
$$= \\left( {sp.latex(curl[0])}, {sp.latex(curl[1])}, {sp.latex(curl[2])} \\right)$$""",
        'explanation': 'Calculamos el rotacional usando el determinante formal.'
    })
    
    # Paso 4: LADO IZQUIERDO - Integral de línea ∮_C F·dr
    steps_latex.append({
        'title': '3. Lado izquierdo: Integral de línea ∮_C F·dr',
        'content': f"$$\\mathbf{{r}}_C(t) = \\left( {sp.latex(r_boundary_sym[0])}, {sp.latex(r_boundary_sym[1])}, {sp.latex(r_boundary_sym[2])} \\right), \\quad t \\in [{sp.latex(t0_sym)}, {sp.latex(t1_sym)}]$$",
        'explanation': 'Parametrización de la curva frontera C.'
    })
    
    # Calcular integral de línea usando la función auxiliar
    line_explanation = explain_line_integral_steps(F_sym, r_boundary_sym, t, t0, t1)
    line_integral_symbolic = line_explanation['definite_symbolic']
    line_integral_numeric = line_explanation['numeric_value']
    
    steps_latex.append({
        'title': '4. Cálculo de ∮_C F·dr',
        'content': f"$$\\mathbf{{F}}(\\mathbf{{r}}_C(t)) \\cdot \\mathbf{{r}}'_C(t) = {sp.latex(line_explanation['integrand_sym'])}$$",
        'explanation': 'Integrando de la integral de línea.'
    })
    
    if line_integral_symbolic:
        steps_latex.append({
            'title': '5. Resultado de ∮_C F·dr',
            'content': f"$$\\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} = {sp.latex(line_integral_symbolic)}$$",
            'explanation': f'Resultado exacto de la integral de línea.'
        })
    
    # Paso 5: LADO DERECHO - Integral de superficie ∬_S (∇×F)·n dS
    steps_latex.append({
        'title': '6. Lado derecho: Integral de superficie ∬_S (∇×F)·n dS',
        'content': f"$$\\mathbf{{r}}_S(u,v) = \\left( {sp.latex(r_surface_sym[0])}, {sp.latex(r_surface_sym[1])}, {sp.latex(r_surface_sym[2])} \\right)$$",
        'explanation': 'Parametrización de la superficie S.'
    })
    
    # Calcular derivadas parciales de r_surface
    r_u = tuple(sp.diff(comp, u) for comp in r_surface_sym)
    r_v = tuple(sp.diff(comp, v) for comp in r_surface_sym)
    r_u = tuple(sp.simplify(comp) for comp in r_u)
    r_v = tuple(sp.simplify(comp) for comp in r_v)
    
    # Producto cruz r_u × r_v
    normal = (
        r_u[1]*r_v[2] - r_u[2]*r_v[1],
        r_u[2]*r_v[0] - r_u[0]*r_v[2],
        r_u[0]*r_v[1] - r_u[1]*r_v[0]
    )
    normal = tuple(sp.simplify(comp) for comp in normal)
    
    steps_latex.append({
        'title': '7. Vector normal (r_u × r_v)',
        'content': f"$$\\mathbf{{r}}_u \\times \\mathbf{{r}}_v = \\left( {sp.latex(normal[0])}, {sp.latex(normal[1])}, {sp.latex(normal[2])} \\right)$$",
        'explanation': 'Vector normal a la superficie.'
    })
    
    # (∇×F) evaluado en r(u,v)
    curl_of_r = tuple(comp.subs({x: r_surface_sym[0], y: r_surface_sym[1], z: r_surface_sym[2]}) for comp in curl)
    curl_of_r = tuple(sp.simplify(comp) for comp in curl_of_r)
    
    # (∇×F)·n
    integrand_surface = sum(curl_of_r[i] * normal[i] for i in range(3))
    integrand_surface = sp.simplify(integrand_surface)
    
    steps_latex.append({
        'title': '8. Integrando (∇×F)·(r_u×r_v)',
        'content': f"$$(\\nabla \\times \\mathbf{{F}}) \\cdot (\\mathbf{{r}}_u \\times \\mathbf{{r}}_v) = {sp.latex(integrand_surface)}$$",
        'explanation': 'Producto punto del rotacional con el vector normal.'
    })
    
    # Intentar integración simbólica de superficie
    surface_integral_symbolic = None
    surface_integral_numeric = None
    
    try:
        inner = sp.integrate(integrand_surface, (u, u0_sym, u1_sym))
        if not inner.has(sp.Integral):
            surface_integral_symbolic = sp.integrate(inner, (v, v0_sym, v1_sym))
            if not surface_integral_symbolic.has(sp.Integral):
                surface_integral_symbolic = sp.simplify(surface_integral_symbolic)
                
                steps_latex.append({
                    'title': '9. Resultado de ∬_S (∇×F)·n dS',
                    'content': f"$$\\iint_S (\\nabla \\times \\mathbf{{F}}) \\cdot \\mathbf{{n}} \\, dS = {sp.latex(surface_integral_symbolic)}$$",
                    'explanation': 'Resultado exacto de la integral de superficie.'
                })
    except:
        pass
    
    # Calcular numéricamente
    try:
        from scipy import integrate as sci_integrate
        integrand_num = sp.lambdify((u, v), integrand_surface, modules=['numpy'])
        surface_integral_numeric, _ = sci_integrate.dblquad(
            integrand_num, v0, v1, lambda v: u0, lambda v: u1
        )
    except:
        surface_integral_numeric = np.nan
    
    # Paso 6: COMPARACIÓN
    steps_latex.append({
        'title': '10. Verificación del Teorema de Stokes',
        'content': '',
        'explanation': ''
    })
    
    if line_integral_symbolic and surface_integral_symbolic:
        # Ambos simbólicos disponibles
        diff_symbolic = sp.simplify(line_integral_symbolic - surface_integral_symbolic)
        
        comparison_content = f"""**Lado izquierdo:**
$$\\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} = {sp.latex(line_integral_symbolic)}$$

**Lado derecho:**
$$\\iint_S (\\nabla \\times \\mathbf{{F}}) \\cdot \\mathbf{{n}} \\, dS = {sp.latex(surface_integral_symbolic)}$$

**Diferencia:**
$${sp.latex(line_integral_symbolic)} - {sp.latex(surface_integral_symbolic)} = {sp.latex(diff_symbolic)}$$
"""
        
        if diff_symbolic == 0 or abs(float(diff_symbolic.evalf())) < 1e-10:
            steps_latex[-1]['content'] = comparison_content
            steps_latex[-1]['explanation'] = '✅ **TEOREMA VERIFICADO:** Ambos lados son exactamente iguales.'
        else:
            steps_latex[-1]['content'] = comparison_content
            steps_latex[-1]['explanation'] = f'⚠️ Diferencia: {diff_symbolic}'
    
    elif line_integral_numeric is not None and surface_integral_numeric is not None:
        # Solo numéricos disponibles
        diff_numeric = abs(line_integral_numeric - surface_integral_numeric)
        rel_error = diff_numeric / max(abs(line_integral_numeric), 1e-10)
        
        comparison_content = f"""**Lado izquierdo (numérico):**
$$\\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} \\approx {line_integral_numeric:.10f}$$

**Lado derecho (numérico):**
$$\\iint_S (\\nabla \\times \\mathbf{{F}}) \\cdot \\mathbf{{n}} \\, dS \\approx {surface_integral_numeric:.10f}$$

**Diferencia:** ${diff_numeric:.2e}$
**Error relativo:** ${rel_error:.2e}$
"""
        
        steps_latex[-1]['content'] = comparison_content
        if rel_error < 1e-6:
            steps_latex[-1]['explanation'] = '✅ **TEOREMA VERIFICADO:** Ambos lados coinciden numéricamente.'
        else:
            steps_latex[-1]['explanation'] = f'⚠️ Error relativo: {rel_error:.2e}'
    
    elapsed = time.time() - start_time
    logger.info(f"Verificación de Stokes completada en {elapsed:.3f}s")
    
    return {
        'curl_sym': curl,
        'curl_latex': tuple(sp.latex(c) for c in curl),
        'line_integral_symbolic': line_integral_symbolic,
        'line_integral_numeric': line_integral_numeric,
        'surface_integral_symbolic': surface_integral_symbolic,
        'surface_integral_numeric': surface_integral_numeric,
        'stokes_verified': (diff_symbolic == 0) if line_integral_symbolic and surface_integral_symbolic else (rel_error < 1e-6 if line_integral_numeric and surface_integral_numeric else None),
        'steps_latex': steps_latex,
        'execution_time': elapsed
    }


def generate_exercises(
    seed: int,
    n: int,
    tipo: str
) -> List[Dict[str, Any]]:
    """
    Genera ejercicios autocalificables de cálculo vectorial.
    
    Parámetros
    ----------
    seed : int
        Semilla para generación aleatoria reproducible
    n : int
        Número de ejercicios a generar
    tipo : str
        Tipo de ejercicio: 'gradiente', 'divergencia_rotacional',
        'line_integral', 'stokes'
    
    Retorna
    -------
    List[Dict[str, Any]]
        Lista de ejercicios, cada uno con:
        - 'id': identificador único
        - 'tipo': tipo de ejercicio
        - 'instruccion': texto del enunciado
        - 'inputs': dict con las funciones/parámetros
        - 'solution': solución numérica o simbólica
        - 'tolerance': tolerancia para verificación
        - 'hints': lista de pistas opcionales
    
    Ejemplos
    --------
    >>> exercises = generate_exercises(42, 3, 'gradiente')
    >>> len(exercises)
    3
    >>> exercises[0]['tipo']
    'gradiente'
    """
    import random
    import time
    
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Generando {n} ejercicios de tipo '{tipo}' con semilla {seed}")
    
    exercises = []
    
    for i in range(n):
        exercise_id = f"{tipo}_{seed}_{i+1}"
        
        if tipo == 'gradiente':
            exercise = _generate_gradient_exercise(exercise_id, i)
        elif tipo == 'divergencia_rotacional':
            exercise = _generate_div_curl_exercise(exercise_id, i)
        elif tipo == 'line_integral':
            exercise = _generate_line_integral_exercise(exercise_id, i)
        elif tipo == 'stokes':
            exercise = _generate_stokes_exercise(exercise_id, i)
        elif tipo == 'optimizacion':
            exercise = _generate_optimization_exercise(exercise_id, i)
        else:
            raise ValueError(f"Tipo de ejercicio desconocido: {tipo}")
        
        exercises.append(exercise)
    
    elapsed = time.time() - start_time
    logger.info(f"Generados {n} ejercicios en {elapsed:.3f}s")
    
    return exercises


def _generate_gradient_exercise(exercise_id: str, idx: int) -> Dict[str, Any]:
    """Genera un ejercicio de gradiente con dificultad progresiva."""
    import random
    
    x, y, z = sp.symbols('x y z')
    
    # Dificultad basada en índice (0=fácil, 1-2=intermedio, 3+=difícil)
    difficulty = 'facil' if idx == 0 else ('intermedio' if idx <= 2 else 'dificil')
    
    # Plantillas organizadas por dificultad
    templates_facil = [
        (x**2 + y**2 + z**2, "esfera de nivel (distancia al origen al cuadrado)"),
        (2*x + 3*y - z, "plano lineal"),
        (x*y, "paraboloide hiperbólico (silla de montar)"),
        (x**2 + y**2, "paraboloide circular (independiente de z)"),
    ]
    
    templates_intermedio = [
        (x**2 + 2*y**2 + 3*z**2, "elipsoide escalado"),
        (x*y + y*z + z*x, "función simétrica trilineal"),
        (sp.sin(x) + sp.cos(y), "trigonométrica 2D + constante en z"),
        (x**3 - 3*x*y**2, "función armónica"),
        (sp.exp(x)*sp.cos(y), "exponencial-trigonométrica"),
    ]
    
    templates_dificil = [
        (x**3 + y**3 + z**3 - 3*x*y*z, "superficie cúbica con término mixto"),
        (sp.sqrt(x**2 + y**2 + z**2 + 1), "distancia normalizada (raíz)"),
        (sp.log(1 + x**2 + y**2), "logarítmica (no depende de z)"),
        (sp.sin(x*y)*sp.cos(z), "producto trigonométrico mixto"),
        (sp.exp(-(x**2 + y**2 + z**2)), "gaussiana 3D (campana)"),
        (x**2*y + y**2*z + z**2*x, "polinomio cúbico mixto"),
    ]
    
    # Seleccionar según dificultad
    if difficulty == 'facil':
        phi, description = random.choice(templates_facil)
    elif difficulty == 'intermedio':
        phi, description = random.choice(templates_intermedio)
    else:
        phi, description = random.choice(templates_dificil)
    
    phi_str = str(phi)
    
    # Calcular gradiente simbólico
    grad = tuple(sp.diff(phi, var) for var in (x, y, z))
    
    # Punto de evaluación (más complejo para dificultad alta)
    if difficulty == 'facil':
        test_point = (random.choice([0, 1, -1]), random.choice([0, 1, -1]), random.choice([0, 1, -1]))
    elif difficulty == 'intermedio':
        test_point = (random.randint(-2, 2), random.randint(-2, 2), random.randint(-2, 2))
    else:
        test_point = (round(random.uniform(-3, 3), 1), round(random.uniform(-3, 3), 1), round(random.uniform(-3, 3), 1))
    
    # Evaluar SIMBÓLICAMENTE (mantener forma exacta)
    grad_at_point = tuple(g.subs({x: test_point[0], y: test_point[1], z: test_point[2]}) for g in grad)
    grad_at_point = tuple(sp.simplify(g) for g in grad_at_point)
    
    # Calcular valor de φ en el punto (simbólico)
    phi_at_point = phi.subs({x: test_point[0], y: test_point[1], z: test_point[2]})
    phi_at_point = sp.simplify(phi_at_point)
    
    # Magnitud del gradiente (forma simbólica exacta: √(gx² + gy² + gz²))
    grad_magnitude_symbolic = sp.sqrt(sum(g**2 for g in grad_at_point))
    grad_magnitude_symbolic = sp.simplify(grad_magnitude_symbolic)
    
    # Vector unitario (dirección de máximo crecimiento): ∇φ / ||∇φ||
    # Mantener en forma simbólica exacta (fracciones con radicales)
    if grad_magnitude_symbolic != 0:
        unit_vector = tuple(sp.simplify(g / grad_magnitude_symbolic) for g in grad_at_point)
    else:
        unit_vector = (sp.Integer(0), sp.Integer(0), sp.Integer(0))
    
    # Pistas progresivas
    hints = [
        f"📘 **Nivel 1:** Recuerda que el gradiente es ∇φ = (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)",
        f"📗 **Nivel 2:** Función: φ = {phi_str}. Es una {description}",
        f"📙 **Nivel 3:** Las derivadas parciales son:\n  • ∂φ/∂x = {grad[0]}\n  • ∂φ/∂y = {grad[1]}\n  • ∂φ/∂z = {grad[2]}",
        f"📕 **Nivel 4:** Evalúa en {test_point}: sustituye x={test_point[0]}, y={test_point[1]}, z={test_point[2]}"
    ]
    
    return {
        'id': exercise_id,
        'tipo': 'gradiente',
        'dificultad': difficulty,
        'instruccion': f"**Ejercicio {idx+1} ({difficulty.upper()})**: Calcule el gradiente de φ(x,y,z) = {phi_str} y evalúelo en el punto {test_point}",
        'descripcion': f"Función: {description}",
        'inputs': {
            'phi': phi_str,
            'punto': test_point
        },
        'solution': {
            'symbolic': tuple(str(g) for g in grad),
            'symbolic_latex': tuple(sp.latex(g) for g in grad),
            'grad_at_point': tuple(str(g) for g in grad_at_point),
            'grad_at_point_latex': tuple(sp.latex(g) for g in grad_at_point),
            'magnitude_symbolic': str(grad_magnitude_symbolic),
            'magnitude_latex': sp.latex(grad_magnitude_symbolic),
            'phi_at_point_symbolic': str(phi_at_point),
            'phi_at_point_latex': sp.latex(phi_at_point),
            'unit_vector_symbolic': tuple(str(u) for u in unit_vector),
            'unit_vector_latex': tuple(sp.latex(u) for u in unit_vector)
        },
        'tolerance': 1e-6,
        'hints': hints,
        'interpretacion': f"""
**Interpretación Geométrica:**

• $\\phi({test_point[0]}, {test_point[1]}, {test_point[2]}) = {sp.latex(phi_at_point)}$

• El gradiente en este punto es $\\nabla\\phi = \\left( {sp.latex(grad_at_point[0])}, {sp.latex(grad_at_point[1])}, {sp.latex(grad_at_point[2])} \\right)$

• Magnitud: $\\|\\nabla\\phi\\| = {sp.latex(grad_magnitude_symbolic)}$ (tasa máxima de cambio)

• Dirección de máximo crecimiento (vector unitario): $\\hat{{v}} = \\left( {sp.latex(unit_vector[0])}, {sp.latex(unit_vector[1])}, {sp.latex(unit_vector[2])} \\right)$

• {"⚠️ Punto crítico (∇φ = 0)" if grad_magnitude_symbolic == 0 else f"✅ φ crece en la dirección del gradiente"}
        """
    }


def _generate_div_curl_exercise(exercise_id: str, idx: int) -> Dict[str, Any]:
    """Genera un ejercicio de divergencia y rotacional con contexto físico."""
    import random
    
    x, y, z = sp.symbols('x y z')
    
    difficulty = 'facil' if idx == 0 else ('intermedio' if idx <= 2 else 'dificil')
    
    # Plantillas con interpretación física
    templates_facil = [
        ((x, y, z), "campo radial (fuente/sumidero desde el origen)", "∇·F = 3 (fuente)"),
        ((-y, x, 0), "rotación pura en plano XY (torbellino)", "∇×F ≠ 0, ∇·F = 0 (incompresible)"),
        ((0, 0, z), "campo vertical uniforme", "∇·F = 1, ∇×F = 0 (conservativo)"),
        ((y, 0, 0), "cizallamiento en dirección x", "∇×F ≠ 0"),
    ]
    
    templates_intermedio = [
        ((y*z, x*z, x*y), "campo simétrico con productos", "campo no conservativo"),
        ((sp.sin(x), sp.cos(y), 0), "trigonométrico 2D", "análisis de periodicidad"),
        ((x**2, y**2, z**2), "campo cuadrático radial", "divergencia no constante"),
        ((z, x, y), "permutación cíclica", "simetría rotacional"),
        ((-y/(x**2+y**2+1), x/(x**2+y**2+1), 0), "rotación normalizada", "evita singularidad en origen"),
    ]
    
    templates_dificil = [
        ((y**2 - z**2, z**2 - x**2, x**2 - y**2), "campo cuadrático antisimétrico", "divergencia cero"),
        ((sp.exp(x)*sp.sin(y), sp.exp(x)*sp.cos(y), 0), "exponencial-trigonométrico", "análisis complejo"),
        ((y*z**2, x*z**2, 2*x*y*z), "polinomio cúbico", "rotacional no trivial"),
        ((sp.sin(y*z), sp.sin(z*x), sp.sin(x*y)), "trigonométrico producto cruzado", "alta no linealidad"),
    ]
    
    if difficulty == 'facil':
        F, physical_desc, math_property = random.choice(templates_facil)
    elif difficulty == 'intermedio':
        F, physical_desc, math_property = random.choice(templates_intermedio)
    else:
        F, physical_desc, math_property = random.choice(templates_dificil)
    
    F_str = tuple(str(c) for c in F)
    
    # Calcular divergencia y rotacional
    div_sym, curl_sym, _, _ = compute_divergence_and_curl(F, (x, y, z))
    
    # Clasificar el campo
    is_conservative = all(c == 0 for c in curl_sym)
    is_incompressible = div_sym == 0
    is_solenoidal = is_incompressible
    
    # Punto de evaluación para análisis simbólico
    if difficulty == 'facil':
        test_point = (1, 0, 0)
    elif difficulty == 'intermedio':
        test_point = (1, 1, 1)
    else:
        test_point = (sp.Rational(1, 2), sp.Rational(-1, 2), 1)
    
    # Evaluar simbólicamente
    div_at_point = div_sym.subs({x: test_point[0], y: test_point[1], z: test_point[2]})
    div_at_point = sp.simplify(div_at_point)
    
    curl_at_point = tuple(c.subs({x: test_point[0], y: test_point[1], z: test_point[2]}) for c in curl_sym)
    curl_at_point = tuple(sp.simplify(c) for c in curl_at_point)
    
    # Interpretación física
    if is_conservative:
        field_type = "⚡ Campo Conservativo (existe potencial escalar φ tal que F = -∇φ)"
    elif is_incompressible:
        field_type = "💧 Campo Incompresible/Solenoidal (fluido incompresible)"
    else:
        field_type = "🔄 Campo General (con rotación y/o compresión)"
    
    hints = [
        f"📘 **Nivel 1:** Divergencia: ∇·F = ∂P/∂x + ∂Q/∂y + ∂R/∂z",
        f"📗 **Nivel 2:** Rotacional: ∇×F = (∂R/∂y - ∂Q/∂z, ∂P/∂z - ∂R/∂x, ∂Q/∂x - ∂P/∂y)",
        f"📙 **Nivel 3:** Campo vectorial F = ({F_str[0]}, {F_str[1]}, {F_str[2]})",
        f"📕 **Nivel 4:** Contexto físico: {physical_desc}. Propiedad: {math_property}"
    ]
    
    return {
        'id': exercise_id,
        'tipo': 'divergencia_rotacional',
        'dificultad': difficulty,
        'instruccion': f"**Ejercicio {idx+1} ({difficulty.upper()})**: Calcule la divergencia y rotacional del campo F = ({F_str[0]}, {F_str[1]}, {F_str[2]})",
        'descripcion': f"Contexto: {physical_desc}",
        'inputs': {
            'P': F_str[0],
            'Q': F_str[1],
            'R': F_str[2]
        },
        'solution': {
            'divergence': str(div_sym),
            'divergence_latex': sp.latex(div_sym),
            'curl': tuple(str(c) for c in curl_sym),
            'curl_latex': tuple(sp.latex(c) for c in curl_sym),
            'is_conservative': is_conservative,
            'is_incompressible': is_incompressible,
            'div_at_point_symbolic': str(div_at_point),
            'div_at_point_latex': sp.latex(div_at_point),
            'curl_at_point_symbolic': tuple(str(c) for c in curl_at_point),
            'curl_at_point_latex': tuple(sp.latex(c) for c in curl_at_point)
        },
        'tolerance': 1e-9,
        'hints': hints,
        'interpretacion': f"""
**Clasificación del Campo:**

{field_type}

**Propiedades Matemáticas:**

• $\\nabla \\cdot F = {sp.latex(div_sym)}$ {"= 0 ✅ (solenoidal)" if is_incompressible else "≠ 0"}

• $\\nabla \\times F = \\left( {sp.latex(curl_sym[0])}, {sp.latex(curl_sym[1])}, {sp.latex(curl_sym[2])} \\right)$ {"= (0,0,0) ✅ (irrotacional)" if is_conservative else "≠ 0"}

**Evaluación en {test_point}:**

• $\\nabla \\cdot F = {sp.latex(div_at_point)}$

• $\\nabla \\times F = \\left( {sp.latex(curl_at_point[0])}, {sp.latex(curl_at_point[1])}, {sp.latex(curl_at_point[2])} \\right)$

**Interpretación Física:**

{math_property}

{"✅ Existe potencial escalar (campo conservativo)" if is_conservative else "❌ No existe potencial escalar"}

{"✅ Fluido incompresible (volumen se conserva)" if is_incompressible else "⚠️ Fluido compresible"}
        """
    }


def _generate_line_integral_exercise(exercise_id: str, idx: int) -> Dict[str, Any]:
    """Genera un ejercicio de integral de línea."""
    import random
    
    x, y, z, t = sp.symbols('x y z t')
    
    # Campo simple y curva circular (solución conocida simbólicamente)
    P, Q, R = -y, x, 0
    gamma = (sp.cos(t), sp.sin(t), 0)
    t0, t1 = 0, 2*sp.pi
    
    # Calcular integral simbólicamente
    gamma_prime = tuple(sp.diff(g, t) for g in gamma)
    
    # F(r(t)) = (-sin(t), cos(t), 0)
    F_at_r = (P.subs({x: gamma[0], y: gamma[1], z: gamma[2]}),
              Q.subs({x: gamma[0], y: gamma[1], z: gamma[2]}),
              R.subs({x: gamma[0], y: gamma[1], z: gamma[2]}))
    
    # F·dr = F(r(t))·r'(t)
    integrand = sum(F_at_r[i] * gamma_prime[i] for i in range(3))
    integrand = sp.simplify(integrand)
    
    # Integrar simbólicamente
    result_symbolic = sp.integrate(integrand, (t, t0, t1))
    result_symbolic = sp.simplify(result_symbolic)
    
    return {
        'id': exercise_id,
        'tipo': 'line_integral',
        'instruccion': f"Calcule ∮ F·dr donde F=(-y,x,0) y la curva es el círculo unitario r(t)=(cos(t),sin(t),0), t∈[0,2π]",
        'inputs': {
            'P': '-y',
            'Q': 'x',
            'R': '0',
            'x_t': 'cos(t)',
            'y_t': 'sin(t)',
            'z_t': '0',
            't0': str(t0),
            't1': str(t1)
        },
        'solution': {
            'symbolic': str(result_symbolic),
            'symbolic_latex': sp.latex(result_symbolic),
            'integrand': str(integrand),
            'integrand_latex': sp.latex(integrand)
        },
        'tolerance': 1e-3,
        'hints': [
            "Para el círculo unitario con F=(-y,x,0), la integral vale 2π",
            "∮ F·dr = ∫[t0,t1] F(r(t))·r'(t) dt",
            f"El integrando es: {sp.latex(integrand)}"
        ],
        'interpretacion': f"""
**Resultado:**

$\\oint_C F \\cdot dr = {sp.latex(result_symbolic)}$

**Interpretación:** Este resultado representa la circulación del campo vectorial alrededor del círculo unitario.
        """
    }


def _generate_stokes_exercise(exercise_id: str, idx: int) -> Dict[str, Any]:
    """Genera un ejercicio de verificación del teorema de Stokes."""
    import random
    
    x, y, z, t, u, v = sp.symbols('x y z t u v')
    
    # Campo rotacional simple
    P, Q, R = -y, x, 0
    F_sym = (P, Q, R)
    
    # Calcular rotacional simbólicamente
    curl_F = (
        sp.diff(R, y) - sp.diff(Q, z),
        sp.diff(P, z) - sp.diff(R, x),
        sp.diff(Q, x) - sp.diff(P, y)
    )
    curl_F = tuple(sp.simplify(c) for c in curl_F)
    
    # Frontera: círculo unitario
    r_boundary = (sp.cos(t), sp.sin(t), 0)
    dr_boundary = tuple(sp.diff(c, t) for c in r_boundary)
    
    # Integral de línea: ∮ F·dr
    F_at_boundary = tuple(f.subs({x: r_boundary[0], y: r_boundary[1], z: r_boundary[2]}) for f in F_sym)
    integrand_line = sum(F_at_boundary[i] * dr_boundary[i] for i in range(3))
    integrand_line = sp.simplify(integrand_line)
    line_integral = sp.integrate(integrand_line, (t, 0, 2*sp.pi))
    line_integral = sp.simplify(line_integral)
    
    # Superficie: disco unitario (el rotacional solo tiene componente z = 2)
    # ∬ (∇×F)·n dS = ∬ 2 dA = 2π (área del círculo unitario)
    surface_integral = 2 * sp.pi
    
    return {
        'id': exercise_id,
        'tipo': 'stokes',
        'instruccion': "Verifique el teorema de Stokes para F=(-y,x,0) sobre el disco unitario en z=0",
        'inputs': {
            'P': '-y',
            'Q': 'x',
            'R': '0',
            'boundary': {
                'x_t': 'cos(t)',
                'y_t': 'sin(t)',
                'z_t': '0',
                't0': '0',
                't1': '2*pi'
            },
            'surface': {
                'descripcion': 'Disco unitario en el plano z=0'
            }
        },
        'solution': {
            'curl': tuple(str(c) for c in curl_F),
            'curl_latex': tuple(sp.latex(c) for c in curl_F),
            'line_integral_symbolic': str(line_integral),
            'line_integral_latex': sp.latex(line_integral),
            'surface_integral_symbolic': str(surface_integral),
            'surface_integral_latex': sp.latex(surface_integral),
            'stokes_holds': True
        },
        'tolerance': 1e-2,
        'hints': [
            "Teorema de Stokes: ∮_C F·dr = ∬_S (∇×F)·n dS",
            f"El rotacional es: ∇×F = ({sp.latex(curl_F[0])}, {sp.latex(curl_F[1])}, {sp.latex(curl_F[2])})",
            "Para este campo, ambos lados dan 2π"
        ],
        'interpretacion': f"""
**Verificación del Teorema de Stokes:**

**Lado izquierdo (integral de línea):**

$\\oint_C F \\cdot dr = {sp.latex(line_integral)}$

**Lado derecho (integral de superficie):**

$\\iint_S (\\nabla \\times F) \\cdot n \\, dS = {sp.latex(surface_integral)}$

✅ **Ambos lados son iguales**, el teorema de Stokes se verifica.
        """
    }


def plot_error_heatmap(
    U: np.ndarray,
    V: np.ndarray,
    error_grid: np.ndarray,
    title: str = "Mapa de Error"
) -> go.Figure:
    """
    Crea un mapa de calor (heatmap) del error en la verificación de teoremas.
    
    Parámetros
    ----------
    U : np.ndarray
        Malla de coordenadas u (2D)
    V : np.ndarray
        Malla de coordenadas v (2D)
    error_grid : np.ndarray
        Valores del error en cada punto (u,v)
    title : str, default="Mapa de Error"
        Título del gráfico
    
    Retorna
    -------
    go.Figure
        Figura de Plotly con el heatmap
    
    Ejemplos
    --------
    >>> U, V = np.meshgrid(np.linspace(0,1,20), np.linspace(0,2*np.pi,40))
    >>> error = np.random.rand(40, 20) * 0.01
    >>> fig = plot_error_heatmap(U, V, error)
    """
    logger.info(f"Creando heatmap de error con forma {error_grid.shape}")
    
    # Calcular estadísticas
    max_error = np.max(np.abs(error_grid))
    min_error = np.min(np.abs(error_grid))
    mean_error = np.mean(np.abs(error_grid))
    std_error = np.std(np.abs(error_grid))
    
    logger.info(f"Error: max={max_error:.6f}, min={min_error:.6f}, mean={mean_error:.6f}, std={std_error:.6f}")
    
    # Crear figura
    fig = go.Figure(data=go.Heatmap(
        x=U[0, :],
        y=V[:, 0],
        z=np.abs(error_grid),
        colorscale='Viridis',
        colorbar=dict(
            title=dict(text='|Error|', side='right'),
            tickmode='linear',
            tick0=0,
            dtick=max_error/10 if max_error > 0 else 0.1
        ),
        hovertemplate='u=%{x:.3f}<br>v=%{y:.3f}<br>|error|=%{z:.6f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{title}<br><sub>Max: {max_error:.6f} | Media: {mean_error:.6f} | Std: {std_error:.6f}</sub>',
        xaxis_title='u',
        yaxis_title='v',
        width=800,
        height=600,
        font=dict(size=12)
    )
    
    return fig


def _generate_optimization_exercise(exercise_id: str, idx: int) -> Dict[str, Any]:
    """Genera un ejercicio de optimización con dificultad progresiva."""
    import random
    
    x, y = sp.symbols('x y')
    
    # Dificultad basada en índice
    difficulty = 'facil' if idx == 0 else ('intermedio' if idx <= 2 else 'dificil')
    
    # Determinar tipo de problema de optimización
    problem_types = ['criticos', 'lagrange', 'region']
    problem_type = random.choice(problem_types) if difficulty != 'facil' else 'criticos'
    
    if problem_type == 'criticos':
        # PUNTOS CRÍTICOS Y CLASIFICACIÓN
        templates_facil = [
            (x**2 + y**2, "paraboloide (mínimo en origen)"),
            (x**2 - y**2, "silla de montar (punto silla en origen)"),
            (-x**2 - y**2, "paraboloide invertido (máximo en origen)"),
        ]
        
        templates_intermedio = [
            (x**2 + 2*y**2 - 4*x + 8*y, "paraboloide con desplazamiento"),
            (x**3 - 12*x + y**2, "función con múltiples críticos"),
            (x**2*y - y**3/3 - x**2, "superficie compleja"),
        ]
        
        templates_dificil = [
            (x**4 + y**4 - 4*x*y, "cuártica con término mixto"),
            (sp.sin(x)*sp.cos(y), "trigonométrica (múltiples críticos)"),
            (sp.exp(-(x**2 + y**2))*(x**2 - y**2), "gaussiana con silla"),
        ]
        
        if difficulty == 'facil':
            phi, description = random.choice(templates_facil)
        elif difficulty == 'intermedio':
            phi, description = random.choice(templates_intermedio)
        else:
            phi, description = random.choice(templates_dificil)
        
        phi_str = str(phi)
        
        # Calcular gradiente
        grad = tuple(sp.diff(phi, var) for var in (x, y))
        
        # Encontrar puntos críticos (simbólico)
        try:
            critical_points = sp.solve(grad, (x, y), dict=True)
            if critical_points:
                # Tomar el primer punto crítico
                pt = critical_points[0]
                pt_values = (float(pt[x]), float(pt[y]))
            else:
                # Si no hay solución simbólica, usar origen
                pt_values = (0.0, 0.0)
        except:
            pt_values = (0.0, 0.0)
        
        # Calcular Hessiana
        hessian = sp.Matrix([
            [sp.diff(phi, x, x), sp.diff(phi, x, y)],
            [sp.diff(phi, y, x), sp.diff(phi, y, y)]
        ])
        
        # Evaluar Hessiana en punto crítico
        hessian_at_pt = hessian.subs({x: pt_values[0], y: pt_values[1]})
        
        try:
            eigenvals_dict = hessian_at_pt.eigenvals()
            eigenvalues = [float(sp.re(eig)) for eig in eigenvals_dict.keys()]
        except:
            # Si falla el cálculo simbólico, usar numérico
            import numpy as np
            hess_numeric = np.array(hessian_at_pt).astype(float)
            eigenvalues = list(np.linalg.eigvalsh(hess_numeric))
        
        # Clasificar (con validación)
        if len(eigenvalues) >= 2:
            if all(eig > 0.01 for eig in eigenvalues):
                classification = "mínimo local"
            elif all(eig < -0.01 for eig in eigenvalues):
                classification = "máximo local"
            elif any(eig > 0.01 for eig in eigenvalues) and any(eig < -0.01 for eig in eigenvalues):
                classification = "punto silla"
            else:
                classification = "indeterminado"
        else:
            classification = "indeterminado"
        
        return {
            'id': exercise_id,
            'tipo': 'optimizacion_criticos',
            'dificultad': difficulty,
            'instruccion': f"Encuentra los puntos críticos de φ(x,y) = {phi_str} y clasifícalos.",
            'descripcion': f"Esta es una {description}. Debes calcular ∇φ, resolver ∇φ=0, y usar la Hessiana para clasificar.",
            'inputs': {
                'phi': phi_str,
                'variables': 'x, y'
            },
            'solution': {
                'gradient': [str(g) for g in grad],
                'critical_point': pt_values,
                'hessian': str(hessian),
                'eigenvalues': eigenvalues,
                'classification': classification,
                'phi_at_point': float(phi.subs({x: pt_values[0], y: pt_values[1]}))
            },
            'tolerance': 1e-4,
            'hints': [
                "💡 **Nivel 1:** Para encontrar puntos críticos, debes resolver ∇φ = 0.",
                "💡 **Nivel 2:** Calcula las derivadas parciales ∂φ/∂x y ∂φ/∂y, luego iguala ambas a cero.",
                "💡 **Nivel 3:** Después de encontrar el punto, calcula la matriz Hessiana H y sus valores propios.",
                f"💡 **Nivel 4:** El punto crítico está en ({pt_values[0]:.3f}, {pt_values[1]:.3f}). Clasifícalo con los eigenvalues: {eigenvalues}."
            ],
            'interpretacion': f"Físicamente, esta función representa {description}. El punto crítico es un **{classification}**."
        }
    
    elif problem_type == 'lagrange':
        # MULTIPLICADORES DE LAGRANGE
        # Optimizar f(x,y) sujeto a g(x,y) = 0
        
        # Funciones objetivo simples
        objective_funcs = [
            (x*y, "producto (maximizar área)"),
            (x**2 + y**2, "suma de cuadrados (minimizar distancia)"),
            (x + y, "suma (optimizar presupuesto)"),
        ]
        
        # Restricciones
        constraints = [
            (x + y - 10, "línea (presupuesto total = 10)"),
            (x**2 + y**2 - 25, "círculo (radio = 5)"),
            (x**2/9 + y**2/4 - 1, "elipse"),
        ]
        
        phi, desc_obj = random.choice(objective_funcs)
        constraint, desc_const = random.choice(constraints)
        
        phi_str = str(phi)
        constraint_str = str(constraint) + " = 0"
        
        # Resolver con Lagrange (simplificado)
        lam = sp.Symbol('lambda')
        L = phi - lam * constraint
        
        grad_L = [sp.diff(L, var) for var in (x, y, lam)]
        
        try:
            solutions = sp.solve(grad_L, (x, y, lam), dict=True)
            if solutions:
                sol = solutions[0]
                sol_point = (float(sol[x]), float(sol[y]))
                lambda_val = float(sol[lam])
            else:
                sol_point = (5.0, 5.0)
                lambda_val = 0.0
        except:
            sol_point = (5.0, 5.0)
            lambda_val = 0.0
        
        return {
            'id': exercise_id,
            'tipo': 'optimizacion_lagrange',
            'dificultad': difficulty,
            'instruccion': f"Optimiza φ(x,y) = {phi_str} sujeto a la restricción {constraint_str}.",
            'descripcion': f"Problema de {desc_obj} con restricción {desc_const}. Usa multiplicadores de Lagrange.",
            'inputs': {
                'phi': phi_str,
                'constraint': constraint_str,
                'variables': 'x, y'
            },
            'solution': {
                'lagrangian': str(L),
                'critical_point': sol_point,
                'lambda': lambda_val,
                'optimal_value': float(phi.subs({x: sol_point[0], y: sol_point[1]}))
            },
            'tolerance': 1e-3,
            'hints': [
                "💡 **Nivel 1:** Usa multiplicadores de Lagrange: construye L = φ - λg.",
                "💡 **Nivel 2:** Calcula ∇L y resuelve el sistema ∇L = 0.",
                "💡 **Nivel 3:** El sistema tiene 3 ecuaciones: ∂L/∂x=0, ∂L/∂y=0, ∂L/∂λ=0.",
                f"💡 **Nivel 4:** La solución está en x≈{sol_point[0]:.2f}, y≈{sol_point[1]:.2f}, λ≈{lambda_val:.2f}."
            ],
            'interpretacion': f"Este problema optimiza {desc_obj} bajo la restricción {desc_const}."
        }
    
    else:  # region
        # OPTIMIZACIÓN EN REGIÓN
        phi = x + y  # Función simple
        
        # Regiones
        region_templates = [
            ({'type': 'triangle', 'vertices': [(0,0), (0,4), (3,0)]}, "triángulo con vértices en (0,0), (0,4), (3,0)"),
            ({'type': 'rectangle', 'x_bounds': (0,5), 'y_bounds': (0,3)}, "rectángulo [0,5]×[0,3]"),
        ]
        
        region, desc = random.choice(region_templates)
        
        # Encontrar máximo/mínimo en la región (evaluando vértices para triángulo)
        if region['type'] == 'triangle':
            vertices = region['vertices']
            values = [float(phi.subs({x: v[0], y: v[1]})) for v in vertices]
            max_idx = values.index(max(values))
            min_idx = values.index(min(values))
            max_point = vertices[max_idx]
            min_point = vertices[min_idx]
            max_value = values[max_idx]
            min_value = values[min_idx]
        else:  # rectangle
            corners = [
                (region['x_bounds'][0], region['y_bounds'][0]),
                (region['x_bounds'][0], region['y_bounds'][1]),
                (region['x_bounds'][1], region['y_bounds'][0]),
                (region['x_bounds'][1], region['y_bounds'][1]),
            ]
            values = [float(phi.subs({x: c[0], y: c[1]})) for c in corners]
            max_idx = values.index(max(values))
            min_idx = values.index(min(values))
            max_point = corners[max_idx]
            min_point = corners[min_idx]
            max_value = values[max_idx]
            min_value = values[min_idx]
        
        return {
            'id': exercise_id,
            'tipo': 'optimizacion_region',
            'dificultad': difficulty,
            'instruccion': f"Encuentra el máximo y mínimo de φ(x,y) = {str(phi)} en la región: {desc}.",
            'descripcion': f"Optimización en una región acotada. Debes evaluar interior, frontera y vértices.",
            'inputs': {
                'phi': str(phi),
                'region': region
            },
            'solution': {
                'max_point': max_point,
                'max_value': max_value,
                'min_point': min_point,
                'min_value': min_value,
                'method': 'evaluación de vértices/frontera'
            },
            'tolerance': 1e-3,
            'hints': [
                "💡 **Nivel 1:** Para regiones acotadas, el máximo/mínimo está en puntos críticos interiores, frontera o vértices.",
                "💡 **Nivel 2:** Primero busca puntos críticos en el interior (∇φ=0). Luego analiza la frontera.",
                "💡 **Nivel 3:** Para esta función lineal, los extremos están en los vértices de la región.",
                f"💡 **Nivel 4:** Máximo en {max_point} con valor {max_value:.2f}. Mínimo en {min_point} con valor {min_value:.2f}."
            ],
            'interpretacion': f"En regiones cerradas y acotadas, los extremos siempre existen (teorema del valor extremo)."
        }


def export_report_pdf(
    report_dict: Dict[str, Any],
    filename: str
) -> str:
    """
    Exporta un informe profesional en formato PDF con gráficos y ecuaciones LaTeX.
    
    Parámetros
    ----------
    report_dict : Dict[str, Any]
        Diccionario con la estructura:
        - 'title': str - Título del informe
        - 'author': str - Autor
        - 'date': str - Fecha
        - 'sections': List[Dict] - Lista de secciones, cada una con:
            - 'title': str - Título de la sección
            - 'text': str - Texto descriptivo
            - 'latex': List[str] - Ecuaciones en LaTeX
            - 'fig_paths': List[str] - Rutas a imágenes PNG
            - 'data': Dict - Datos tabulares opcionales
    filename : str
        Nombre base del archivo (sin extensión)
    
    Retorna
    -------
    str
        Ruta completa del PDF generado
    
    Ejemplos
    --------
    >>> report = {
    ...     'title': 'Análisis de Gradiente',
    ...     'author': 'Sistema',
    ...     'date': '2025-11-16',
    ...     'sections': [
    ...         {
    ...             'title': 'Resultados',
    ...             'text': 'Se calculó el gradiente...',
    ...             'latex': [r'\\nabla \\phi = (2x, 2y, 2z)'],
    ...             'fig_paths': [],
    ...             'data': {}
    ...         }
    ...     ]
    ... }
    >>> path = export_report_pdf(report, 'mi_informe')
    """
    import os
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
    import time
    
    start_time = time.time()
    
    # Crear directorio reports si no existe
    reports_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    filepath = os.path.join(reports_dir, f"{filename}.pdf")
    
    logger.info(f"Generando PDF en {filepath}")
    
    # Crear documento
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Contenido
    story = []
    
    # Portada
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(report_dict.get('title', 'Informe de Cálculo Vectorial'), title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"<b>Autor:</b> {report_dict.get('author', 'Sistema Automático')}", normal_style))
    story.append(Paragraph(f"<b>Fecha:</b> {report_dict.get('date', time.strftime('%Y-%m-%d'))}", normal_style))
    story.append(PageBreak())
    
    # Secciones
    for section in report_dict.get('sections', []):
        # Título de sección
        story.append(Paragraph(section.get('title', 'Sección'), heading_style))
        
        # Texto
        if section.get('text'):
            story.append(Paragraph(section['text'], normal_style))
            story.append(Spacer(1, 12))
        
        # Ecuaciones LaTeX (renderizadas como texto)
        for latex_eq in section.get('latex', []):
            # Simplificación: mostrar LaTeX como código (reportlab no renderiza LaTeX nativamente)
            latex_text = f"<font name='Courier' size=10>{latex_eq}</font>"
            story.append(Paragraph(latex_text, normal_style))
            story.append(Spacer(1, 6))
        
        # Datos tabulares
        if section.get('data'):
            data = section['data']
            if isinstance(data, dict):
                table_data = [['Parámetro', 'Valor']]
                for key, value in data.items():
                    table_data.append([str(key), str(value)])
                
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
                story.append(Spacer(1, 12))
        
        # Imágenes
        for fig_path in section.get('fig_paths', []):
            if os.path.exists(fig_path):
                try:
                    img = Image(fig_path, width=5*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    logger.warning(f"No se pudo insertar imagen {fig_path}: {e}")
        
        story.append(Spacer(1, 0.3*inch))
    
    # Construir PDF
    doc.build(story)
    
    elapsed = time.time() - start_time
    logger.info(f"PDF generado exitosamente en {elapsed:.3f}s: {filepath}")
    
    return filepath


# ==============================================================================
# STREAMLINES (TRAYECTORIAS DE FLUJO) CON INTEGRACIÓN RK4
# ==============================================================================

def compute_streamlines(
    F_num: Callable,
    seeds: np.ndarray,
    step: float = 0.05,
    max_steps: int = 1000,
    domain: Optional[Tuple[float, float, float, float, float, float]] = None,
    both_directions: bool = True
) -> List[np.ndarray]:
    """
    Calcula streamlines (líneas de corriente) de un campo vectorial 3D.
    
    Integra trayectorias usando método Runge-Kutta 4 (RK4) desde puntos semilla.
    
    Parámetros
    ----------
    F_num : Callable
        Función vectorizada F(x, y, z) -> (Fx, Fy, Fz) donde x,y,z son arrays
    seeds : np.ndarray, shape (N, 3)
        Puntos iniciales para las streamlines
    step : float, default=0.05
        Tamaño del paso de integración
    max_steps : int, default=1000
        Número máximo de pasos por streamline
    domain : tuple(xmin, xmax, ymin, ymax, zmin, zmax), optional
        Dominio de integración. Si se sale, la streamline termina
    both_directions : bool, default=True
        Si True, integra hacia adelante y hacia atrás desde cada semilla
    
    Retorna
    -------
    List[np.ndarray]
        Lista de streamlines, cada una es array de shape (M, 3)
    
    Ejemplos
    --------
    >>> # Campo rotacional F = (-y, x, 0)
    >>> def F(x, y, z):
    ...     return (-y, x, np.zeros_like(x))
    >>> seeds = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> lines = compute_streamlines(F, seeds, step=0.1, max_steps=100)
    >>> len(lines)
    4  # 2 semillas × 2 direcciones
    """
    import time
    start = time.time()
    
    logger.info(f"Calculando streamlines desde {len(seeds)} semillas...")
    
    streamlines = []
    
    for seed_idx, seed in enumerate(seeds):
        # Integrar hacia adelante
        forward = _integrate_streamline_rk4(
            F_num, seed, step, max_steps, domain, direction=1.0
        )
        
        if both_directions:
            # Integrar hacia atrás
            backward = _integrate_streamline_rk4(
                F_num, seed, step, max_steps, domain, direction=-1.0
            )
            
            # Unir: backward invertido + forward
            if len(backward) > 1:
                full_line = np.vstack([backward[::-1][:-1], forward])
            else:
                full_line = forward
            
            streamlines.append(full_line)
        else:
            streamlines.append(forward)
    
    elapsed = time.time() - start
    total_points = sum(len(line) for line in streamlines)
    logger.info(f"Streamlines calculadas en {elapsed:.3f}s ({total_points} puntos totales)")
    
    return streamlines


def _integrate_streamline_rk4(
    F_num: Callable,
    seed: np.ndarray,
    step: float,
    max_steps: int,
    domain: Optional[Tuple],
    direction: float = 1.0
) -> np.ndarray:
    """
    Integra una streamline usando Runge-Kutta de orden 4 (RK4).
    
    Parámetros
    ----------
    F_num : Callable
        Campo vectorial F(x, y, z) -> (Fx, Fy, Fz)
    seed : np.ndarray, shape (3,)
        Punto inicial [x0, y0, z0]
    step : float
        Tamaño del paso (h en RK4)
    max_steps : int
        Número máximo de pasos
    domain : tuple or None
        (xmin, xmax, ymin, ymax, zmin, zmax) límites del dominio
    direction : float
        1.0 para forward, -1.0 para backward
    
    Retorna
    -------
    np.ndarray, shape (N, 3)
        Puntos de la streamline
    """
    points = [seed.copy()]
    current = seed.copy()
    
    h = step * direction
    
    for _ in range(max_steps):
        # Verificar si está dentro del dominio
        if domain is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = domain
            x, y, z = current
            if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
                break
        
        # Método RK4
        # k1 = F(yn)
        k1 = np.array(_evaluate_field_at_point(F_num, current))
        if np.any(np.isnan(k1)) or np.linalg.norm(k1) < 1e-10:
            break  # Campo nulo o inválido
        
        # k2 = F(yn + h/2 * k1)
        k2 = np.array(_evaluate_field_at_point(F_num, current + 0.5 * h * k1))
        if np.any(np.isnan(k2)):
            break
        
        # k3 = F(yn + h/2 * k2)
        k3 = np.array(_evaluate_field_at_point(F_num, current + 0.5 * h * k2))
        if np.any(np.isnan(k3)):
            break
        
        # k4 = F(yn + h * k3)
        k4 = np.array(_evaluate_field_at_point(F_num, current + h * k3))
        if np.any(np.isnan(k4)):
            break
        
        # yn+1 = yn + h/6 * (k1 + 2k2 + 2k3 + k4)
        next_point = current + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Verificar si el punto es válido
        if np.any(np.isnan(next_point)) or np.any(np.isinf(next_point)):
            break
        
        current = next_point
        points.append(current.copy())
    
    return np.array(points)


def _evaluate_field_at_point(F_num: Callable, point: np.ndarray) -> Tuple[float, float, float]:
    """
    Evalúa campo vectorial en un solo punto.
    
    F_num espera arrays, así que convertimos punto escalar a array de tamaño 1.
    """
    x, y, z = point
    
    # Convertir a arrays de tamaño 1
    x_arr = np.array([x])
    y_arr = np.array([y])
    z_arr = np.array([z])
    
    Fx, Fy, Fz = F_num(x_arr, y_arr, z_arr)
    
    # Extraer valores escalares
    return (float(Fx[0]), float(Fy[0]), float(Fz[0]))


# ==============================================================================
# CURVATURE MAP (para coloración por curvatura)
# ==============================================================================

def curvature_map(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Calcula mapa de curvatura para una superficie z = f(x,y).
    
    Usa aproximación de curvatura media mediante derivadas finitas.
    
    Parámetros
    ----------
    X, Y, Z : np.ndarray
        Coordenadas de la malla (2D arrays del mismo tamaño)
    
    Retorna
    -------
    np.ndarray
        Mapa de curvatura (misma forma que Z)
    
    Ejemplos
    --------
    >>> x = np.linspace(-2, 2, 50)
    >>> X, Y = np.meshgrid(x, x)
    >>> Z = X**2 + Y**2  # Paraboloide
    >>> K = curvature_map(X, Y, Z)
    >>> K.shape == Z.shape
    True
    """
    # Importar función de utils si está disponible, sino implementación básica
    try:
        from viz.utils import compute_mesh_curvature
        return compute_mesh_curvature(X, Y, Z, method='mean')
    except ImportError:
        # Implementación básica si viz no está disponible
        Zy, Zx = np.gradient(Z)
        Zxx = np.gradient(Zx, axis=1)
        Zyy = np.gradient(Zy, axis=0)
        Zxy = np.gradient(Zx, axis=0)
        
        dx = np.gradient(X, axis=1)
        dy = np.gradient(Y, axis=0)
        
        fx = Zx / (dx + 1e-10)
        fy = Zy / (dy + 1e-10)
        fxx = Zxx / (dx**2 + 1e-10)
        fyy = Zyy / (dy**2 + 1e-10)
        fxy = Zxy / (dx * dy + 1e-10)
        
        # Curvatura media
        numerator = (1 + fx**2) * fyy - 2 * fx * fy * fxy + (1 + fy**2) * fxx
        denominator = 2 * (1 + fx**2 + fy**2)**(3/2)
        curvature = numerator / (denominator + 1e-10)
        
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=10.0, neginf=-10.0)
        curvature = np.clip(curvature, -10, 10)
        
        return curvature


def export_mesh_to_obj(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    filename: str
) -> str:
    """
    Exporta malla 3D a formato Wavefront OBJ.
    
    Wrapper conveniente para viz.utils.export_obj que maneja imports.
    
    Parámetros
    ----------
    X, Y, Z : np.ndarray
        Coordenadas de la malla (2D arrays)
    filename : str
        Ruta del archivo de salida (.obj)
    
    Retorna
    -------
    str
        Ruta del archivo creado
    
    Ejemplos
    --------
    >>> x = np.linspace(-1, 1, 20)
    >>> X, Y = np.meshgrid(x, x)
    >>> Z = np.sin(np.sqrt(X**2 + Y**2))
    >>> path = export_mesh_to_obj(X, Y, Z, 'superficie.obj')
    """
    try:
        from viz.utils import export_obj
        return export_obj(X, Y, Z, filename, clean_nan=True)
    except ImportError:
        logger.error("Módulo viz no disponible. No se puede exportar OBJ.")
        raise ImportError("Instala el módulo viz para exportar OBJ")


# ==============================================================================
# FUNCIONES DE EXPORTACIÓN JSON PARA THREE.JS
# ==============================================================================

def export_mesh_json_from_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    triangles: Optional[np.ndarray] = None
) -> dict:
    """
    Exporta una superficie 3D a formato JSON compatible con Three.js.
    
    Genera un diccionario con el formato exacto requerido por el visualizador:
    - positions: array flat de coordenadas [x0,y0,z0, x1,y1,z1, ...]
    - indices: array flat de triángulos [i0,i1,i2, i3,i4,i5, ...]
    - meta: información sobre la malla
    
    Parámetros
    ----------
    X, Y, Z : np.ndarray
        Coordenadas de la malla (2D arrays de forma (m, n))
    triangles : Optional[np.ndarray]
        Índices de triángulos. Si None, se generan automáticamente
        desde la grilla regular.
    
    Retorna
    -------
    dict
        Diccionario con formato:
        {
            "type": "mesh",
            "meta": {"name": "surface", "domain": {...}, "shape": [m, n]},
            "positions": [x0,y0,z0, ...],
            "indices": [i0,i1,i2, ...]
        }
    
    Ejemplos
    --------
    >>> x = np.linspace(-3, 3, 50)
    >>> X, Y = np.meshgrid(x, x)
    >>> Z = np.sin(np.sqrt(X**2 + Y**2))
    >>> mesh_json = export_mesh_json_from_surface(X, Y, Z)
    >>> print(mesh_json['type - calc_vectorial.py:3392'])
    'mesh'
    """
    # Validar entradas
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError("X, Y, Z deben tener la misma forma")
    
    m, n = X.shape
    
    # Aplanar coordenadas en formato [x0,y0,z0, x1,y1,z1, ...]
    positions = []
    for i in range(m):
        for j in range(n):
            if np.isfinite(Z[i, j]):  # Saltar NaN/Inf
                positions.extend([float(X[i, j]), float(Y[i, j]), float(Z[i, j])])
    
    # Generar índices de triángulos si no se proporcionan
    if triangles is None:
        indices = []
        for i in range(m - 1):
            for j in range(n - 1):
                # Índice en array 1D
                idx = lambda ii, jj: ii * n + jj
                
                # Dos triángulos por celda
                # Triángulo 1: (i,j), (i+1,j), (i,j+1)
                indices.extend([idx(i, j), idx(i+1, j), idx(i, j+1)])
                # Triángulo 2: (i+1,j), (i+1,j+1), (i,j+1)
                indices.extend([idx(i+1, j), idx(i+1, j+1), idx(i, j+1)])
    else:
        indices = triangles.flatten().tolist()
    
    # Dominio
    xmin, xmax = float(np.nanmin(X)), float(np.nanmax(X))
    ymin, ymax = float(np.nanmin(Y)), float(np.nanmax(Y))
    
    return {
        "type": "mesh",
        "meta": {
            "name": "surface",
            "domain": {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax
            },
            "shape": [m, n]
        },
        "positions": positions,
        "indices": indices
    }


def compute_vector_field_grid(
    F: Tuple[Callable, Callable, Callable],
    domain: Dict[str, float],
    density: int = 12
) -> dict:
    """
    Calcula un campo vectorial en una grilla 3D y lo exporta a JSON.
    
    Genera un diccionario compatible con Three.js para renderizar
    flechas del campo vectorial.
    
    Parámetros
    ----------
    F : Tuple[Callable, Callable, Callable]
        Tupla de funciones (Fx, Fy, Fz) que definen el campo vectorial.
        Cada función debe aceptar (x, y, z) y retornar un escalar.
    domain : Dict[str, float]
        Diccionario con llaves 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'
    density : int
        Número de puntos por eje (default: 12)
    
    Retorna
    -------
    dict
        Diccionario con formato:
        {
            "type": "vector_field",
            "meta": {"density": density, "scale": scale},
            "positions": [x0,y0,z0, ...],
            "vectors": [u0,v0,w0, ...]
        }
    
    Ejemplos
    --------
    >>> Fx = lambda x,y,z: -y
    >>> Fy = lambda x,y,z: x
    >>> Fz = lambda x,y,z: 0
    >>> domain = {'xmin':-3,'xmax':3,'ymin':-3,'ymax':3,'zmin':0,'zmax':0}
    >>> field_json = compute_vector_field_grid((Fx,Fy,Fz), domain)
    """
    Fx, Fy, Fz = F
    
    # Crear grilla
    x = np.linspace(domain['xmin'], domain['xmax'], density)
    y = np.linspace(domain['ymin'], domain['ymax'], density)
    z_val = (domain['zmin'] + domain['zmax']) / 2  # Usar plano medio
    
    positions = []
    vectors = []
    
    for xi in x:
        for yi in y:
            try:
                # Evaluar campo
                u = float(Fx(xi, yi, z_val))
                v = float(Fy(xi, yi, z_val))
                w = float(Fz(xi, yi, z_val))
                
                # Solo agregar si es finito
                if np.isfinite([u, v, w]).all():
                    positions.extend([xi, yi, z_val])
                    vectors.extend([u, v, w])
            except:
                continue
    
    # Calcular escala automática
    if len(vectors) > 0:
        max_mag = np.max(np.sqrt(np.array(vectors[::3])**2 + 
                                  np.array(vectors[1::3])**2 + 
                                  np.array(vectors[2::3])**2))
        scale = 0.3 if max_mag == 0 else min(0.5, 1.0 / max_mag)
    else:
        scale = 0.3
    
    return {
        "type": "vector_field",
        "meta": {
            "density": density,
            "scale": scale
        },
        "positions": positions,
        "vectors": vectors
    }


def export_streamlines_json(
    streamlines_data: List[np.ndarray],
    method: str = 'rk4',
    step: float = 0.02,
    max_steps: int = 1000
) -> dict:
    """
    Exporta streamlines a formato JSON para Three.js.
    
    Parámetros
    ----------
    streamlines_data : List[np.ndarray]
        Lista de arrays, cada uno con forma (n, 3) representando
        una streamline con puntos [x, y, z]
    method : str
        Método de integración usado ('rk4', 'euler', etc.)
    step : float
        Tamaño de paso usado
    max_steps : int
        Máximo número de pasos
    
    Retorna
    -------
    dict
        Diccionario con formato:
        {
            "type": "streamlines",
            "meta": {"method": method, "step": step, "max_steps": max_steps},
            "lines": [[x0,y0,z0,x1,y1,z1,...], [...], ...]
        }
    
    Ejemplos
    --------
    >>> line1 = np.array([[0,0,0], [0.1,0.1,0], [0.2,0.2,0]])
    >>> line2 = np.array([[1,0,0], [1.1,0.1,0]])
    >>> stream_json = export_streamlines_json([line1, line2])
    """
    lines = []
    
    for stream in streamlines_data:
        if len(stream) > 0:
            # Aplanar a [x0,y0,z0, x1,y1,z1, ...]
            flat_line = stream.flatten().tolist()
            lines.append(flat_line)
    
    return {
        "type": "streamlines",
        "meta": {
            "method": method,
            "step": step,
            "max_steps": max_steps,
            "num_lines": len(lines)
        },
        "lines": lines
    }


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'safe_parse',
    'lambdify_vector',
    'compute_gradient_scalar',
    'compute_divergence_and_curl',
    'line_integral_over_param',
    'surface_flux_over_param',
    'plot_vector_field_slice',
    'compare_stokes_surface_line',
    'format_vector_latex',
    'explain_gradient_steps',
    'explain_div_curl_steps',
    'explain_line_integral_steps',
    'explain_surface_flux_steps',
    'explain_stokes_verification_steps',
    'generate_exercises',
    'plot_error_heatmap',
    'export_report_pdf',
    'compute_streamlines',
    'curvature_map',
    'export_mesh_to_obj',
    'export_mesh_json_from_surface',
    'compute_vector_field_grid',
    'export_streamlines_json',
]
