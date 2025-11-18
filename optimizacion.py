"""
Módulo de Optimización de Funciones Multivariables
===================================================

Este módulo proporciona herramientas completas para:
- Cálculo de gradientes y derivadas direccionales
- Clasificación de puntos críticos (mínimos, máximos, puntos silla)
- Optimización sin restricciones
- Optimización con restricciones (Multiplicadores de Lagrange)
- Optimización sobre regiones definidas
- Visualizaciones 3D/2D estilo GeoGebra

Autor: Equipo de Desarrollo
Fecha: Noviembre 2025
Versión: 1.0.0
"""

import sympy as sp
import numpy as np
from scipy.optimize import minimize, fsolve
from typing import Tuple, List, Dict, Optional, Callable, Any, Union
import logging
from functools import lru_cache
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILIDADES DE FORMATO
# ============================================================================

def format_number_prefer_exact(value: Union[float, sp.Expr], tolerance: float = 1e-10) -> str:
    """
    Formatea un número preferiendo representación exacta cuando sea posible.
    
    Parámetros
    ----------
    value : float o sp.Expr
        Valor a formatear
    tolerance : float
        Tolerancia para detectar fracciones y raíces
        
    Retorna
    -------
    str
        Representación LaTeX del número (exacta si es posible)
        
    Ejemplos
    --------
    >>> format_number_prefer_exact(0.707106781)
    '\\frac{\\sqrt{2}}{2}'
    >>> format_number_prefer_exact(1.414213562)
    '\\sqrt{2}'
    """
    if isinstance(value, sp.Expr):
        return sp.latex(sp.simplify(value))
    
    # Intentar detectar si es una fracción simple
    for den in range(1, 21):
        for num in range(-20 * den, 20 * den + 1):
            if abs(value - num/den) < tolerance:
                if den == 1:
                    return str(num)
                return f"\\frac{{{num}}}{{{den}}}"
    
    # Intentar detectar raíces cuadradas
    for n in range(2, 101):
        sqrt_n = np.sqrt(n)
        # Exactamente sqrt(n)
        if abs(value - sqrt_n) < tolerance:
            return f"\\sqrt{{{n}}}"
        # Exactamente -sqrt(n)
        if abs(value + sqrt_n) < tolerance:
            return f"-\\sqrt{{{n}}}"
        # Fracciones con sqrt
        for den in range(2, 11):
            if abs(value - sqrt_n/den) < tolerance:
                return f"\\frac{{\\sqrt{{{n}}}}}{{{den}}}"
            if abs(value + sqrt_n/den) < tolerance:
                return f"-\\frac{{\\sqrt{{{n}}}}}{{{den}}}"
    
    # Si no se detecta forma exacta, devolver decimal
    return f"{value:.8f}"


def format_point_exact(point: Tuple[float, ...], var_names: Tuple[str, ...] = None) -> str:
    """
    Formatea un punto con coordenadas exactas cuando sea posible.
    
    Parámetros
    ----------
    point : Tuple[float, ...]
        Coordenadas del punto
    var_names : Tuple[str, ...], opcional
        Nombres de variables (x, y, z, etc.)
        
    Retorna
    -------
    str
        Representación LaTeX del punto
    """
    if var_names is None:
        var_names = ('x', 'y', 'z', 'w')[:len(point)]
    
    coords = [format_number_prefer_exact(p) for p in point]
    assignments = [f"{var}={coord}" for var, coord in zip(var_names, coords)]
    return "(" + ", ".join(assignments) + ")"


# ============================================================================
# GRADIENTE Y DERIVADA DIRECCIONAL
# ============================================================================

@lru_cache(maxsize=128)
def compute_gradient(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...]
) -> Tuple[Tuple[sp.Expr, ...], Callable]:
    """
    Calcula el gradiente simbólico y función numpy vectorizada.
    
    El gradiente es el vector de derivadas parciales:
    ∇φ = (∂φ/∂x₁, ∂φ/∂x₂, ..., ∂φ/∂xₙ)
    
    Parámetros
    ----------
    phi : sp.Expr
        Función escalar a derivar
    vars : Tuple[sp.Symbol, ...]
        Variables (x, y, z, ...)
        
    Retorna
    -------
    gradient_sym : Tuple[sp.Expr, ...]
        Gradiente simbólico
    gradient_func : Callable
        Función vectorizada que acepta arrays/escalares
        
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> grad_sym, grad_func = compute_gradient(x**2 + y**2, (x, y))
    >>> grad_sym
    (2*x, 2*y)
    >>> grad_func(1.0, 2.0)
    array([2., 4.])
    """
    try:
        # Calcular derivadas parciales
        gradient_sym = tuple(sp.diff(phi, var) for var in vars)
        
        # Simplificar cada componente
        gradient_sym = tuple(sp.simplify(g) for g in gradient_sym)
        
        # Crear función numpy vectorizada
        gradient_func = sp.lambdify(vars, gradient_sym, modules=['numpy'])
        
        logger.info(f"Gradiente calculado: ∇φ = {gradient_sym}")
        
        return gradient_sym, gradient_func
        
    except Exception as e:
        logger.error(f"Error al calcular gradiente: {e}")
        raise ValueError(f"No se pudo calcular el gradiente: {e}")


def directional_derivative(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    point: Tuple[float, ...],
    direction: Tuple[float, ...]
) -> Dict[str, Any]:
    """
    Calcula la derivada direccional en un punto y dirección dados.
    
    La derivada direccional es: D_u φ(p) = ∇φ(p) · û
    donde û es el vector dirección normalizado.
    
    Parámetros
    ----------
    phi : sp.Expr
        Función escalar
    vars : Tuple[sp.Symbol, ...]
        Variables
    point : Tuple[float, ...]
        Punto donde evaluar
    direction : Tuple[float, ...]
        Vector dirección (se normalizará automáticamente)
        
    Retorna
    -------
    Dict con claves:
        - 'gradient_sym': Gradiente simbólico
        - 'gradient_at_point': Gradiente evaluado en el punto
        - 'direction_normalized': Vector dirección normalizado
        - 'directional_derivative': Valor de D_u φ(p)
        - 'is_maximum_direction': bool, True si dirección de máximo crecimiento
        - 'is_minimum_direction': bool, True si dirección de mínimo crecimiento
        - 'latex_steps': Lista de pasos en LaTeX
        - 'method': 'symbolic'
        
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> result = directional_derivative(x**2 + y**2, (x,y), (1,1), (1,0))
    >>> result['directional_derivative']
    2.0
    """
    try:
        latex_steps = []
        
        # Paso 1: Definir función
        latex_steps.append(
            f"\\text{{1. Función: }} \\phi = {sp.latex(phi)}"
        )
        
        # Paso 2: Calcular gradiente
        grad_sym, grad_func = compute_gradient(phi, vars)
        latex_steps.append(
            f"\\text{{2. Gradiente: }} \\nabla\\phi = \\left({', '.join(sp.latex(g) for g in grad_sym)}\\right)"
        )
        
        # Paso 3: Evaluar gradiente en el punto
        grad_at_point = np.array(grad_func(*point))
        point_str = format_point_exact(point, tuple(str(v) for v in vars))
        grad_coords = ", ".join(format_number_prefer_exact(g) for g in grad_at_point)
        latex_steps.append(
            f"\\text{{3. Evaluar en }} {point_str}: \\nabla\\phi = ({grad_coords})"
        )
        
        # Paso 4: Normalizar dirección
        direction_array = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction_array)
        
        if norm < 1e-10:
            raise ValueError("El vector dirección no puede ser cero")
        
        direction_normalized = direction_array / norm
        dir_coords = ", ".join(format_number_prefer_exact(d) for d in direction_normalized)
        latex_steps.append(
            f"\\text{{4. Normalizar dirección: }} \\hat{{u}} = \\frac{{1}}{{{format_number_prefer_exact(norm)}}}({', '.join(str(d) for d in direction)}) = ({dir_coords})"
        )
        
        # Paso 5: Producto punto
        dir_deriv = np.dot(grad_at_point, direction_normalized)
        latex_steps.append(
            f"\\text{{5. Derivada direccional: }} D_{{\\hat{{u}}}}\\phi = \\nabla\\phi \\cdot \\hat{{u}} = {format_number_prefer_exact(dir_deriv)}"
        )
        
        # Determinar si es dirección de máximo/mínimo
        grad_magnitude = np.linalg.norm(grad_at_point)
        is_maximum = abs(dir_deriv - grad_magnitude) < 1e-6 if grad_magnitude > 1e-10 else False
        is_minimum = abs(dir_deriv + grad_magnitude) < 1e-6 if grad_magnitude > 1e-10 else False
        
        if is_maximum:
            latex_steps.append(
                "\\text{✓ Esta es la dirección de MÁXIMO crecimiento}"
            )
        elif is_minimum:
            latex_steps.append(
                "\\text{✓ Esta es la dirección de MÍNIMO crecimiento (máximo decrecimiento)}"
            )
        else:
            latex_steps.append(
                f"\\text{{Magnitud del gradiente: }} ||\\nabla\\phi|| = {format_number_prefer_exact(grad_magnitude)}"
            )
        
        return {
            'gradient_sym': grad_sym,
            'gradient_at_point': grad_at_point.tolist(),
            'gradient_magnitude': float(grad_magnitude),
            'direction_normalized': direction_normalized.tolist(),
            'directional_derivative': float(dir_deriv),
            'is_maximum_direction': bool(is_maximum),
            'is_minimum_direction': bool(is_minimum),
            'latex_steps': latex_steps,
            'method': 'symbolic'
        }
        
    except Exception as e:
        logger.error(f"Error en derivada direccional: {e}")
        raise ValueError(f"No se pudo calcular derivada direccional: {e}")


# ============================================================================
# HESSIANA Y CLASIFICACIÓN DE PUNTOS CRÍTICOS
# ============================================================================

def hessian_and_eig(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    point: Optional[Tuple[float, ...]] = None
) -> Dict[str, Any]:
    """
    Calcula la matriz Hessiana y sus valores propios.
    
    La Hessiana es la matriz de segundas derivadas parciales:
    H_ij = ∂²φ/∂x_i∂x_j
    
    Parámetros
    ----------
    phi : sp.Expr
        Función escalar
    vars : Tuple[sp.Symbol, ...]
        Variables
    point : Tuple[float, ...], opcional
        Punto donde evaluar la Hessiana
        
    Retorna
    -------
    Dict con claves:
        - 'hessian_sym': Matriz Hessiana simbólica (sp.Matrix)
        - 'hessian_at_point': Hessiana evaluada (numpy array) si point dado
        - 'eigenvalues': Valores propios si point dado
        - 'determinant': Determinante si point dado
        - 'latex_steps': Pasos en LaTeX
        
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> result = hessian_and_eig(x**2 - y**2, (x, y), (0, 0))
    >>> result['eigenvalues']
    array([-2.,  2.])
    """
    try:
        latex_steps = []
        n = len(vars)
        
        # Calcular Hessiana simbólica
        hessian_sym = sp.Matrix([[sp.diff(phi, vi, vj) for vj in vars] for vi in vars])
        hessian_sym = sp.simplify(hessian_sym)
        
        latex_steps.append(
            f"\\text{{Matriz Hessiana:}} H = {sp.latex(hessian_sym)}"
        )
        
        result = {
            'hessian_sym': hessian_sym,
            'latex_steps': latex_steps,
            'method': 'symbolic'
        }
        
        # Si se proporciona punto, evaluar numéricamente
        if point is not None:
            # Convertir a función numpy
            hessian_func = sp.lambdify(vars, hessian_sym, modules=['numpy'])
            hessian_numeric = np.array(hessian_func(*point), dtype=float)
            
            # Calcular valores propios
            eigenvalues = np.linalg.eigvalsh(hessian_numeric)
            determinant = np.linalg.det(hessian_numeric)
            
            point_str = format_point_exact(point, tuple(str(v) for v in vars))
            latex_steps.append(
                f"\\text{{Evaluar en }} {point_str}:"
            )
            
            # Mostrar matriz numérica
            hess_latex = "\\begin{pmatrix}\n"
            for i in range(n):
                row = " & ".join(format_number_prefer_exact(hessian_numeric[i, j]) for j in range(n))
                hess_latex += row + " \\\\\n"
            hess_latex += "\\end{pmatrix}"
            latex_steps.append(f"H = {hess_latex}")
            
            # Valores propios
            eig_str = ", ".join(format_number_prefer_exact(eig) for eig in eigenvalues)
            latex_steps.append(
                f"\\text{{Valores propios: }} \\lambda = [{eig_str}]"
            )
            latex_steps.append(
                f"\\text{{Determinante: }} \\det(H) = {format_number_prefer_exact(determinant)}"
            )
            
            result.update({
                'hessian_at_point': hessian_numeric.tolist(),
                'eigenvalues': eigenvalues.tolist(),
                'determinant': float(determinant)
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error al calcular Hessiana: {e}")
        raise ValueError(f"No se pudo calcular la Hessiana: {e}")


def classify_critical_point(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    point: Tuple[float, ...]
) -> Dict[str, Any]:
    """
    Clasifica un punto crítico usando la Hessiana.
    
    Criterios de clasificación:
    - Todos λ > 0: Mínimo local
    - Todos λ < 0: Máximo local
    - λ con signos mixtos: Punto silla
    - Algún λ = 0: Indeterminado (requiere análisis adicional)
    
    Parámetros
    ----------
    phi : sp.Expr
        Función escalar
    vars : Tuple[sp.Symbol, ...]
        Variables
    point : Tuple[float, ...]
        Punto crítico a clasificar
        
    Retorna
    -------
    Dict con claves:
        - 'classification': str ('mínimo local', 'máximo local', 'punto silla', 'indeterminado')
        - 'eigenvalues': Valores propios
        - 'determinant': Determinante de la Hessiana
        - 'function_value': φ(point)
        - 'explanation': str con explicación detallada
        - 'latex_steps': Pasos en LaTeX
        
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> result = classify_critical_point(x**2 + y**2, (x, y), (0, 0))
    >>> result['classification']
    'mínimo local'
    """
    try:
        latex_steps = []
        point_str = format_point_exact(point, tuple(str(v) for v in vars))
        
        latex_steps.append(
            f"\\text{{Clasificar punto crítico: }} {point_str}"
        )
        
        # Calcular Hessiana y valores propios
        hess_result = hessian_and_eig(phi, vars, point)
        latex_steps.extend(hess_result['latex_steps'])
        
        eigenvalues = np.array(hess_result['eigenvalues'])
        determinant = hess_result['determinant']
        
        # Evaluar función en el punto
        phi_func = sp.lambdify(vars, phi, modules=['numpy'])
        function_value = float(phi_func(*point))
        
        latex_steps.append(
            f"\\phi{point_str} = {format_number_prefer_exact(function_value)}"
        )
        
        # Clasificar según valores propios
        tolerance = 1e-8
        
        if np.all(eigenvalues > tolerance):
            classification = 'mínimo local'
            explanation = f"Todos los valores propios son positivos → MÍNIMO LOCAL"
            latex_steps.append(f"\\text{{✓ }} {explanation}")
            
        elif np.all(eigenvalues < -tolerance):
            classification = 'máximo local'
            explanation = f"Todos los valores propios son negativos → MÁXIMO LOCAL"
            latex_steps.append(f"\\text{{✓ }} {explanation}")
            
        elif np.any(eigenvalues > tolerance) and np.any(eigenvalues < -tolerance):
            classification = 'punto silla'
            explanation = f"Valores propios con signos mixtos → PUNTO SILLA"
            latex_steps.append(f"\\text{{✓ }} {explanation}")
            
        else:
            classification = 'indeterminado'
            explanation = "Algún valor propio es cero → Se requiere análisis adicional (test de derivadas superiores o método direccional)"
            latex_steps.append(f"\\text{{⚠ }} {explanation}")
        
        return {
            'classification': classification,
            'eigenvalues': eigenvalues.tolist(),
            'determinant': float(determinant),
            'function_value': float(function_value),
            'point': point,
            'explanation': explanation,
            'latex_steps': latex_steps,
            'method': 'symbolic'
        }
        
    except Exception as e:
        logger.error(f"Error al clasificar punto crítico: {e}")
        raise ValueError(f"No se pudo clasificar el punto crítico: {e}")


# ============================================================================
# OPTIMIZACIÓN SIN RESTRICCIONES
# ============================================================================

def optimize_unconstrained(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...]
) -> Dict[str, Any]:
    """
    Encuentra y clasifica puntos críticos resolviendo ∇φ = 0.
    
    Procedimiento:
    1. Calcular ∇φ
    2. Resolver ∇φ = 0 simbólicamente
    3. Si falla, usar métodos numéricos con múltiples puntos iniciales
    4. Clasificar cada punto crítico encontrado
    
    Parámetros
    ----------
    phi : sp.Expr
        Función a optimizar
    vars : Tuple[sp.Symbol, ...]
        Variables
        
    Retorna
    -------
    Dict con claves:
        - 'critical_points': Lista de dict con cada punto crítico
        - 'global_minimum': dict con mínimo global (si existe)
        - 'global_maximum': dict con máximo global (si existe)
        - 'latex_steps': Pasos en LaTeX
        - 'method': 'symbolic' o 'numeric'
        
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> result = optimize_unconstrained(x**2 + y**2 - 2*x, (x, y))
    >>> len(result['critical_points'])
    1
    """
    try:
        latex_steps = []
        n = len(vars)
        
        latex_steps.append(
            f"\\text{{Optimizar sin restricciones: }} \\phi = {sp.latex(phi)}"
        )
        
        # Calcular gradiente
        grad_sym, _ = compute_gradient(phi, vars)
        latex_steps.append(
            f"\\nabla\\phi = \\left({', '.join(sp.latex(g) for g in grad_sym)}\\right)"
        )
        
        latex_steps.append(
            "\\text{Resolver } \\nabla\\phi = \\mathbf{0}:"
        )
        
        critical_points = []
        method = 'symbolic'
        
        # Intentar resolución simbólica
        try:
            solutions = sp.solve(grad_sym, vars, dict=True)
            
            if solutions:
                logger.info(f"Encontradas {len(solutions)} soluciones simbólicas")
                
                for sol in solutions:
                    # Verificar que todas las variables tengan valores
                    if all(v in sol for v in vars):
                        # Convertir a numérico
                        point_numeric = tuple(float(sol[v].evalf()) for v in vars)
                        
                        # Verificar que sea real
                        if all(np.isfinite(p) for p in point_numeric):
                            # Clasificar
                            classification = classify_critical_point(phi, vars, point_numeric)
                            critical_points.append(classification)
                            
                            point_str = format_point_exact(point_numeric, tuple(str(v) for v in vars))
                            latex_steps.append(
                                f"\\text{{Punto crítico: }} {point_str} \\quad (\\text{{{classification['classification']}}})"
                            )
        except Exception as e:
            logger.warning(f"Resolución simbólica falló: {e}")
        
        # Si no hay soluciones simbólicas, usar métodos numéricos
        if not critical_points:
            logger.info("Intentando resolución numérica con múltiples inicios")
            method = 'numeric'
            latex_steps.append(
                "\\text{Resolución simbólica no disponible. Usando métodos numéricos...}"
            )
            
            # Crear función del gradiente para fsolve
            grad_func = sp.lambdify(vars, grad_sym, modules=['numpy'])
            
            def grad_equations(x):
                return grad_func(*x)
            
            # Múltiples puntos iniciales
            initial_guesses = [
                tuple([0.0] * n),
                tuple([1.0] * n),
                tuple([-1.0] * n)
            ]
            # Añadir 5 puntos aleatorios
            for _ in range(5):
                initial_guesses.append(tuple(np.random.randn(n)))
            
            found_points = set()
            
            for guess in initial_guesses:
                try:
                    solution = fsolve(grad_equations, guess, full_output=True)
                    if solution[2] == 1:  # Convergió
                        point = tuple(solution[0])
                        
                        # Verificar que no sea duplicado
                        is_duplicate = False
                        for existing_point in found_points:
                            if np.allclose(point, existing_point, atol=1e-6):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate and all(np.isfinite(p) for p in point):
                            found_points.add(point)
                            classification = classify_critical_point(phi, vars, point)
                            critical_points.append(classification)
                            
                            point_str = format_point_exact(point, tuple(str(v) for v in vars))
                            latex_steps.append(
                                f"\\text{{Punto crítico: }} {point_str} \\quad (\\text{{{classification['classification']}}})"
                            )
                except Exception:
                    continue
        
        # Encontrar mínimo y máximo globales
        global_min = None
        global_max = None
        
        if critical_points:
            # Ordenar por valor de la función
            sorted_points = sorted(critical_points, key=lambda p: p['function_value'])
            
            # Mínimo global (candidato)
            min_candidates = [p for p in sorted_points if p['classification'] == 'mínimo local']
            if min_candidates:
                global_min = min_candidates[0]
            
            # Máximo global (candidato)
            max_candidates = [p for p in sorted_points if p['classification'] == 'máximo local']
            if max_candidates:
                global_max = max_candidates[-1]
            
            # Tabla comparativa
            latex_steps.append("\\text{Tabla comparativa:}")
            latex_steps.append("\\begin{array}{|c|c|c|}")
            latex_steps.append("\\hline")
            latex_steps.append("\\text{Punto} & \\phi(\\text{punto}) & \\text{Clasificación} \\\\")
            latex_steps.append("\\hline")
            
            for cp in sorted_points:
                point_str = format_point_exact(cp['point'], tuple(str(v) for v in vars))
                val_str = format_number_prefer_exact(cp['function_value'])
                latex_steps.append(f"{point_str} & {val_str} & \\text{{{cp['classification']}}} \\\\")
            
            latex_steps.append("\\hline")
            latex_steps.append("\\end{array}")
        else:
            latex_steps.append(
                "\\text{⚠ No se encontraron puntos críticos}"
            )
        
        return {
            'critical_points': critical_points,
            'global_minimum': global_min,
            'global_maximum': global_max,
            'latex_steps': latex_steps,
            'method': method
        }
        
    except Exception as e:
        logger.error(f"Error en optimización sin restricciones: {e}")
        raise ValueError(f"No se pudo optimizar: {e}")


# ============================================================================
# MULTIPLICADORES DE LAGRANGE
# ============================================================================

def solve_lagrange(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    constraints: List[sp.Expr]
) -> Dict[str, Any]:
    """
    Resuelve optimización con restricciones usando multiplicadores de Lagrange.
    
    Método:
    - Construir Lagrangiano: L = φ - Σ λ_i g_i
    - Resolver sistema: ∇L = 0 y g_i = 0
    - Clasificar puntos (si aplicable)
    
    Parámetros
    ----------
    phi : sp.Expr
        Función objetivo
    vars : Tuple[sp.Symbol, ...]
        Variables
    constraints : List[sp.Expr]
        Restricciones en forma g_i(x) = 0
        
    Retorna
    -------
    Dict con claves:
        - 'lagrangian': Expresión del Lagrangiano
        - 'solutions': Lista de dict con soluciones
        - 'optimal_value': Valor óptimo encontrado
        - 'latex_steps': Pasos en LaTeX
        - 'method': 'symbolic' o 'numeric'
        
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> # Maximizar f(x,y) = xy sujeto a x + y = 10
    >>> result = solve_lagrange(x*y, (x,y), [x + y - 10])
    """
    try:
        latex_steps = []
        n_constraints = len(constraints)
        
        latex_steps.append(
            f"\\text{{Optimizar con restricciones: }} \\phi = {sp.latex(phi)}"
        )
        
        # Mostrar restricciones
        for i, g in enumerate(constraints, 1):
            latex_steps.append(
                f"g_{i} : {sp.latex(g)} = 0"
            )
        
        # Crear variables lambda
        if n_constraints == 1:
            lambdas = (sp.Symbol('lambda1'),)
        else:
            lambdas = sp.symbols(f'lambda1:{n_constraints + 1}')
        
        # Construir Lagrangiano: L = φ - Σ λ_i g_i
        lagrangian = phi - sum(lam * g for lam, g in zip(lambdas, constraints))
        lagrangian = sp.simplify(lagrangian)
        
        latex_steps.append(
            f"\\text{{Lagrangiano: }} \\mathcal{{L}} = {sp.latex(lagrangian)}"
        )
        
        # Sistema de ecuaciones: ∇L = 0 y g_i = 0
        all_vars = vars + lambdas
        equations = []
        
        # Derivadas parciales del Lagrangiano
        for var in all_vars:
            eq = sp.diff(lagrangian, var)
            equations.append(eq)
        
        # Añadir restricciones
        equations.extend(constraints)
        
        latex_steps.append(
            "\\text{Sistema a resolver:}"
        )
        for i, eq in enumerate(equations[:len(vars)], 1):
            latex_steps.append(
                f"\\frac{{\\partial \\mathcal{{L}}}}{{\\partial {sp.latex(all_vars[i-1])}}} = {sp.latex(eq)} = 0"
            )
        
        solutions_list = []
        method = 'symbolic'
        
        # Intentar resolución simbólica
        try:
            solutions = sp.solve(equations, all_vars, dict=True)
            
            if solutions:
                logger.info(f"Encontradas {len(solutions)} soluciones con Lagrange")
                
                for sol in solutions:
                    # Extraer valores de variables y lambdas
                    if all(v in sol for v in vars):
                        point = tuple(float(sol[v].evalf()) for v in vars)
                        lambda_vals = tuple(float(sol[lam].evalf()) for lam in lambdas)
                        
                        # Verificar que sean valores reales
                        if all(np.isfinite(p) for p in point) and all(np.isfinite(lv) for lv in lambda_vals):
                            # Evaluar función objetivo
                            phi_func = sp.lambdify(vars, phi, modules=['numpy'])
                            phi_value = float(phi_func(*point))
                            
                            point_str = format_point_exact(point, tuple(str(v) for v in vars))
                            lambda_str = ", ".join(f"\\lambda_{i+1}={format_number_prefer_exact(lv)}" 
                                                   for i, lv in enumerate(lambda_vals))
                            
                            latex_steps.append(
                                f"\\text{{Solución: }} {point_str}, \\quad {lambda_str}"
                            )
                            latex_steps.append(
                                f"\\phi = {format_number_prefer_exact(phi_value)}"
                            )
                            
                            solutions_list.append({
                                'point': point,
                                'lambda_values': lambda_vals,
                                'function_value': phi_value
                            })
        
        except Exception as e:
            logger.warning(f"Resolución simbólica de Lagrange falló: {e}")
        
        # Si falla, intentar numérico
        if not solutions_list:
            method = 'numeric'
            latex_steps.append(
                "\\text{Usando métodos numéricos...}"
            )
            
            # Crear función para fsolve
            eqs_func = sp.lambdify(all_vars, equations, modules=['numpy'])
            
            def system(x):
                return eqs_func(*x)
            
            # Múltiples puntos iniciales
            n_total = len(all_vars)
            initial_guesses = [
                tuple([1.0] * n_total),
                tuple([0.5] * n_total)
            ]
            # Añadir 5 puntos aleatorios
            for _ in range(5):
                initial_guesses.append(tuple(np.random.randn(n_total)))
            
            for guess in initial_guesses:
                try:
                    solution = fsolve(system, guess, full_output=True)
                    if solution[2] == 1:
                        all_vals = solution[0]
                        point = tuple(all_vals[:len(vars)])
                        lambda_vals = tuple(all_vals[len(vars):])
                        
                        if all(np.isfinite(p) for p in point):
                            phi_func = sp.lambdify(vars, phi, modules=['numpy'])
                            phi_value = float(phi_func(*point))
                            
                            # Verificar si es duplicado
                            is_duplicate = any(
                                np.allclose(point, s['point'], atol=1e-6)
                                for s in solutions_list
                            )
                            
                            if not is_duplicate:
                                solutions_list.append({
                                    'point': point,
                                    'lambda_values': lambda_vals,
                                    'function_value': phi_value
                                })
                                
                                point_str = format_point_exact(point, tuple(str(v) for v in vars))
                                latex_steps.append(
                                    f"\\text{{Solución numérica: }} {point_str}, \\phi = {format_number_prefer_exact(phi_value)}"
                                )
                except Exception:
                    continue
        
        # Encontrar óptimo
        optimal_value = None
        if solutions_list:
            optimal_value = max(s['function_value'] for s in solutions_list)
            latex_steps.append(
                f"\\text{{Valor óptimo: }} {format_number_prefer_exact(optimal_value)}"
            )
        
        return {
            'lagrangian': lagrangian,
            'solutions': solutions_list,
            'optimal_value': optimal_value,
            'latex_steps': latex_steps,
            'method': method
        }
        
    except Exception as e:
        logger.error(f"Error en Lagrange: {e}")
        raise ValueError(f"No se pudo resolver con Lagrange: {e}")


# ============================================================================
# OPTIMIZACIÓN SOBRE REGIONES
# ============================================================================

def optimize_on_region(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    region: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Optimiza función sobre región definida (triángulo, rectángulo, elipse, etc.).
    
    Procedimiento:
    1. Buscar críticos interiores
    2. Optimizar sobre cada borde (parametrizar y resolver 1D)
    3. Evaluar en vértices
    4. Comparar todos y devolver máximo/mínimo global
    
    Parámetros
    ----------
    phi : sp.Expr
        Función a optimizar
    vars : Tuple[sp.Symbol, ...]
        Variables (usualmente x, y)
    region : Dict
        Especificación de la región:
        - {'type': 'triangle', 'vertices': [(x1,y1), (x2,y2), (x3,y3)]}
        - {'type': 'rectangle', 'bounds': [(xmin,xmax), (ymin,ymax)]}
        - {'type': 'ellipse', 'a': float, 'b': float, 'center': (h,k)}
        - {'type': 'implicit', 'constraint': sp.Expr <= 0}
        
    Retorna
    -------
    Dict con claves:
        - 'global_maximum': dict con punto y valor máximo
        - 'global_minimum': dict con punto y valor mínimo
        - 'interior_critical_points': lista de puntos críticos interiores
        - 'boundary_critical_points': lista de puntos críticos en bordes
        - 'vertex_values': valores en vértices
        - 'latex_steps': Pasos en LaTeX
        - 'comparison_table': Tabla con todos los candidatos
        
    Ejemplos
    --------
    >>> x, y = sp.symbols('x y')
    >>> region = {'type': 'triangle', 'vertices': [(0,0), (0,8), (4,0)]}
    >>> result = optimize_on_region(x + y, (x,y), region)
    """
    try:
        latex_steps = []
        region_type = region.get('type')
        
        latex_steps.append(
            f"\\text{{Optimizar en región: }} \\phi = {sp.latex(phi)}"
        )
        latex_steps.append(
            f"\\text{{Tipo de región: {region_type}}}"
        )
        
        all_candidates = []
        
        # ====================================================================
        # PASO 1: Críticos interiores
        # ====================================================================
        latex_steps.append("\\text{\\textbf{1. Puntos críticos interiores:}}")
        
        interior_result = optimize_unconstrained(phi, vars)
        interior_critical = interior_result['critical_points']
        
        # Filtrar por pertenencia a región
        interior_valid = []
        for cp in interior_critical:
            point = cp['point']
            if _point_in_region(point, region):
                interior_valid.append(cp)
                all_candidates.append({
                    'point': point,
                    'value': cp['function_value'],
                    'type': f"interior ({cp['classification']})"
                })
                
                point_str = format_point_exact(point, tuple(str(v) for v in vars))
                latex_steps.append(
                    f"✓ {point_str} \\quad \\phi = {format_number_prefer_exact(cp['function_value'])}"
                )
        
        if not interior_valid:
            latex_steps.append("\\text{No hay puntos críticos en el interior}")
        
        # ====================================================================
        # PASO 2: Optimizar sobre bordes
        # ====================================================================
        latex_steps.append("\\text{\\textbf{2. Optimización sobre bordes:}}")
        
        boundary_critical = []
        
        if region_type == 'triangle':
            vertices = region['vertices']
            n_vertices = len(vertices)
            
            # Cada borde es un segmento
            for i in range(n_vertices):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % n_vertices]
                
                latex_steps.append(f"\\text{{Borde {i+1}: de {v1} a {v2}}}")
                
                # Parametrizar: P(t) = v1 + t(v2 - v1), t ∈ [0,1]
                t = sp.Symbol('t', real=True)
                param_x = v1[0] + t * (v2[0] - v1[0])
                param_y = v1[1] + t * (v2[1] - v1[1])
                
                # Sustituir en phi
                phi_t = phi.subs([(vars[0], param_x), (vars[1], param_y)])
                phi_t = sp.simplify(phi_t)
                
                latex_steps.append(f"\\phi(t) = {sp.latex(phi_t)}")
                
                # Derivar respecto a t
                dphi_dt = sp.diff(phi_t, t)
                
                # Resolver dphi/dt = 0
                try:
                    t_critical = sp.solve(dphi_dt, t)
                    
                    for t_val in t_critical:
                        t_num = float(t_val.evalf())
                        
                        # Verificar que esté en [0, 1]
                        if 0 < t_num < 1:  # Excluir vértices
                            x_val = float(param_x.subs(t, t_num).evalf())
                            y_val = float(param_y.subs(t, t_num).evalf())
                            point = (x_val, y_val)
                            
                            phi_func = sp.lambdify(vars, phi, modules=['numpy'])
                            phi_val = float(phi_func(*point))
                            
                            boundary_critical.append({
                                'point': point,
                                'value': phi_val,
                                'edge': i + 1
                            })
                            all_candidates.append({
                                'point': point,
                                'value': phi_val,
                                'type': f'borde {i+1}'
                            })
                            
                            point_str = format_point_exact(point, tuple(str(v) for v in vars))
                            latex_steps.append(
                                f"✓ t={format_number_prefer_exact(t_num)}: {point_str}, \\phi = {format_number_prefer_exact(phi_val)}"
                            )
                except Exception:
                    pass
        
        elif region_type == 'rectangle':
            bounds = region['bounds']
            xmin, xmax = bounds[0]
            ymin, ymax = bounds[1]
            
            # 4 bordes
            edges = [
                ('y = ymin', vars[1], ymin, vars[0], xmin, xmax),
                ('y = ymax', vars[1], ymax, vars[0], xmin, xmax),
                ('x = xmin', vars[0], xmin, vars[1], ymin, ymax),
                ('x = xmax', vars[0], xmax, vars[1], ymin, ymax)
            ]
            
            for edge_name, fixed_var, fixed_val, free_var, free_min, free_max in edges:
                latex_steps.append(f"\\text{{Borde: {edge_name}}}")
                
                phi_edge = phi.subs(fixed_var, fixed_val)
                dphi = sp.diff(phi_edge, free_var)
                
                try:
                    critical_vals = sp.solve(dphi, free_var)
                    
                    for val in critical_vals:
                        val_num = float(val.evalf())
                        
                        if free_min < val_num < free_max:
                            if fixed_var == vars[1]:  # y fijo
                                point = (val_num, fixed_val)
                            else:  # x fijo
                                point = (fixed_val, val_num)
                            
                            phi_func = sp.lambdify(vars, phi, modules=['numpy'])
                            phi_val = float(phi_func(*point))
                            
                            boundary_critical.append({
                                'point': point,
                                'value': phi_val,
                                'edge': edge_name
                            })
                            all_candidates.append({
                                'point': point,
                                'value': phi_val,
                                'type': f'borde ({edge_name})'
                            })
                except Exception:
                    pass
        
        # ====================================================================
        # PASO 3: Evaluar en vértices
        # ====================================================================
        latex_steps.append("\\text{\\textbf{3. Evaluación en vértices:}}")
        
        vertex_values = []
        vertices_to_check = []
        
        if region_type == 'triangle':
            vertices_to_check = region['vertices']
        elif region_type == 'rectangle':
            bounds = region['bounds']
            vertices_to_check = [
                (bounds[0][0], bounds[1][0]),
                (bounds[0][0], bounds[1][1]),
                (bounds[0][1], bounds[1][0]),
                (bounds[0][1], bounds[1][1])
            ]
        
        phi_func = sp.lambdify(vars, phi, modules=['numpy'])
        
        for vertex in vertices_to_check:
            phi_val = float(phi_func(*vertex))
            vertex_values.append({
                'point': vertex,
                'value': phi_val
            })
            all_candidates.append({
                'point': vertex,
                'value': phi_val,
                'type': 'vértice'
            })
            
            vertex_str = format_point_exact(vertex, tuple(str(v) for v in vars))
            latex_steps.append(
                f"{vertex_str}: \\phi = {format_number_prefer_exact(phi_val)}"
            )
        
        # ====================================================================
        # PASO 4: Comparar y encontrar extremos globales
        # ====================================================================
        latex_steps.append("\\text{\\textbf{4. Tabla comparativa:}}")
        
        # Ordenar candidatos
        all_candidates_sorted = sorted(all_candidates, key=lambda c: c['value'])
        
        # Tabla LaTeX
        latex_steps.append("\\begin{array}{|c|c|c|}")
        latex_steps.append("\\hline")
        latex_steps.append("\\text{Punto} & \\phi & \\text{Tipo} \\\\")
        latex_steps.append("\\hline")
        
        for cand in all_candidates_sorted:
            point_str = format_point_exact(cand['point'], tuple(str(v) for v in vars))
            val_str = format_number_prefer_exact(cand['value'])
            latex_steps.append(f"{point_str} & {val_str} & \\text{{{cand['type']}}} \\\\")
        
        latex_steps.append("\\hline")
        latex_steps.append("\\end{array}")
        
        # Mínimo y máximo globales
        global_min = all_candidates_sorted[0]
        global_max = all_candidates_sorted[-1]
        
        latex_steps.append("\\text{\\textbf{Conclusión:}}")
        min_str = format_point_exact(global_min['point'], tuple(str(v) for v in vars))
        max_str = format_point_exact(global_max['point'], tuple(str(v) for v in vars))
        latex_steps.append(
            f"\\text{{Mínimo global: }} {min_str}, \\phi = {format_number_prefer_exact(global_min['value'])}"
        )
        latex_steps.append(
            f"\\text{{Máximo global: }} {max_str}, \\phi = {format_number_prefer_exact(global_max['value'])}"
        )
        
        return {
            'global_maximum': global_max,
            'global_minimum': global_min,
            'interior_critical_points': interior_valid,
            'boundary_critical_points': boundary_critical,
            'vertex_values': vertex_values,
            'latex_steps': latex_steps,
            'comparison_table': all_candidates_sorted,
            'method': 'complete_analysis'
        }
        
    except Exception as e:
        logger.error(f"Error en optimización sobre región: {e}")
        raise ValueError(f"No se pudo optimizar sobre la región: {e}")


def _point_in_region(point: Tuple[float, float], region: Dict[str, Any]) -> bool:
    """Helper: verifica si punto está dentro de región."""
    region_type = region.get('type')
    x, y = point
    
    if region_type == 'triangle':
        vertices = region['vertices']
        # Usar coordenadas baricéntricas
        v0, v1, v2 = vertices
        
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(point, v0, v1)
        d2 = sign(point, v1, v2)
        d3 = sign(point, v2, v0)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    elif region_type == 'rectangle':
        bounds = region['bounds']
        return (bounds[0][0] <= x <= bounds[0][1] and 
                bounds[1][0] <= y <= bounds[1][1])
    
    elif region_type == 'ellipse':
        a = region['a']
        b = region['b']
        center = region.get('center', (0, 0))
        h, k = center
        return ((x - h)**2 / a**2 + (y - k)**2 / b**2) <= 1
    
    return True  # Por defecto asumir que está dentro


# ============================================================================
# VISUALIZACIONES 3D/2D ESTILO GEOGEBRA
# ============================================================================

def visualize_optimization_3d(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    critical_points: List[Dict] = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
    resolution: int = 50,
    show_gradient: bool = True,
    show_contours: bool = True,
    gradient_density: int = 10
) -> 'plotly.graph_objs.Figure':
    """
    Crea visualización 3D estilo GeoGebra con superficie, gradiente y puntos críticos.
    
    Características:
    - Ejes con ticks numerados
    - Superficie z = φ(x,y) con colormap
    - Contornos proyectados en plano XY
    - Campo de gradiente con flechas
    - Puntos críticos marcados con colores según clasificación
    - Leyendas y tooltips informativos
    
    Parámetros
    ----------
    phi : sp.Expr
        Función a visualizar
    vars : Tuple[sp.Symbol, ...]
        Variables (x, y)
    critical_points : List[Dict], opcional
        Lista de puntos críticos a marcar
    bounds : Tuple, opcional
        ((xmin, xmax), (ymin, ymax))
    resolution : int
        Densidad de la malla
    show_gradient : bool
        Mostrar campo de gradiente
    show_contours : bool
        Mostrar curvas de nivel
    gradient_density : int
        Densidad del campo de gradiente
        
    Retorna
    -------
    fig : plotly.graph_objs.Figure
        Figura interactiva de Plotly
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Determinar límites si no se proporcionan
        if bounds is None:
            bounds = ((-5, 5), (-5, 5))
        
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        
        # Crear malla
        x_vals = np.linspace(xmin, xmax, resolution)
        y_vals = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Evaluar función
        phi_func = sp.lambdify(vars, phi, modules=['numpy'])
        
        try:
            Z = phi_func(X, Y)
            if np.isscalar(Z):
                Z = np.full_like(X, Z)
        except:
            Z = np.array([[phi_func(x, y) for x in x_vals] for y in y_vals])
        
        # Crear figura
        fig = go.Figure()
        
        # ====== Superficie 3D ======
        surface = go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',
            name='φ(x,y)',
            showscale=True,
            colorbar=dict(
                title='φ',
                tickmode='linear',
                tick0=np.min(Z),
                dtick=(np.max(Z) - np.min(Z)) / 10
            ),
            hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>φ: %{z:.3f}<extra></extra>'
        )
        fig.add_trace(surface)
        
        # ====== Contornos en plano XY ======
        if show_contours:
            contour = go.Contour(
                x=x_vals,
                y=y_vals,
                z=Z,
                colorscale='Viridis',
                showscale=False,
                contours=dict(
                    coloring='lines',
                    showlabels=True,
                    labelfont=dict(size=10, color='white')
                ),
                name='Curvas de nivel',
                hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            )
            # Proyectar en z = zmin
            # (En 3D real necesitaríamos scatter3d, aquí mostramos aparte)
        
        # ====== Campo de gradiente ======
        if show_gradient:
            grad_sym, grad_func = compute_gradient(phi, vars)
            
            # Malla más gruesa para gradiente
            x_grad = np.linspace(xmin, xmax, gradient_density)
            y_grad = np.linspace(ymin, ymax, gradient_density)
            X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
            
            # Evaluar gradiente
            grad_vals = grad_func(X_grad.flatten(), Y_grad.flatten())
            if isinstance(grad_vals[0], (int, float)):
                U = np.full_like(X_grad.flatten(), grad_vals[0])
                V = np.full_like(X_grad.flatten(), grad_vals[1])
            else:
                U = grad_vals[0]
                V = grad_vals[1]
            
            # Evaluar Z en puntos de gradiente
            Z_grad = phi_func(X_grad.flatten(), Y_grad.flatten())
            if np.isscalar(Z_grad):
                Z_grad = np.full(X_grad.size, Z_grad)
            
            # Magnitud para colorear
            magnitude = np.sqrt(U**2 + V**2)
            
            # Crear conos para vectores
            # Nota: go.Cone no usa 'marker', usa propiedades directamente en la raíz
            cone = go.Cone(
                x=X_grad.flatten(),
                y=Y_grad.flatten(),
                z=Z_grad,
                u=U,
                v=V,
                w=np.zeros_like(U),  # Gradiente en plano tangente
                colorscale='Reds',
                sizemode='absolute',
                sizeref=0.3,
                showscale=True,
                colorbar=dict(
                    title='||∇φ||',
                    x=1.1
                ),
                name='∇φ',
                hovertemplate='∇φ: (%{u:.3f}, %{v:.3f})<br>Magnitud: %{text:.3f}<extra></extra>',
                text=magnitude  # Mostrar magnitud en hover
            )
            fig.add_trace(cone)
        
        # ====== Marcar puntos críticos ======
        if critical_points:
            colors = {
                'mínimo local': 'blue',
                'máximo local': 'red',
                'punto silla': 'yellow',
                'indeterminado': 'gray'
            }
            
            for cp in critical_points:
                point = cp['point']
                classification = cp['classification']
                value = cp['function_value']
                
                scatter = go.Scatter3d(
                    x=[point[0]],
                    y=[point[1]],
                    z=[value],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors.get(classification, 'white'),
                        symbol='diamond',
                        line=dict(color='black', width=2)
                    ),
                    name=classification,
                    hovertemplate=f'<b>{classification}</b><br>x: {point[0]:.4f}<br>y: {point[1]:.4f}<br>φ: {value:.4f}<extra></extra>',
                    showlegend=True
                )
                fig.add_trace(scatter)
        
        # ====== Configurar ejes estilo GeoGebra ======
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='x',
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    showbackground=True,
                    tickmode='linear',
                    tick0=xmin,
                    dtick=(xmax - xmin) / 10,
                    showticklabels=True
                ),
                yaxis=dict(
                    title='y',
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    showbackground=True,
                    tickmode='linear',
                    tick0=ymin,
                    dtick=(ymax - ymin) / 10,
                    showticklabels=True
                ),
                zaxis=dict(
                    title='φ',
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    showbackground=True,
                    tickmode='auto',
                    nticks=10,
                    showticklabels=True
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            title=f'Optimización: φ = {sp.latex(phi)}',
            showlegend=True,
            height=700,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error en visualización 3D: {e}")
        raise ValueError(f"No se pudo crear visualización: {e}")


def visualize_contour_2d(
    phi: sp.Expr,
    vars: Tuple[sp.Symbol, ...],
    critical_points: List[Dict] = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
    resolution: int = 100,
    show_gradient: bool = True,
    gradient_density: int = 15,
    region: Dict[str, Any] = None
) -> 'plotly.graph_objs.Figure':
    """
    Crea visualización 2D con contornos, gradiente y región factible.
    
    Parámetros
    ----------
    phi : sp.Expr
        Función a visualizar
    vars : Tuple[sp.Symbol, ...]
        Variables (x, y)
    critical_points : List[Dict], opcional
        Puntos críticos a marcar
    bounds : Tuple, opcional
        Límites de visualización
    resolution : int
        Densidad de contornos
    show_gradient : bool
        Mostrar campo vectorial del gradiente
    gradient_density : int
        Densidad de flechas del gradiente
    region : Dict, opcional
        Región factible a resaltar
        
    Retorna
    -------
    fig : plotly.graph_objs.Figure
        Figura de Plotly
    """
    try:
        import plotly.graph_objects as go
        
        if bounds is None:
            bounds = ((-5, 5), (-5, 5))
        
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        
        # Malla
        x_vals = np.linspace(xmin, xmax, resolution)
        y_vals = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Evaluar función
        phi_func = sp.lambdify(vars, phi, modules=['numpy'])
        try:
            Z = phi_func(X, Y)
            if np.isscalar(Z):
                Z = np.full_like(X, Z)
        except:
            Z = np.array([[phi_func(x, y) for x in x_vals] for y in y_vals])
        
        fig = go.Figure()
        
        # Contornos
        contour = go.Contour(
            x=x_vals,
            y=y_vals,
            z=Z,
            colorscale='Viridis',
            showscale=True,
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title='φ'),
            hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>φ: %{z:.3f}<extra></extra>'
        )
        fig.add_trace(contour)
        
        # Campo de gradiente
        if show_gradient:
            grad_sym, grad_func = compute_gradient(phi, vars)
            
            x_grad = np.linspace(xmin, xmax, gradient_density)
            y_grad = np.linspace(ymin, ymax, gradient_density)
            X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
            
            grad_vals = grad_func(X_grad, Y_grad)
            if isinstance(grad_vals[0], (int, float)):
                U = np.full_like(X_grad, grad_vals[0])
                V = np.full_like(X_grad, grad_vals[1])
            else:
                U = grad_vals[0]
                V = grad_vals[1]
            
            # Normalizar para mejor visualización
            magnitude = np.sqrt(U**2 + V**2)
            U_norm = np.where(magnitude > 0, U / magnitude, 0)
            V_norm = np.where(magnitude > 0, V / magnitude, 0)
            
            # Escalar
            scale = (xmax - xmin) / gradient_density * 0.5
            
            # Crear flechas manualmente
            for i in range(gradient_density):
                for j in range(gradient_density):
                    x0 = X_grad[i, j]
                    y0 = Y_grad[i, j]
                    dx = U_norm[i, j] * scale
                    dy = V_norm[i, j] * scale
                    
                    if magnitude[i, j] > 1e-6:
                        # Flecha (línea + triángulo)
                        fig.add_annotation(
                            x=x0 + dx,
                            y=y0 + dy,
                            ax=x0,
                            ay=y0,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='white',
                            opacity=0.7
                        )
        
        # Dibujar región si se proporciona
        if region:
            region_type = region.get('type')
            
            if region_type == 'triangle':
                vertices = region['vertices']
                vertices_closed = vertices + [vertices[0]]
                xs = [v[0] for v in vertices_closed]
                ys = [v[1] for v in vertices_closed]
                
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Región',
                    showlegend=True
                ))
            
            elif region_type == 'rectangle':
                b = region['bounds']
                xs = [b[0][0], b[0][1], b[0][1], b[0][0], b[0][0]]
                ys = [b[1][0], b[1][0], b[1][1], b[1][1], b[1][0]]
                
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Región',
                    showlegend=True
                ))
            
            elif region_type == 'ellipse':
                a = region['a']
                b = region['b']
                center = region.get('center', (0, 0))
                theta = np.linspace(0, 2*np.pi, 100)
                xs = center[0] + a * np.cos(theta)
                ys = center[1] + b * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Región',
                    showlegend=True,
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)'
                ))
        
        # Marcar puntos críticos
        if critical_points:
            colors = {
                'mínimo local': 'blue',
                'máximo local': 'red',
                'punto silla': 'yellow',
                'indeterminado': 'gray',
                'vértice': 'purple',
                'borde': 'orange'
            }
            
            for cp in critical_points:
                point = cp['point']
                classification = cp.get('classification', cp.get('type', 'punto'))
                value = cp.get('function_value', cp.get('value'))
                
                # Determinar color
                color = colors.get(classification, 'white')
                for key in colors:
                    if key in classification:
                        color = colors[key]
                        break
                
                fig.add_trace(go.Scatter(
                    x=[point[0]],
                    y=[point[1]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol='diamond',
                        line=dict(color='black', width=2)
                    ),
                    name=classification,
                    hovertemplate=f'<b>{classification}</b><br>x: {point[0]:.4f}<br>y: {point[1]:.4f}<br>φ: {value:.4f}<extra></extra>',
                    showlegend=True
                ))
        
        # Configurar layout
        fig.update_layout(
            title=f'Curvas de nivel: φ = {sp.latex(phi)}',
            xaxis=dict(
                title='x',
                scaleanchor='y',
                scaleratio=1,
                gridcolor='lightgray',
                tickmode='linear',
                tick0=xmin,
                dtick=(xmax - xmin) / 10
            ),
            yaxis=dict(
                title='y',
                gridcolor='lightgray',
                tickmode='linear',
                tick0=ymin,
                dtick=(ymax - ymin) / 10
            ),
            plot_bgcolor='white',
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error en visualización 2D: {e}")
        raise ValueError(f"No se pudo crear visualización 2D: {e}")


# ============================================================================
# CASOS ESPECIALES PRE-CONFIGURADOS
# ============================================================================

def max_rectangle_in_ellipse(a: float, b: float) -> Dict[str, Any]:
    """
    Resuelve el problema clásico: máximo rectángulo inscrito en elipse.
    
    Elipse: x²/a² + y²/b² = 1
    Rectángulo con vértice en (x, y) tiene área A = 4xy
    
    Solución analítica: x = a/√2, y = b/√2, A_max = 2ab
    
    Parámetros
    ----------
    a : float
        Semi-eje mayor
    b : float
        Semi-eje menor
        
    Retorna
    -------
    Dict con solución exacta y pasos LaTeX
    """
    latex_steps = []
    
    latex_steps.append(
        f"\\text{{Maximizar área de rectángulo inscrito en elipse: }} \\frac{{x^2}}{{{a}^2}} + \\frac{{y^2}}{{{b}^2}} = 1"
    )
    latex_steps.append(
        "\\text{Área del rectángulo: } A = 4xy"
    )
    
    # Solución analítica
    x_opt = a / np.sqrt(2)
    y_opt = b / np.sqrt(2)
    area_max = 2 * a * b
    
    latex_steps.append(
        f"\\text{{Solución: }} x = \\frac{{{a}}}{{\\sqrt{{2}}}}, \\quad y = \\frac{{{b}}}{{\\sqrt{{2}}}}"
    )
    latex_steps.append(
        f"\\text{{Área máxima: }} A_{{\\max}} = 2ab = {area_max}"
    )
    
    return {
        'optimal_point': (x_opt, y_opt),
        'maximum_area': area_max,
        'latex_steps': latex_steps,
        'method': 'analytical'
    }


def cobb_douglas_optimization(alpha: float, beta: float, px: float, py: float, M: float) -> Dict[str, Any]:
    """
    Resuelve optimización de Cobb-Douglas con restricción presupuestaria.
    
    Maximizar: f(x,y) = x^α · y^β
    Sujeto a: px·x + py·y = M
    
    Solución analítica: x* = αM/px, y* = βM/py (si α+β=1)
    
    Parámetros
    ----------
    alpha, beta : float
        Exponentes de Cobb-Douglas
    px, py : float
        Precios
    M : float
        Presupuesto
        
    Retorna
    -------
    Dict con solución óptima
    """
    latex_steps = []
    
    latex_steps.append(
        f"\\text{{Maximizar: }} f(x,y) = x^{{{alpha}}} y^{{{beta}}}"
    )
    latex_steps.append(
        f"\\text{{Sujeto a: }} {px}x + {py}y = {M}"
    )
    
    # Solución con Lagrange
    x, y = sp.symbols('x y', positive=True, real=True)
    phi = x**alpha * y**beta
    constraint = px*x + py*y - M
    
    result = solve_lagrange(phi, (x, y), [constraint])
    
    latex_steps.extend(result['latex_steps'])
    
    # Solución analítica conocida (si α+β=1)
    if abs(alpha + beta - 1.0) < 1e-6:
        x_opt = alpha * M / px
        y_opt = beta * M / py
        f_opt = (x_opt**alpha) * (y_opt**beta)
        
        latex_steps.append(
            f"\\text{{Solución analítica (α+β=1): }} x^* = \\frac{{\\alpha M}}{{p_x}} = {x_opt:.4f}, \\quad y^* = \\frac{{\\beta M}}{{p_y}} = {y_opt:.4f}"
        )
        latex_steps.append(
            f"f(x^*, y^*) = {f_opt:.4f}"
        )
    
    return {
        'result': result,
        'latex_steps': latex_steps,
        'method': 'lagrange'
    }


# ============================================================================
# EXPORTACIÓN
# ============================================================================

__all__ = [
    'compute_gradient',
    'directional_derivative',
    'hessian_and_eig',
    'classify_critical_point',
    'optimize_unconstrained',
    'solve_lagrange',
    'optimize_on_region',
    'visualize_optimization_3d',
    'visualize_contour_2d',
    'max_rectangle_in_ellipse',
    'cobb_douglas_optimization',
    'format_number_prefer_exact',
    'format_point_exact'
]
