"""
Tests para el módulo de optimización
=====================================

Autor: Equipo de Desarrollo
Fecha: Noviembre 2025
"""

import pytest
import sympy as sp
import numpy as np
from optimizacion import (
    compute_gradient,
    directional_derivative,
    hessian_and_eig,
    classify_critical_point,
    optimize_unconstrained,
    solve_lagrange,
    optimize_on_region,
    max_rectangle_in_ellipse,
    cobb_douglas_optimization,
    format_number_prefer_exact
)


# ============================================================================
# TESTS DE GRADIENTE
# ============================================================================

def test_compute_gradient_simple():
    """Test gradiente de x*y"""
    x, y = sp.symbols('x y')
    phi = x * y
    
    grad_sym, grad_func = compute_gradient(phi, (x, y))
    
    # Verificar simbólico
    assert grad_sym == (y, x)
    
    # Verificar numérico
    result = grad_func(2.0, 3.0)
    assert np.allclose(result, [3.0, 2.0])


def test_compute_gradient_quadratic():
    """Test gradiente de x²+y²+z²"""
    x, y, z = sp.symbols('x y z')
    phi = x**2 + y**2 + z**2
    
    grad_sym, grad_func = compute_gradient(phi, (x, y, z))
    
    # Verificar simbólico
    assert grad_sym == (2*x, 2*y, 2*z)
    
    # Verificar en (1,1,1)
    result = grad_func(1.0, 1.0, 1.0)
    assert np.allclose(result, [2.0, 2.0, 2.0])


def test_directional_derivative():
    """Test derivada direccional"""
    x, y = sp.symbols('x y')
    phi = x**2 + y**2
    
    # En (1,1) hacia (1,0) debería ser 2
    result = directional_derivative(phi, (x, y), (1.0, 1.0), (1.0, 0.0))
    
    assert 'directional_derivative' in result
    assert np.isclose(result['directional_derivative'], 2.0)
    assert 'latex_steps' in result


def test_directional_derivative_maximum_direction():
    """Test que detecte dirección de máximo crecimiento"""
    x, y = sp.symbols('x y')
    phi = x**2 + y**2
    
    # ∇φ(1,1) = (2,2), magnitud = 2√2
    # Dirección (1,1) normalizada debería ser máximo
    result = directional_derivative(phi, (x, y), (1.0, 1.0), (1.0, 1.0))
    
    assert result['is_maximum_direction'] == True


# ============================================================================
# TESTS DE CLASIFICACIÓN
# ============================================================================

def test_classify_minimum():
    """Test clasificación de mínimo"""
    x, y = sp.symbols('x y')
    phi = x**2 + y**2
    
    result = classify_critical_point(phi, (x, y), (0.0, 0.0))
    
    assert result['classification'] == 'mínimo local'
    assert result['function_value'] == 0.0
    assert all(eig > 0 for eig in result['eigenvalues'])


def test_classify_maximum():
    """Test clasificación de máximo"""
    x, y = sp.symbols('x y')
    phi = -x**2 - y**2
    
    result = classify_critical_point(phi, (x, y), (0.0, 0.0))
    
    assert result['classification'] == 'máximo local'
    assert all(eig < 0 for eig in result['eigenvalues'])


def test_classify_saddle():
    """Test clasificación de punto silla"""
    x, y = sp.symbols('x y')
    phi = x**2 - y**2  # Silla típica
    
    result = classify_critical_point(phi, (x, y), (0.0, 0.0))
    
    assert result['classification'] == 'punto silla'
    # Debe tener valores propios con signos opuestos
    eigenvalues = result['eigenvalues']
    assert (eigenvalues[0] > 0 and eigenvalues[1] < 0) or (eigenvalues[0] < 0 and eigenvalues[1] > 0)


# ============================================================================
# TESTS DE OPTIMIZACIÓN SIN RESTRICCIONES
# ============================================================================

def test_optimize_unconstrained_simple():
    """Test optimización de (x-1)² + (y-2)²"""
    x, y = sp.symbols('x y')
    phi = (x - 1)**2 + (y - 2)**2
    
    result = optimize_unconstrained(phi, (x, y))
    
    # Debe encontrar mínimo en (1, 2)
    assert len(result['critical_points']) >= 1
    
    cp = result['critical_points'][0]
    assert np.allclose(cp['point'], [1.0, 2.0], atol=1e-4)
    assert cp['classification'] == 'mínimo local'
    assert np.isclose(cp['function_value'], 0.0, atol=1e-6)


def test_optimize_unconstrained_multiple_points():
    """Test con múltiples puntos críticos"""
    x, y = sp.symbols('x y')
    # Función con múltiples extremos
    phi = sp.sin(x) * sp.cos(y)
    
    result = optimize_unconstrained(phi, (x, y))
    
    # Debe encontrar al menos un punto crítico
    assert len(result['critical_points']) >= 1
    assert 'latex_steps' in result


# ============================================================================
# TESTS DE LAGRANGE
# ============================================================================

def test_lagrange_simple():
    """Test Lagrange: maximizar xy sujeto a x+y=10"""
    x, y = sp.symbols('x y')
    phi = x * y
    constraint = x + y - 10
    
    result = solve_lagrange(phi, (x, y), [constraint])
    
    # Solución: x=5, y=5, f=25
    assert len(result['solutions']) >= 1
    
    sol = result['solutions'][0]
    assert np.allclose(sol['point'], [5.0, 5.0], atol=1e-3)
    assert np.isclose(sol['function_value'], 25.0, atol=1e-3)


def test_lagrange_on_circle():
    """Test optimización sobre círculo"""
    x, y = sp.symbols('x y')
    phi = x + y
    constraint = x**2 + y**2 - 1  # Círculo unitario
    
    result = solve_lagrange(phi, (x, y), [constraint])
    
    # Máximo en (1/√2, 1/√2), valor √2
    assert 'solutions' in result
    assert len(result['solutions']) >= 1


def test_cobb_douglas():
    """Test Cobb-Douglas con valores específicos"""
    # x^0.5 * y^0.5, presupuesto 150x + 250y = 50000
    result = cobb_douglas_optimization(
        alpha=0.5,
        beta=0.5,
        px=150,
        py=250,
        M=50000
    )
    
    # Debe tener solución
    assert 'result' in result
    assert len(result['result']['solutions']) >= 1
    
    # x* ≈ 166.67, y* ≈ 100
    sol = result['result']['solutions'][0]
    x_opt, y_opt = sol['point']
    
    assert np.isclose(x_opt, 50000 * 0.5 / 150, atol=1)
    assert np.isclose(y_opt, 50000 * 0.5 / 250, atol=1)


# ============================================================================
# TESTS DE OPTIMIZACIÓN EN REGIONES
# ============================================================================

def test_optimize_triangle():
    """Test optimización en triángulo (caso del quiz)"""
    x, y = sp.symbols('x y')
    # Usar una función simple para testing
    phi = x + y
    
    region = {
        'type': 'triangle',
        'vertices': [(0, 0), (0, 8), (4, 0)]
    }
    
    result = optimize_on_region(phi, (x, y), region)
    
    # Debe tener máximo y mínimo
    assert 'global_maximum' in result
    assert 'global_minimum' in result
    
    # Mínimo debe ser en (0,0)
    min_point = result['global_minimum']['point']
    assert np.allclose(min_point, [0, 0], atol=1e-3)
    
    # Máximo debe ser en (0,8) o (4,0)
    max_point = result['global_maximum']['point']
    assert (np.allclose(max_point, [0, 8], atol=1e-3) or 
            np.allclose(max_point, [4, 0], atol=1e-3))


def test_optimize_rectangle():
    """Test optimización en rectángulo"""
    x, y = sp.symbols('x y')
    phi = (x - 1)**2 + (y - 1)**2
    
    region = {
        'type': 'rectangle',
        'bounds': [(0, 2), (0, 2)]
    }
    
    result = optimize_on_region(phi, (x, y), region)
    
    # Mínimo debe ser en (1,1) interior
    min_point = result['global_minimum']['point']
    assert np.allclose(min_point, [1, 1], atol=1e-2)


def test_max_rectangle_in_ellipse():
    """Test rectángulo inscrito en elipse"""
    a, b = 3.0, 2.0
    result = max_rectangle_in_ellipse(a, b)
    
    # x = a/√2, y = b/√2
    expected_x = a / np.sqrt(2)
    expected_y = b / np.sqrt(2)
    expected_area = 2 * a * b
    
    assert np.isclose(result['optimal_point'][0], expected_x)
    assert np.isclose(result['optimal_point'][1], expected_y)
    assert np.isclose(result['maximum_area'], expected_area)


# ============================================================================
# TESTS DE FORMATO
# ============================================================================

def test_format_exact_sqrt2():
    """Test formateo de √2"""
    value = np.sqrt(2)
    formatted = format_number_prefer_exact(value)
    
    assert '\\sqrt{2}' in formatted


def test_format_exact_fraction():
    """Test formateo de fracciones"""
    value = 0.5
    formatted = format_number_prefer_exact(value)
    
    assert '\\frac{1}{2}' in formatted or '0.5' in formatted


def test_format_exact_sqrt2_over_2():
    """Test formateo de √2/2"""
    value = np.sqrt(2) / 2
    formatted = format_number_prefer_exact(value)
    
    assert '\\sqrt{2}' in formatted or 'frac' in formatted


# ============================================================================
# TESTS DE VISUALIZACIÓN (básicos)
# ============================================================================

def test_visualization_3d_no_crash():
    """Test que visualización 3D no falle"""
    try:
        from optimizacion import visualize_optimization_3d
        
        x, y = sp.symbols('x y')
        phi = x**2 + y**2
        
        fig = visualize_optimization_3d(
            phi,
            (x, y),
            bounds=((-2, 2), (-2, 2)),
            resolution=20
        )
        
        assert fig is not None
    except ImportError:
        pytest.skip("Plotly no disponible")


def test_visualization_2d_no_crash():
    """Test que visualización 2D no falle"""
    try:
        from optimizacion import visualize_contour_2d
        
        x, y = sp.symbols('x y')
        phi = x**2 + y**2
        
        fig = visualize_contour_2d(
            phi,
            (x, y),
            bounds=((-2, 2), (-2, 2)),
            resolution=30
        )
        
        assert fig is not None
    except ImportError:
        pytest.skip("Plotly no disponible")


# ============================================================================
# TESTS DE CASOS EXTREMOS
# ============================================================================

def test_constant_function():
    """Test con función constante"""
    x, y = sp.symbols('x y')
    phi = sp.Float(5.0)
    
    grad_sym, grad_func = compute_gradient(phi, (x, y))
    
    # Gradiente debe ser cero
    assert grad_sym == (0, 0)


def test_linear_function():
    """Test con función lineal (sin extremos)"""
    x, y = sp.symbols('x y')
    phi = 2*x + 3*y
    
    result = optimize_unconstrained(phi, (x, y))
    
    # Puede no encontrar puntos o encontrar en infinito
    # (depende de implementación numérica)
    assert 'critical_points' in result


def test_zero_direction():
    """Test con dirección cero (debe fallar)"""
    x, y = sp.symbols('x y')
    phi = x**2 + y**2
    
    with pytest.raises(ValueError):
        directional_derivative(phi, (x, y), (1, 1), (0, 0))


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

def test_complete_workflow():
    """Test workflow completo: optimización + clasificación + visualización"""
    x, y = sp.symbols('x y')
    phi = x**2 - 2*x + y**2 + 1  # Mínimo en (1, 0)
    
    # 1. Encontrar críticos
    result = optimize_unconstrained(phi, (x, y))
    assert len(result['critical_points']) >= 1
    
    # 2. Verificar clasificación
    cp = result['critical_points'][0]
    assert cp['classification'] == 'mínimo local'
    assert np.allclose(cp['point'], [1, 0], atol=1e-3)
    
    # 3. Verificar valor
    assert np.isclose(cp['function_value'], 0.0, atol=1e-4)
    
    # 4. LaTeX steps deben existir
    assert len(result['latex_steps']) > 0


def test_optimization_with_constraints_workflow():
    """Test workflow con restricciones"""
    x, y = sp.symbols('x y')
    phi = x**2 + y**2
    constraint = x + y - 1  # Línea x+y=1
    
    # Lagrange
    result = solve_lagrange(phi, (x, y), [constraint])
    
    assert len(result['solutions']) >= 1
    assert 'latex_steps' in result
    
    # Mínimo debe estar en (0.5, 0.5)
    sol = result['solutions'][0]
    assert np.allclose(sol['point'], [0.5, 0.5], atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
