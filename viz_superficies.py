"""
Módulo de Visualización de Superficies y Gradientes
===================================================

Funciones para visualizar campos escalares, superficies de nivel,
y vectores de gradiente.

Autor: Sistema de Cálculo Vectorial
Versión: 1.0
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from typing import Tuple, Optional, Dict, Any, List


def ensure_array(func, *args):
    """Asegura que el resultado de una función lambdify sea siempre un array."""
    result = func(*args)
    if np.isscalar(result):
        # Si es escalar, crear array con el mismo shape que el primer argumento
        shape = args[0].shape if hasattr(args[0], 'shape') else (len(args[0]),)
        return np.full(shape, result)
    return result


def plot_scalar_field_3d(
    phi_sym: sp.Expr,
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2), (-2, 2)),
    n_points: int = 30,
    method: str = 'isosurface',
    title: str = "Campo Escalar φ(x,y,z)"
) -> go.Figure:
    """
    Visualiza un campo escalar φ(x,y,z) en 3D.
    
    Parámetros
    ----------
    phi_sym : sp.Expr
        Campo escalar simbólico
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables (x, y, z)
    bounds : Tuple[Tuple[float, float], ...]
        Límites de visualización
    n_points : int
        Resolución de la malla
    method : str
        'isosurface' para superficies de nivel, 'volume' para volumen
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Visualización del campo escalar
    """
    x_sym, y_sym, z_sym = vars_
    
    # Convertir a función numérica
    phi_func = sp.lambdify(vars_, phi_sym, modules=['numpy'])
    
    # Crear malla 3D
    x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
    y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
    z_range = np.linspace(bounds[2][0], bounds[2][1], n_points)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    # Evaluar campo escalar
    phi_values = ensure_array(phi_func, X, Y, Z)
    
    # Crear figura
    fig = go.Figure()
    
    if method == 'isosurface':
        # Superficies de nivel (isosuperficies)
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=phi_values.flatten(),
            isomin=phi_values.min(),
            isomax=phi_values.max(),
            surface_count=5,  # Número de superficies
            colorscale='Viridis',
            colorbar=dict(title='φ'),
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
    else:
        # Visualización de volumen
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=phi_values.flatten(),
            isomin=phi_values.min(),
            isomax=phi_values.max(),
            opacity=0.1,
            surface_count=10,
            colorscale='Viridis',
            colorbar=dict(title='φ')
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20)),
        scene=dict(
            xaxis=dict(title='x', range=bounds[0]),
            yaxis=dict(title='y', range=bounds[1]),
            zaxis=dict(title='z', range=bounds[2]),
            aspectmode='cube'
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def plot_gradient_field(
    phi_sym: sp.Expr,
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2), (-2, 2)),
    n_points: int = 6,
    scale: float = 0.5,
    show_scalar_surface: bool = True,
    z_plane: Optional[float] = None,
    title: str = "Gradiente ∇φ"
) -> go.Figure:
    """
    Visualiza el gradiente como campo vectorial, opcionalmente con superficie escalar.
    
    Parámetros
    ----------
    phi_sym : sp.Expr
        Campo escalar
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables
    bounds : Tuple[Tuple[float, float], ...]
        Límites
    n_points : int
        Puntos por eje
    scale : float
        Escala de vectores
    show_scalar_surface : bool
        Mostrar superficie del campo escalar
    z_plane : Optional[float]
        Si se especifica, grafica en plano z=constante
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Visualización del gradiente
    """
    x_sym, y_sym, z_sym = vars_
    
    # Calcular gradiente simbólicamente
    grad_x = sp.diff(phi_sym, x_sym)
    grad_y = sp.diff(phi_sym, y_sym)
    grad_z = sp.diff(phi_sym, z_sym)
    
    # Convertir a funciones
    phi_func = sp.lambdify(vars_, phi_sym, modules=['numpy'])
    grad_x_func = sp.lambdify(vars_, grad_x, modules=['numpy'])
    grad_y_func = sp.lambdify(vars_, grad_y, modules=['numpy'])
    grad_z_func = sp.lambdify(vars_, grad_z, modules=['numpy'])
    
    fig = go.Figure()
    
    if z_plane is not None:
        # Visualización 2D en plano z
        x_range = np.linspace(bounds[0][0], bounds[0][1], 30)
        y_range = np.linspace(bounds[1][0], bounds[1][1], 30)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.full_like(X, z_plane)
        
        phi_vals = ensure_array(phi_func, X, Y, Z)
        
        # Contorno del campo escalar
        fig.add_trace(go.Contour(
            x=x_range,
            y=y_range,
            z=phi_vals,
            colorscale='Viridis',
            colorbar=dict(title='φ'),
            contours=dict(showlabels=True),
            name='φ(x,y,z₀)'
        ))
        
        # Vectores de gradiente
        x_vec = np.linspace(bounds[0][0], bounds[0][1], n_points)
        y_vec = np.linspace(bounds[1][0], bounds[1][1], n_points)
        X_vec, Y_vec = np.meshgrid(x_range, y_range)
        Z_vec = np.full_like(X_vec, z_plane)
        
        U = ensure_array(grad_x_func, X_vec, Y_vec, Z_vec)
        V = ensure_array(grad_y_func, X_vec, Y_vec, Z_vec)
        
        # Añadir flechas (quiver en 2D)
        for i in range(len(x_vec)):
            for j in range(len(y_vec)):
                fig.add_annotation(
                    x=X_vec[j, i],
                    y=Y_vec[j, i],
                    ax=X_vec[j, i] + U[j, i] * scale,
                    ay=Y_vec[j, i] + V[j, i] * scale,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='red',
                    opacity=0.7
                )
        
        fig.update_layout(
            title=f"{title} (z={z_plane})",
            xaxis=dict(title='x'),
            yaxis=dict(title='y', scaleanchor='x'),
            width=700,
            height=700
        )
    else:
        # Visualización 3D
        x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
        y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
        z_range = np.linspace(bounds[2][0], bounds[2][1], n_points)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        
        # Evaluar gradiente
        U = ensure_array(grad_x_func, X, Y, Z)
        V = ensure_array(grad_y_func, X, Y, Z)
        W = ensure_array(grad_z_func, X, Y, Z)
        
        # Si se pide, añadir superficie escalar
        if show_scalar_surface:
            # Crear isosuperficie
            x_iso = np.linspace(bounds[0][0], bounds[0][1], 20)
            y_iso = np.linspace(bounds[1][0], bounds[1][1], 20)
            z_iso = np.linspace(bounds[2][0], bounds[2][1], 20)
            X_iso, Y_iso, Z_iso = np.meshgrid(x_iso, y_iso, z_iso, indexing='ij')
            phi_vals = ensure_array(phi_func, X_iso, Y_iso, Z_iso)
            
            fig.add_trace(go.Isosurface(
                x=X_iso.flatten(),
                y=Y_iso.flatten(),
                z=Z_iso.flatten(),
                value=phi_vals.flatten(),
                isomin=phi_vals.min(),
                isomax=phi_vals.max(),
                surface_count=3,
                colorscale='Viridis',
                opacity=0.3,
                showscale=False,
                name='φ'
            ))
        
        # Añadir vectores de gradiente
        magnitude = np.sqrt(U**2 + V**2 + W**2)
        
        fig.add_trace(go.Cone(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            u=U.flatten(),
            v=V.flatten(),
            w=W.flatten(),
            colorscale='Reds',
            sizemode="scaled",
            sizeref=scale,
            showscale=True,
            colorbar=dict(title='||∇φ||', x=1.1),
            name='∇φ'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            scene=dict(
                xaxis=dict(title='x', range=bounds[0]),
                yaxis=dict(title='y', range=bounds[1]),
                zaxis=dict(title='z', range=bounds[2]),
                aspectmode='cube'
            ),
            width=800,
            height=700
        )
    
    return fig


def plot_level_surfaces(
    phi_sym: sp.Expr,
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    level_values: List[float],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2), (-2, 2)),
    n_points: int = 30,
    title: str = "Superficies de Nivel"
) -> go.Figure:
    """
    Grafica múltiples superficies de nivel φ(x,y,z) = c.
    
    Parámetros
    ----------
    phi_sym : sp.Expr
        Campo escalar
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables
    level_values : List[float]
        Valores de c para las superficies de nivel
    bounds : Tuple[Tuple[float, float], ...]
        Límites
    n_points : int
        Resolución
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Superficies de nivel
    """
    x_sym, y_sym, z_sym = vars_
    phi_func = sp.lambdify(vars_, phi_sym, modules=['numpy'])
    
    # Crear malla
    x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
    y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
    z_range = np.linspace(bounds[2][0], bounds[2][1], n_points)
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    phi_values = ensure_array(phi_func, X, Y, Z)
    
    fig = go.Figure()
    
    # Colores para cada superficie
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    for i, level in enumerate(level_values):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=phi_values.flatten(),
            isomin=level,
            isomax=level,
            surface_count=1,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f'φ = {level}',
            opacity=0.7
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='x', range=bounds[0]),
            yaxis=dict(title='y', range=bounds[1]),
            zaxis=dict(title='z', range=bounds[2]),
            aspectmode='cube'
        ),
        width=800,
        height=700,
        showlegend=True
    )
    
    return fig


def plot_gradient_at_point(
    phi_sym: sp.Expr,
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    point: Tuple[float, float, float],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2), (-2, 2)),
    n_points: int = 30,
    title: str = "Gradiente en un Punto"
) -> go.Figure:
    """
    Visualiza el gradiente en un punto específico con plano tangente.
    
    Parámetros
    ----------
    phi_sym : sp.Expr
        Campo escalar
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables
    point : Tuple[float, float, float]
        Punto (x₀, y₀, z₀) donde evaluar
    bounds : Tuple[Tuple[float, float], ...]
        Límites
    n_points : int
        Resolución
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Visualización con punto, gradiente y plano tangente
    """
    x_sym, y_sym, z_sym = vars_
    x0, y0, z0 = point
    
    # Calcular gradiente
    grad_x = sp.diff(phi_sym, x_sym)
    grad_y = sp.diff(phi_sym, y_sym)
    grad_z = sp.diff(phi_sym, z_sym)
    
    # Evaluar en el punto
    phi_func = sp.lambdify(vars_, phi_sym, modules=['numpy'])
    grad_x_func = sp.lambdify(vars_, grad_x, modules=['numpy'])
    grad_y_func = sp.lambdify(vars_, grad_y, modules=['numpy'])
    grad_z_func = sp.lambdify(vars_, grad_z, modules=['numpy'])
    
    phi_0 = float(phi_func(x0, y0, z0))
    grad_x_0 = float(grad_x_func(x0, y0, z0))
    grad_y_0 = float(grad_y_func(x0, y0, z0))
    grad_z_0 = float(grad_z_func(x0, y0, z0))
    
    # Crear figura
    fig = go.Figure()
    
    # Superficie de nivel que pasa por el punto
    x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
    y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
    z_range = np.linspace(bounds[2][0], bounds[2][1], n_points)
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    phi_values = ensure_array(phi_func, X, Y, Z)
    
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=phi_values.flatten(),
        isomin=phi_0,
        isomax=phi_0,
        surface_count=1,
        colorscale='Viridis',
        opacity=0.5,
        showscale=False,
        name=f'φ = {phi_0:.2f}'
    ))
    
    # Punto
    fig.add_trace(go.Scatter3d(
        x=[x0],
        y=[y0],
        z=[z0],
        mode='markers',
        marker=dict(size=8, color='red'),
        name=f'P({x0},{y0},{z0})'
    ))
    
    # Vector gradiente
    scale = 0.5
    fig.add_trace(go.Cone(
        x=[x0],
        y=[y0],
        z=[z0],
        u=[grad_x_0],
        v=[grad_y_0],
        w=[grad_z_0],
        colorscale='Reds',
        sizemode="absolute",
        sizeref=scale,
        showscale=False,
        name='∇φ'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='x', range=bounds[0]),
            yaxis=dict(title='y', range=bounds[1]),
            zaxis=dict(title='z', range=bounds[2]),
            aspectmode='cube'
        ),
        width=800,
        height=700,
        showlegend=True
    )
    
    return fig
