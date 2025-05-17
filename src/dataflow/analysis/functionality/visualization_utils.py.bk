#!/usr/bin/env python3
"""
Visualization Utilities

This module provides helper functions for visualization styling and formatting
to ensure a consistent and appealing look in the report.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from matplotlib.patches import FancyBboxPatch

def set_default_plot_style():
    """Set up default matplotlib styles for better appearance."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (8.5, 11),
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 
                           'Bitstream Vera Sans', 'sans-serif'],
    })
    
    # Avoid using Unicode characters that might cause warnings
    plt.rcParams['axes.unicode_minus'] = False

def repository_color_palette(num_repos):
    """Generate a color palette for repositories that's more distinct.
    
    Args:
        num_repos: Number of repositories to generate colors for
        
    Returns:
        List of distinct colors
    """
    # Use a combination of qualitative palettes for better distinction
    if num_repos <= 10:
        return sns.color_palette("tab10", num_repos)
    elif num_repos <= 20:
        # Combine two qualitative palettes
        return sns.color_palette("tab10", 10) + sns.color_palette("Set3", num_repos - 10)
    else:
        # For more repositories, use a larger palette with interpolation
        return sns.color_palette("hsv", num_repos)

def adjust_color_brightness(color, factor):
    """Adjust the brightness of a color.
    
    Args:
        color: Color specification (name, hex, or RGB tuple)
        factor: Brightness factor (>1 brightens, <1 darkens)
        
    Returns:
        Adjusted color as hex string
    """
    try:
        # Convert to RGB
        rgb = mcolors.to_rgb(color)
        
        # Adjust brightness
        adjusted = [min(max(c * factor, 0), 1) for c in rgb]
        
        # Convert back to color
        return mcolors.rgb2hex(adjusted)
    except:
        # If conversion fails, return the original color
        return color

def get_quality_color(score):
    """Get a color representing a quality score.
    
    Args:
        score: Quality score between 0 and 1
        
    Returns:
        Color hex code
    """
    if score >= 0.8:
        return '#2ecc71'  # Green for high quality
    elif score >= 0.6:
        return '#f1c40f'  # Yellow for medium-high quality
    elif score >= 0.4:
        return '#e67e22'  # Orange for medium quality
    else:
        return '#e74c3c'  # Red for low quality

def get_quality_gradient_colors(num_colors=10):
    """Get a color gradient for quality visualization.
    
    Args:
        num_colors: Number of colors in the gradient
        
    Returns:
        List of colors from red (low quality) to green (high quality)
    """
    return sns.color_palette("RdYlGn", num_colors)

def add_gradient_line(fig, x_start, x_end, y_pos, color='#3498db', alpha_max=1.0):
    """Add a horizontal line with gradient effect.
    
    Args:
        fig: Matplotlib figure
        x_start: Starting x position (0-1)
        x_end: Ending x position (0-1)
        y_pos: Y position (0-1)
        color: Base color
        alpha_max: Maximum alpha value
    """
    x_points = np.linspace(x_start, x_end, 100)
    for i, x in enumerate(x_points):
        # Calculate alpha to create a gradient effect
        rel_pos = i / len(x_points)
        alpha = alpha_max * (1 - abs(2 * (rel_pos - 0.5)))
        fig.axes[0].plot([x, x+0.01], [y_pos, y_pos], color=color, 
                       alpha=alpha, linewidth=2, transform=fig.transFigure)

def add_styled_box(fig, x, y, width, height, color='#ecf0f1', edge_color='#bdc3c7', 
                 alpha=0.8, corner_radius=0.02):
    """Add a styled box to a figure.
    
    Args:
        fig: Matplotlib figure
        x, y: Bottom-left corner position (0-1)
        width, height: Box dimensions (0-1)
        color: Fill color
        edge_color: Border color
        alpha: Transparency
        corner_radius: Rounded corner radius
        
    Returns:
        The created box patch
    """
    box = FancyBboxPatch(
        (x, y), width, height, 
        fill=True, facecolor=color, alpha=alpha,
        boxstyle=f"round,pad={corner_radius}",
        transform=fig.transFigure,
        edgecolor=edge_color, linewidth=1
    )
    fig.axes[0].add_patch(box)
    return box

def create_diverging_colormap(num_colors=10):
    """Create a diverging colormap.
    
    Args:
        num_colors: Number of colors in the colormap
        
    Returns:
        Matplotlib colormap
    """
    return sns.diverging_palette(230, 20, as_cmap=True)

def create_sequential_colormap(start_color, end_color, num_colors=10):
    """Create a sequential colormap between two colors.
    
    Args:
        start_color: Starting color
        end_color: Ending color
        num_colors: Number of colors in the gradient
        
    Returns:
        List of colors
    """
    # Convert to RGB
    rgb_start = mcolors.to_rgb(start_color)
    rgb_end = mcolors.to_rgb(end_color)
    
    # Create a gradient
    colors = []
    for i in range(num_colors):
        t = i / (num_colors - 1)
        r = rgb_start[0] * (1 - t) + rgb_end[0] * t
        g = rgb_start[1] * (1 - t) + rgb_end[1] * t
        b = rgb_start[2] * (1 - t) + rgb_end[2] * t
        colors.append((r, g, b))
    
    return colors

def add_gradient_background(ax, cmap='Blues', alpha=0.2):
    """Add a gradient background to an axis.
    
    Args:
        ax: Matplotlib axis
        cmap: Colormap name
        alpha: Transparency
    """
    gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
    ax.imshow(gradient, cmap=cmap, alpha=alpha, aspect='auto',
             extent=[0, 1, 0, 1], transform=ax.figure.transFigure)

def format_axis_for_percentage(ax):
    """Format axis ticks as percentages.
    
    Args:
        ax: Matplotlib axis to format
    """
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))