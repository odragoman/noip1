import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_contour_with_paths(func, xlim=(-2, 2), ylim=(-2, 2), title="Optimization Function", paths=None, method_names=None, num_levels=20, figsize=(10, 8)):
    """    
    Parameters:
    -----------
    func : function
        Function that takes x and returns (f_val, grad, hess)
    xlim : tuple
        X-axis limits (min, max)
    ylim : tuple  
        Y-axis limits (min, max)
    title : str
        Plot title
    paths : list of lists
        List of optimization paths, each path is a list of [x, y] points
    method_names : list of str
        Names of methods corresponding to each path
    num_levels : int
        Number of contour levels
    figsize : tuple
        Figure size (width, height)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = func(point)[0]  
    
    Z = np.clip(Z, -1e10, 1e10)  
    
    if np.max(Z) / np.min(Z[Z > 0]) > 1000:
        contour = ax.contour(X, Y, Z, levels=num_levels, colors='gray', alpha=0.6)
        contourf = ax.contourf(X, Y, Z, levels=num_levels, alpha=0.3, cmap='viridis')
    else:
        contour = ax.contour(X, Y, Z, levels=num_levels, colors='gray', alpha=0.6)
        contourf = ax.contourf(X, Y, Z, levels=num_levels, alpha=0.3, cmap='viridis')
    
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2e')
    
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Function Value', rotation=270, labelpad=20)
    
    if paths is not None:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, path in enumerate(paths):
            if len(path) == 0:
                continue
                
            path_array = np.array(path)
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   color=color, linewidth=2, alpha=0.8, 
                   label=method_names[i] if method_names else f'Method {i+1}')
            
            ax.plot(path_array[0, 0], path_array[0, 1], 
                   marker='*', color=color, markersize=15, 
                   markeredgecolor='black', markeredgewidth=1)
            
            ax.plot(path_array[1:-1, 0], path_array[1:-1, 1], 
                   marker=marker, color=color, markersize=6, 
                   markerfacecolor='white', markeredgecolor=color, 
                   markeredgewidth=2, linestyle='')

            if len(path_array) > 1:
                ax.plot(path_array[-1, 0], path_array[-1, 1], 
                       marker='X', color=color, markersize=12, 
                       markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if paths is not None and method_names is not None:
        ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    return fig, ax

def plot_function_convergence(f_values_list, method_names, title="Function Value Convergence", figsize=(10, 6), use_log_scale=True):
    """    
    Parameters:
    -----------
    f_values_list : list of lists
        List of function value sequences, one for each method
    method_names : list of str
        Names of methods corresponding to each sequence
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    use_log_scale : bool
        Whether to use logarithmic scale for y-axis
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (f_values, method_name) in enumerate(zip(f_values_list, method_names)):
        if len(f_values) == 0:
            continue
            
        iterations = range(len(f_values))
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        
        marker_every = max(1, len(f_values) // 20)
        
        ax.plot(iterations, f_values, 
               color=color, linewidth=2, linestyle=line_style,
               marker=marker, markersize=4, markevery=marker_every,
               label=method_name, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Function Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    if use_log_scale:
        try:
            all_values = [val for f_vals in f_values_list for val in f_vals if val > 0]
            if len(all_values) > 0 and max(all_values) / min(all_values) > 100:
                ax.set_yscale('log')
        except (ValueError, ZeroDivisionError):
            pass 
    
    plt.tight_layout()
    return fig, ax

def get_function_limits(func, center=(0, 0), radius=3):
    """    
    Parameters:
    -----------
    func : function
        Function to analyze
    center : tuple
        Center point for exploration
    radius : float
        Search radius around center
    
    Returns:
    --------
    xlim, ylim : tuples
        Suggested plot limits
    """
    
    x = np.linspace(center[0] - radius, center[0] + radius, 20)
    y = np.linspace(center[1] - radius, center[1] + radius, 20)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = func(point)[0]
            except:
                Z[i, j] = np.inf
    
    finite_mask = np.isfinite(Z)
    if np.any(finite_mask):
        median_val = np.median(Z[finite_mask])
        mad = np.median(np.abs(Z[finite_mask] - median_val))
        threshold = median_val + 10 * mad
        
        good_region = (Z < threshold) & finite_mask
        if np.any(good_region):
            x_coords = X[good_region]
            y_coords = Y[good_region]
            
            xlim = (np.min(x_coords) - 0.5, np.max(x_coords) + 0.5)
            ylim = (np.min(y_coords) - 0.5, np.max(y_coords) + 0.5)
            
            return xlim, ylim
    
    return (center[0] - radius, center[0] + radius), (center[1] - radius, center[1] + radius)

def plot_optimization_summary(func, func_name, paths, method_names, f_values_list, xlim=None, ylim=None, save_path=None):
    """  
    Parameters:
    -----------
    func : function
        Objective function
    func_name : str
        Name of the function for titles
    paths : list of lists
        Optimization paths for each method
    method_names : list of str
        Names of methods
    f_values_list : list of lists
        Function values for each method
    xlim, ylim : tuples
        Plot limits (auto-determined if None)
    save_path : str
        Path to save the figure (optional)
    """
    
    if xlim is None or ylim is None:
        if paths and len(paths[0]) > 0:
            all_points = np.vstack([np.array(path) for path in paths if len(path) > 0])
            center = np.mean(all_points, axis=0)
            span = np.max(all_points, axis=0) - np.min(all_points, axis=0)
            margin = np.max(span) * 0.3
            
            xlim = (center[0] - span[0]/2 - margin, center[0] + span[0]/2 + margin)
            ylim = (center[1] - span[1]/2 - margin, center[1] + span[1]/2 + margin)
        else:
            xlim, ylim = get_function_limits(func)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    plt.sca(ax1)
    plot_contour_with_paths(func, xlim, ylim, title=f'{func_name} - Optimization Paths', paths=paths, method_names=method_names)
    
    plt.sca(ax2)
    plot_function_convergence(f_values_list, method_names, title=f'{func_name} - Convergence')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_separate_plots(func, func_name, paths, method_names, f_values_list, 
                        xlim=None, ylim=None, save_dir="test_results"):
    """    
    Parameters:
    -----------
    func : function
        Objective function
    func_name : str
        Name of the function for titles
    paths : list of lists
        Optimization paths for each method
    method_names : list of str
        Names of methods
    f_values_list : list of lists
        Function values for each method
    xlim, ylim : tuples
        Plot limits (auto-determined if None)
    save_dir : str
        Directory to save plots
    
    Returns:
    --------
    tuple : (contour_plot_path, convergence_plot_path)
    """
    
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    clean_name = func_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    
    if xlim is None or ylim is None:
        if paths and len(paths[0]) > 0:
            all_points = np.vstack([np.array(path) for path in paths if len(path) > 0])
            center = np.mean(all_points, axis=0)
            span = np.max(all_points, axis=0) - np.min(all_points, axis=0)
            margin = np.max(span) * 0.3
            
            xlim = (center[0] - span[0]/2 - margin, center[0] + span[0]/2 + margin)
            ylim = (center[1] - span[1]/2 - margin, center[1] + span[1]/2 + margin)
        else:
            xlim, ylim = get_function_limits(func)
    
    fig1, ax1 = plot_contour_with_paths(func, xlim, ylim, title=f'{func_name} - Optimization Paths', paths=paths, method_names=method_names)
    
    contour_path = os.path.join(save_dir, f"{clean_name}_contour_paths.png")
    fig1.savefig(contour_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2, ax2 = plot_function_convergence(f_values_list, method_names, title=f'{func_name} - Function Value Convergence')
    
    convergence_path = os.path.join(save_dir, f"{clean_name}_convergence.png")
    fig2.savefig(convergence_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    return contour_path, convergence_path

def print_optimization_summary(x_final, f_final, success, method_name, func_name):

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION SUMMARY: {func_name} - {method_name}")
    print(f"{'='*60}")
    print(f"Final location: x* = [{x_final[0]:10.8f}, {x_final[1]:10.8f}]")
    print(f"Final objective: f(x*) = {f_final:15.10e}")
    print(f"Success: {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}\n")