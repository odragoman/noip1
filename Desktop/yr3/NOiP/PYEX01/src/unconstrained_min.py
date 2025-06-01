import numpy as np

def wolfe_line_search(f, x, direction, c1=0.01, backtrack_factor=0.5, max_backtracks=50):

    alpha = 1.0  
    f_x, grad_x, _ = f(x)
    armijo_threshold = c1 * grad_x.T @ direction
    
    if np.linalg.norm(grad_x) < 1e-10:
        alpha = 1.0

    for _ in range(max_backtracks):
        x_new = x + alpha * direction
        f_new, _, _ = f(x_new)
        
        if f_new <= f_x + alpha * armijo_threshold:
            return alpha
        
        alpha *= backtrack_factor
    
    return alpha  

def line_search_minimize(f, x0, method='gradient_descent', obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    """    
    Parameters:
    -----------
    f : function
        Objective function that returns (f_val, gradient, hessian)
    x0 : numpy.ndarray
        Starting point
    method : str
        'gradient_descent' or 'newton' (user's choice)
    obj_tol : float
        Tolerance for objective function change
    param_tol : float
        Tolerance for parameter change
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    x_final : numpy.ndarray
        Final location
    f_final : float
        Final objective value
    success : bool
        Success flag
    path : list
        List of all x values during optimization
    f_values : list
        List of all function values during optimization
    """
    
    valid_methods = ['gradient_descent', 'newton', 'gd', 'nt']
    if method.lower() not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Valid choices: {valid_methods}")
    
    if method.lower() in ['gd', 'gradient_descent']:
        method = 'gradient_descent'
    elif method.lower() in ['nt', 'newton']:
        method = 'newton'
    
    x = x0.copy()
    path = [x.copy()]
    f_values = []
    
    print(f"Starting {method} optimization")
    print("-" * 50)
    
    for i in range(max_iter):
        hessian_needed = (method == 'newton')
        f_val, grad, hess = f(x, hessian_needed)
        f_values.append(f_val)
        print(f"Iteration {i:3d}: x = [{x[0]:8.6f}, {x[1]:8.6f}], f(x) = {f_val:12.8e}")
        
        if method == 'newton' and hess is not None:
            try:
                newton_decrement_sq = grad.T @ np.linalg.solve(hess, grad)
                if newton_decrement_sq / 2 < obj_tol:
                    print(f"Converged: Newton decrement < tolerance")
                    return x, f_val, True, path, f_values
            except np.linalg.LinAlgError:
                print("Warning: Singular Hessian encountered")
                if np.linalg.norm(grad) < np.sqrt(obj_tol):
                    print(f"Converged: Gradient norm < tolerance")
                    return x, f_val, True, path, f_values
        else:
            if np.linalg.norm(grad) < np.sqrt(obj_tol):
                print(f"Converged: Gradient norm < tolerance")
                return x, f_val, True, path, f_values
        
        if method == 'gradient_descent':
            direction = -grad
        elif method == 'newton':
            try:
                direction = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                print("Warning: Singular Hessian, using gradient direction")
                direction = -grad
        else:
            raise ValueError(f"Unknown method: {method}")
        

        alpha = wolfe_line_search(f, x, direction)
        
        x_new = x + alpha * direction
        
        if np.linalg.norm(x_new - x) < param_tol:
            print(f"Converged: Parameter change < tolerance")
            return x_new, f(x_new)[0], True, path + [x_new], f_values + [f(x_new)[0]]
        

        if len(f_values) > 0:
            f_new = f(x_new)[0]
            if abs(f_new - f_val) < obj_tol:
                print(f"Converged: Objective change < tolerance")
                return x_new, f_new, True, path + [x_new], f_values + [f_new]
        
        x = x_new
        path.append(x.copy())
    
    f_final = f(x)[0]
    print(f"Maximum iterations ({max_iter}) reached")
    return x, f_final, False, path, f_values + [f_final]

class LineSearchOptimizer:

    def __init__(self):
        self.path = []
        self.f_values = []
        self.success = False
        
    def minimize(self, f, x0, method='gradient_descent', obj_tol=1e-12, param_tol=1e-8, max_iter=100):
        """Minimize function using line search"""
        result = line_search_minimize(f, x0, method, obj_tol, param_tol, max_iter)
        
        self.path = result[3]
        self.f_values = result[4]
        self.success = result[2]
        
        return result[:3]  