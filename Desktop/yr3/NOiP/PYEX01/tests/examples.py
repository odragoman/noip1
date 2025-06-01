import numpy as np

def quadratic_1(x, hessian_needed=False):
    """Quadratic function with Q = I (circular contours)"""
    Q = np.array([[1, 0], [0, 1]])
    f_val = x.T @ Q @ x
    gradient = 2 * Q @ x
    hessian = 2 * Q if hessian_needed else None
    return f_val, gradient, hessian

def quadratic_2(x, hessian_needed=False):
    """Quadratic function with Q = diag([1, 100]) (axis-aligned ellipses)"""
    Q = np.array([[1, 0], [0, 100]])
    f_val = x.T @ Q @ x
    gradient = 2 * Q @ x
    hessian = 2 * Q if hessian_needed else None
    return f_val, gradient, hessian

def quadratic_3(x, hessian_needed=False):
    """Quadratic function with rotated ellipse"""

    c, s = np.sqrt(3)/2, 0.5
    R = np.array([[c, -s], [s, c]])
    D = np.array([[100, 0], [0, 1]])
    Q = R.T @ D @ R
    
    f_val = x.T @ Q @ x
    gradient = 2 * Q @ x
    hessian = 2 * Q if hessian_needed else None
    return f_val, gradient, hessian

def rosenbrock(x, hessian_needed=False):
    """Rosenbrock function: 100(x2 - x1^2)^2 + (1 - x1)^2"""
    x1, x2 = x[0], x[1]
    
    f_val = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    

    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    gradient = np.array([df_dx1, df_dx2])
    
    hessian = None
    if hessian_needed:

        d2f_dx1dx1 = -400 * (x2 - 3*x1**2) + 2
        d2f_dx1dx2 = -400 * x1
        d2f_dx2dx2 = 200
        hessian = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                           [d2f_dx1dx2, d2f_dx2dx2]])
    
    return f_val, gradient, hessian

def linear_function(x, hessian_needed=False):
    """Linear function f(x) = a^T x"""
    a = np.array([2, -1]) 
    f_val = a.T @ x
    gradient = a
    hessian = np.zeros((2, 2)) if hessian_needed else None
    return f_val, gradient, hessian

def exponential_function(x, hessian_needed=False):
    """f(x1, x2) = exp(x1 + 3*x2 - 0.1) + exp(x1 - 3*x2 - 0.1) + exp(-x1 - 0.1)"""
    x1, x2 = x[0], x[1]
    
    term1 = np.exp(x1 + 3*x2 - 0.1)
    term2 = np.exp(x1 - 3*x2 - 0.1)
    term3 = np.exp(-x1 - 0.1)
    f_val = term1 + term2 + term3

    df_dx1 = term1 + term2 - term3
    df_dx2 = 3*term1 - 3*term2
    gradient = np.array([df_dx1, df_dx2])
    
    hessian = None
    if hessian_needed:
        
        d2f_dx1dx1 = term1 + term2 + term3
        d2f_dx1dx2 = 3*term1 - 3*term2
        d2f_dx2dx2 = 9*term1 + 9*term2
        hessian = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                           [d2f_dx1dx2, d2f_dx2dx2]])
    
    return f_val, gradient, hessian

FUNCTIONS = {
    'quadratic_1': quadratic_1,
    'quadratic_2': quadratic_2,
    'quadratic_3': quadratic_3,
    'rosenbrock': rosenbrock,
    'linear': linear_function,
    'exponential': exponential_function
}