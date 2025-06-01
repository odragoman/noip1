import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from examples import *
from unconstrained_min import line_search_minimize
from utils import plot_optimization_summary, print_optimization_summary, create_separate_plots

class TestUnconstrainedMinimization(unittest.TestCase):
    """
    Test suite for unconstrained minimization algorithms
    """
    
    def setUp(self):
        """Set up test parameters"""
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.max_iter_rosenbrock_gd = 10000
        
        self.output_dir = "test_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def run_optimization_test(self, func, func_name, x0, xlim, ylim, 
                             max_iter_gd=None, max_iter_newton=None):
        """
        Helper method to run optimization with both methods and create plots
        
        Parameters:
        -----------
        func : function
            Objective function to test
        func_name : str
            Name of the function for reporting
        x0 : numpy.ndarray
            Starting point
        xlim, ylim : tuples
            Plot limits
        max_iter_gd, max_iter_newton : int
            Maximum iterations for each method
        
        Returns:
        --------
        dict : Results dictionary with all optimization data
        """
        
        if max_iter_gd is None:
            max_iter_gd = self.max_iter
        if max_iter_newton is None:
            max_iter_newton = self.max_iter
        
        print(f"\n{'='*80}")
        print(f"TESTING: {func_name}")
        print(f"Starting point: x0 = {x0}")
        print(f"{'='*80}")
        
        print(f"\n--- Running Gradient Descent ---")
        results_gd = line_search_minimize(
            func, x0.copy(), 'gradient_descent', 
            self.obj_tol, self.param_tol, max_iter_gd
        )
        x_gd, f_gd, success_gd, path_gd, fvals_gd = results_gd
        
        print(f"\n--- Running Newton's Method ---")
        results_newton = line_search_minimize(
            func, x0.copy(), 'newton',
            self.obj_tol, self.param_tol, max_iter_newton
        )
        x_newton, f_newton, success_newton, path_newton, fvals_newton = results_newton
        
        print_optimization_summary(x_gd, f_gd, success_gd, "Gradient Descent", func_name)
        print_optimization_summary(x_newton, f_newton, success_newton, "Newton's Method", func_name)
        
        paths = [path_gd, path_newton]
        method_names = ['Gradient Descent', "Newton's Method"]
        f_values_list = [fvals_gd, fvals_newton]
        
        contour_path, convergence_path = create_separate_plots(func, func_name, paths, method_names, f_values_list, xlim=xlim, ylim=ylim, save_dir=self.output_dir)
        
        print(f"Created plots:")
        print(f"  1. Contour with paths: {contour_path}")
        print(f"  2. Function convergence: {convergence_path}")
        
        results = {
            'function_name': func_name,
            'gd_result': {'x': x_gd, 'f': f_gd, 'success': success_gd, 
                         'path': path_gd, 'f_values': fvals_gd},
            'newton_result': {'x': x_newton, 'f': f_newton, 'success': success_newton,
                            'path': path_newton, 'f_values': fvals_newton},
            'contour_plot': contour_path,
            'convergence_plot': convergence_path
        }
        
        return results
    
    def test_quadratic_1(self):

        x0 = np.array([1.0, 1.0])
        results = self.run_optimization_test( quadratic_1, "Quadratic Function 1", x0, xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
        
        np.testing.assert_allclose(results['gd_result']['x'], [0, 0], atol=1e-4)
        np.testing.assert_allclose(results['newton_result']['x'], [0, 0], atol=1e-4)
        
        self.assertLessEqual(len(results['newton_result']['path']), len(results['gd_result']['path']))
        
        self.assertTrue(results['gd_result']['success'])
        self.assertTrue(results['newton_result']['success'])
    
    def test_quadratic_2(self):

        x0 = np.array([1.0, 1.0])
        results = self.run_optimization_test(quadratic_2, "Quadratic Function 2", x0, xlim=(-0.2, 1.2), ylim=(-0.05, 1.05))
        
        np.testing.assert_allclose(results['gd_result']['x'], [0, 0], atol=1e-3)
        np.testing.assert_allclose(results['newton_result']['x'], [0, 0], atol=1e-4)
        
        self.assertLessEqual(len(results['newton_result']['path']), 
                            len(results['gd_result']['path']))
    
    def test_quadratic_3(self):

        x0 = np.array([1.0, 1.0])
        results = self.run_optimization_test(quadratic_3, "Quadratic Function 3", x0, xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
        
        np.testing.assert_allclose(results['gd_result']['x'], [0, 0], atol=1e-3)
        np.testing.assert_allclose(results['newton_result']['x'], [0, 0], atol=1e-4)
        
        self.assertTrue(results['gd_result']['success'])
        self.assertTrue(results['newton_result']['success'])
    
    def test_rosenbrock(self):

        x0 = np.array([-1.0, 2.0])
        results = self.run_optimization_test(rosenbrock, "Rosenbrock Function", x0, xlim=(-2, 2), ylim=(-1, 3), max_iter_gd=self.max_iter_rosenbrock_gd, max_iter_newton=self.max_iter)
        
        expected = np.array([1.0, 1.0])
        
        newton_error = np.linalg.norm(results['newton_result']['x'] - expected)
        self.assertLess(newton_error, 1e-4, "Newton's method should converge close to [1,1]")
        
        gd_error = np.linalg.norm(results['gd_result']['x'] - expected)
        initial_error = np.linalg.norm(x0 - expected)
        self.assertLess(gd_error, initial_error, "GD should make progress toward minimum")
        
        self.assertLess(results['newton_result']['f'], 1e-6, "Newton should achieve low function value")
    
    def test_linear_function(self):

        x0 = np.array([1.0, 1.0])
        results = self.run_optimization_test(linear_function, "Linear Function", x0, xlim=(-2, 4), ylim=(-2, 4))
        
        self.assertFalse(results['gd_result']['success'], "GD should not converge for linear function")
        self.assertFalse(results['newton_result']['success'], "Newton should not converge for linear function")
        
        self.assertEqual(len(results['gd_result']['path']), self.max_iter + 1) 
        self.assertEqual(len(results['newton_result']['path']), self.max_iter + 1)
    
    def test_exponential_function(self):

        x0 = np.array([1.0, 1.0])
        results = self.run_optimization_test(exponential_function, "Exponential Function", x0, xlim=(-1, 2), ylim=(-1, 2))
        
        self.assertTrue(results['gd_result']['success'] or results['newton_result']['success'], "At least one method should converge")
        
        gd_fvals = results['gd_result']['f_values']
        newton_fvals = results['newton_result']['f_values']
        
        if len(gd_fvals) > 1:
            self.assertLess(gd_fvals[-1], gd_fvals[0], "GD should decrease function value")
        
        if len(newton_fvals) > 1:
            self.assertLess(newton_fvals[-1], newton_fvals[0], "Newton should decrease function value")
    
    def test_gradient_accuracy(self):

        print(f"\n{'='*80}")
        print("TESTING GRADIENT ACCURACY")
        print(f"{'='*80}")
        
        functions = [
            (quadratic_1, "Quadratic 1"),
            (quadratic_2, "Quadratic 2"), 
            (quadratic_3, "Quadratic 3"),
            (rosenbrock, "Rosenbrock"),
            (linear_function, "Linear"),
            (exponential_function, "Exponential")
        ]
        
        test_points = [
            np.array([1.0, 1.0]),
            np.array([0.5, -0.5]),
            np.array([-1.0, 2.0])
        ]
        
        def finite_diff_gradient(func, x, h=1e-8):

            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                grad[i] = (func(x_plus)[0] - func(x_minus)[0]) / (2 * h)
            return grad
        
        for func, func_name in functions:
            print(f"\nTesting gradients for {func_name}:")
            for i, x in enumerate(test_points):
                try:

                    _, grad_analytical, _ = func(x)
                    grad_fd = finite_diff_gradient(func, x)
                    
                    error = np.linalg.norm(grad_analytical - grad_fd)
                    print(f"  Point {i+1}: ||grad_analytical - grad_fd|| = {error:.2e}")
                    
                    self.assertLess(error, 1e-6, f"Gradient error too large for {func_name} at point {x}")

                except Exception as e:
                    print(f"  Error testing {func_name} at {x}: {e}")
    
    def test_hessian_accuracy(self):

        print(f"\n{'='*80}")
        print("TESTING HESSIAN ACCURACY")
        print(f"{'='*80}")
        
        functions = [
            (quadratic_1, "Quadratic 1"),
            (quadratic_2, "Quadratic 2"),
            (quadratic_3, "Quadratic 3"),
            (rosenbrock, "Rosenbrock"),
            (exponential_function, "Exponential")
        ]
        
        test_point = np.array([0.5, 0.5])
        
        def finite_diff_hessian(func, x, h=1e-6):

            n = len(x)
            hess = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    x_pp = x.copy()
                    x_pm = x.copy() 
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h
                    x_pp[j] += h
                    
                    x_pm[i] += h
                    x_pm[j] -= h
                    
                    x_mp[i] -= h
                    x_mp[j] += h
                    
                    x_mm[i] -= h
                    x_mm[j] -= h
                    
                    hess[i,j] = (func(x_pp)[0] - func(x_pm)[0] - 
                                func(x_mp)[0] + func(x_mm)[0]) / (4 * h * h)
            
            return hess
        
        for func, func_name in functions:
            if func_name == "Linear":  
                continue
                
            print(f"\nTesting Hessian for {func_name}:")
            try:

                _, _, hess_analytical = func(test_point, hessian_needed=True)
                hess_fd = finite_diff_hessian(func, test_point)
                error = np.linalg.norm(hess_analytical - hess_fd)
                print(f"  ||hess_analytical - hess_fd|| = {error:.2e}")
                
                self.assertLess(error, 1e-4, f"Hessian error too large for {func_name}")
                
            except Exception as e:
                print(f"  Error testing {func_name}: {e}")

def run_all_tests():

    suite = unittest.TestLoader().loadTestsFromTestCase(TestUnconstrainedMinimization)
    
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"  {test}: {error}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL RESULT: {'PASS' if success else 'FAIL'}")
    
    return success

if __name__ == '__main__':
    run_all_tests()