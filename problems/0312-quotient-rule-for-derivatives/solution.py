import numpy as np
from typing import List

def evaluate_polynomial(coeffs: List[int], x: float) -> float:
    result = 0
    degree = len(coeffs) - 1
    for coeff in coeffs:
        result += coeff * (x ** degree)
        degree -= 1
    return result

def derivative_polynomial(coeffs: List[int]) -> List[int]:
    result = []
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if power > 0:
            result.append(coeff * power)
    return result

def quotient_rule_derivative(g_coeffs: List[int], h_coeffs: List[int], x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    """
    g_val = evaluate_polynomial(g_coeffs, x)
    h_val = evaluate_polynomial(h_coeffs, x)
    g_prime_val = evaluate_polynomial(derivative_polynomial(g_coeffs), x)
    h_prime_val = evaluate_polynomial(derivative_polynomial(h_coeffs), x)
    
    return (g_prime_val * h_val - g_val * h_prime_val) / (h_val ** 2)
