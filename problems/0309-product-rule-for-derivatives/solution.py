from typing import List

def derivative(poly: List[float]) -> List[float]:
    """Return derivative of polynomial given as coefficient list."""
    if len(poly) <= 1:
        return [0.0]
    result = [round(i * poly[i], 4) for i in range(1, len(poly))]
    return result if any(result) else [0.0]

def multiply(p: List[float], q: List[float]) -> List[float]:
    """Multiply two polynomials."""
    res = [0.0] * (len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            res[i+j] += p[i] * q[j]
    return [round(c, 4) for c in res]

def trim(poly: List[float]) -> List[float]:
    """Remove trailing zeros, keep [0.0] if zero polynomial."""
    while len(poly) > 1 and abs(poly[-1]) < 1e-9:
        poly.pop()
    return poly

def product_rule_derivative(f_coeffs: List[float], g_coeffs: List[float]) -> List[float]:
    f_prime = derivative(f_coeffs)
    g_prime = derivative(g_coeffs)
    
    term1 = multiply(f_prime, g_coeffs)
    term2 = multiply(f_coeffs, g_prime)
    
    # Add term1 and term2
    max_len = max(len(term1), len(term2))
    result = [0.0] * max_len
    for i in range(len(term1)):
        result[i] += term1[i]
    for i in range(len(term2)):
        result[i] += term2[i]
    
    result = [round(c, 4) for c in result]
    return trim(result)
