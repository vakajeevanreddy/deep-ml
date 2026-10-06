def conditional_probability(joint_distribution: dict) -> float:
    """
    Compute conditional probability P(A|B) from a joint probability distribution.

    Args:
        joint_distribution (dict): dictionary with keys 
            ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

    Returns:
        float: Conditional probability P(A|B)
    """
    # Joint probability P(A ∩ B)
    p_A_and_B = joint_distribution.get(('A','B'), 0.0)

    # Marginal probability P(B) = P(A ∩ B) + P(A` ∩ B)
    p_B = p_A_and_B + joint_distribution.get(('`A','B'), 0.0)

    # Avoid division by zero
    if p_B == 0:
        return 0.0

    return p_A_and_B / p_B