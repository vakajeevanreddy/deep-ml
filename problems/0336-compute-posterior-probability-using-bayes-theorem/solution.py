def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	if len(priors) != len(likelihoods):
		raise ValueError("Priors and likelihoods must have the same length.")
	unnormalized_posteriors = [p *  l for p,l in zip(priors,likelihoods)]
	evidence = sum(unnormalized_posteriors)
	if evidence == 0:
		raise ValueError("Evidence is zero; check priors and likelihoods.")
	posteriors = [up/ evidence for up in unnormalized_posteriors]
	return posteriors