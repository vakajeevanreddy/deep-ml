def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if not matrix or not matrix[0]:
		return []
	if mode == "row":
		return [sum(row)/len(row) for row in matrix]
	elif mode == "column":
		n_rows = len(matrix)
		n_cols = len(matrix[0])
		return [sum(matrix[r][c] for r in range(n_rows))/n_rows for c in range(n_cols)]
	else:
		raise  ValueError("Mode must be 'row' or 'column'")