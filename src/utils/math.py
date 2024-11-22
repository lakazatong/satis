def get_divisors(n):
	return (x for x in range(2, n+1) if n % x == 0)

def compute_gcd(*values):
	from functools import reduce
	import math
	return reduce(math.gcd, values)

def _find_linear_combinations(x, y, idx, current_solution):
	if y == 0:
		yield list(current_solution)
		return
	if idx == len(x):
		return
	max_value = y // x[idx]
	for c in range(max_value + 1):
		current_solution[idx] = c
		yield from _find_linear_combinations(x, y - c * x[idx], idx + 1, current_solution)

def find_linear_combinations(x, y):
	if not x or y < 0:
		return iter([])
	return _find_linear_combinations(x, y, 0, [0] * len(x))

def all_sums(numbers):
	sums = {0: 0}
	for num in numbers:
		new_sums = {s + num: count + 1 for s, count in sums.items()}
		sums.update(new_sums)
	sums.pop(0)
	return sums