def get_divisors(n):
	return (x for x in range(2, n+1) if n % x == 0)

def compute_gcd(*values):
	from functools import reduce
	import math
	return reduce(math.gcd, values)