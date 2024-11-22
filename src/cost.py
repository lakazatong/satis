def count_splits(n, m):
	result = 0
	nodes = 1
	for _ in range(n):
		nodes *= 2
		result += nodes // 2
	for _ in range(m):
		nodes *= 3
		result += nodes // 3
	return result

def find_n_m_l(X):
	# solves 2^n*3^m - l = X
	# by first solving 2^n*3^m > X and deducing l (l stands for 'loop' since it represents how many 1-branches will loop back later)
	# finds such n, m that minimizes count_splits(n, m)
	if X == 0: raise ValueError("X == 0")
	max_n = 0
	max_m = 0
	while 2 ** max_n <= X: max_n += 1
	while 3 ** max_m <= X: max_m += 1
	min_splits = float('inf')
	best_n = best_m = l = 0
	for m in range(max_m + 1):
		for n in range(max_n + 1):
			product = 2 ** n * 3 ** m
			if product < X: continue
			splits = count_splits(n, m)
			if splits >= min_splits: continue
			min_splits = splits
			best_n = n
			best_m = m
			l = product - X
	return best_n, best_m, l, min_splits

def compute_splitters_count(n, m):
	splitters_count = [0] # starting from the bottom
	for _ in range(m): splitters_count.append(splitters_count[-1] * 3 + 1)
	for _ in range(n): splitters_count.append(splitters_count[-1] * 2 + 1)
	return [x for x in reversed(splitters_count)]

def compute_branches_count(n, m):
	branches_count = [2**n*3**m] # starting from the top
	for _ in range(n): branches_count.append(branches_count[-1] // 2)
	for _ in range(m): branches_count.append(branches_count[-1] // 3)
	return branches_count

def compute_tree_info(n, m):
	return compute_splitters_count(n, m), compute_branches_count(n, m)

def compute_n_looping_branches(l, splitters_count, branches_count):
	n_looping_branches = n_saved_splitters = 0
	while l:
		i = 0
		while branches_count[i] > l: i += 1
		n_saved_splitters += splitters_count[i]
		l -= branches_count[i]
		# print(f"{l = }, {n_saved_splitters = }, {i = }")
		n_looping_branches += 1
	return n_looping_branches, n_saved_splitters

def compute_looping_branches(n, m, l, branches_count):
	looping_branches = {}
	while l:
		i = 0
		cur_n, cur_m = 0, 0
		while branches_count[i] > l:
			if cur_n < n:
				cur_n += 1
			else:
				cur_m += 1
			i += 1
		old_val = looping_branches.get((cur_n, cur_m), (0, 0))
		looping_branches[(cur_n, cur_m)] = (old_val[0] + 1, old_val[1])
		to_add = 1
		while cur_n < n:
			cur_n += 1
			to_add *= 2
			old_val = looping_branches.get((cur_n, 0), (0, 0))
			looping_branches[(cur_n, 0)] = (old_val[0], old_val[1] + to_add)
		while cur_m < m:
			cur_m += 1
			to_add *= 3
			old_val = looping_branches.get((n, cur_m), (0, 0))
			looping_branches[(n, cur_m)] = (old_val[0], old_val[1] + to_add)
		l -= branches_count[i]
	return looping_branches
	
def extract_cost(x, c):
	# minimum number of splitters + mergers to extract c from x
	if x <= 2: raise ValueError("x <= 2")
	if c == 0 or c == x: return 0
	from config import config
	if c in config.conveyor_speeds: return 1
	import math
	from fractions import Fraction
	d = x // math.gcd(x, x - c)
	n, m, l, n_splitters = find_n_m_l(d)
	n_divided_value = 2**n*3**m
	divided_value = Fraction(x, n_divided_value)
	to_loop_value = l * divided_value
	new_x = x + to_loop_value
	n_extract = c // Fraction(x, d)
	splitters_count, branches_count = compute_tree_info(n, m)
	# print(compute_looping_branches(n, m, l, branches_count))
	# print(compute_looping_branches(n, m, d - n_extract, branches_count))
	# print(compute_looping_branches(n, m, n_extract, branches_count))
	# print(f"\n{x = }\n{c = }\n{n = }\n{m = }\n{l = }\n{n_splitters = }\n{d = }\n{n_divided_value = }\n{divided_value = }\n{to_loop_value = }\n{n_extract = }\n{splitters_count = }\n{branches_count = }\n{new_x = }")
	n_branches_loop, n_saved_splitters_loop = compute_n_looping_branches(l, splitters_count, branches_count)
	n_branches_overflow, n_saved_splitters_overflow = compute_n_looping_branches(d - n_extract, splitters_count, branches_count)
	n_branches_extract, n_saved_splitters_extract = compute_n_looping_branches(n_extract, splitters_count, branches_count)
	# print(f"{n_saved_splitters_loop = }\n{n_saved_splitters_overflow = }\n{n_saved_splitters_extract = }")
	# print(f"{n_branches_loop = }\n{n_branches_overflow = }\n{n_branches_extract = }\n{merge_cost(n_branches_loop, 1) = }\n{merge_cost(n_branches_overflow, 1) = }\n{merge_cost(n_branches_extract, 1) = }")
	return n_splitters - n_saved_splitters_loop - n_saved_splitters_overflow - n_saved_splitters_extract \
		+ ((merge_cost(n_branches_loop, 1) + 1 + (2 if n > 0 else 3)) if new_x > config.conveyor_speed_limit else merge_cost(n_branches_loop + 1, 1)) \
		+ merge_cost(n_branches_overflow, 1) \
		+ merge_cost(n_branches_extract, 1)

def divide_cost(x, d):
	# minimum number of splitters + mergers to divide x into d values with x % d == 0
	if x <= 1: raise ValueError("x <= 1")
	if d == 0: raise ValueError("d == 0")
	if d == 1: return 0
	if d == 2 or d == 3: return 1
	n, m, l, n_splitters = find_n_m_l(d)
	if l == 0: return n_splitters
	if l == x: return 0
	if l < 3:
		# no optimization to be done about looping l or 2 branches back to x
		return n_splitters + 1
	from config import config
	from fractions import Fraction
	# print(f"{n = }\n{m = }\n{l = }\n{n_splitters = }\n{c = }\n{x = }\n{d = }")
	new_x = x + Fraction(l * x, 2**n*3**m)
	splitters_count, branches_count = compute_tree_info(n, m)
	n_branches_loop, n_saved_splitters_loop = compute_n_looping_branches(l, splitters_count, branches_count)
	return n_splitters - n_saved_splitters_loop + \
		+ (merge_cost(n_branches_loop, 1) + 1 + (2 if n > 0 else 3)) if new_x > config.conveyor_speed_limit else merge_cost(n_branches_loop + 1, 1)

def merge_cost(n, t):
	# how many mergers at minimum to merge n values into t
	if n <= t: return 0
	r = 0
	while n > t:
		r += 1
		n -= 2
	return r

def split_cost():
	return 3