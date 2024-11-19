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
	# how many splitters + mergers at minimum to extract c from x
	if c == 0: raise ValueError("c == 0")
	from config import config
	if c in config.conveyor_speeds: return 1
	# split in c's instead of 1's when possible
	# if divides(c, x):
	if x % c == 0:
		n_splits = x // c
		n, m, l, n_splitters = find_n_m_l(n_splits)
		# print(f"{n = }\n{m = }\n{l = }\n{n_splitters = }\n{c = }\n{x = }\n{n_splits = }")
		splitters_count, branches_count = compute_tree_info(n, m)
		n_looping_branches_overflow, n_saved_splitters_overflow = compute_n_looping_branches(n_splits - 1, splitters_count, branches_count)
		n_looping_branches, n_saved_splitters = compute_n_looping_branches(2**n*3**m - n_splits, splitters_count, branches_count)
		return n_splitters - n_saved_splitters_overflow - n_saved_splitters \
			+ merge_cost(n_looping_branches_overflow, 1) \
			+ merge_cost(n_looping_branches, 2) \
			+ merge_cost(l + 1, 1)
	n, m, l, n_splitters = find_n_m_l(x)
	# print(n, m, n_splitters)
	splitters_count, branches_count = compute_tree_info(n, m)
	# print(f"{n = }\n{m = }\n{n_splitters = }\n{c = }\n{x = }\n{splitters_count = }\n{branches_count = }")
	# print(splitters_count, branches_count)
	n_looping_branches_extracted, n_saved_splitters_extracted = compute_n_looping_branches(c, splitters_count, branches_count)
	# print()
	n_looping_branches_overflow, n_saved_splitters_overflow = compute_n_looping_branches(x - c, splitters_count, branches_count)
	return n_splitters - n_saved_splitters_extracted - n_saved_splitters_overflow + \
		+ merge_cost(n_looping_branches_extracted, 1) \
		+ merge_cost(n_looping_branches_overflow, 1) \
		+ merge_cost(l + 1, 1)

def divide_cost(x, d, force_l=None):
	# how many splitters + mergers at minimum to divide x into d
	# d is such that divides(d, x) is True
	if d == 0: raise ValueError("d == 0")
	if d == 1: return 0
	if d == 2 or d == 3: return 1
	n, m, l, n_splitters = find_n_m_l(d)
	# print(f"{n = }, {m = }, {l = }, {n_splitters = }")
	if force_l: l = force_l
	if l == 0: return n_splitters
	if l == x: return 0
	if l < 3:
		# no optimization to be done about looping l or 2 branches back to x
		return n_splitters + 1
	splitters_count, branches_count = compute_tree_info(n, m)
	# print(f"{splitters_count = }")
	# print(f"{branches_count = }")
	r = n_splitters + 1
	n_looping_branches, n_saved_splitters = compute_n_looping_branches(l, splitters_count, branches_count)
	return r - n_saved_splitters + merge_cost(n_looping_branches, 2) # + TODO: handling of the merged node avoiding bottlenecks

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