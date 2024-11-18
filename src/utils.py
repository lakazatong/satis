import os, json, random, networkx as nx, matplotlib.pyplot as plt, re, math, functools

from collections import Counter
from config import config
from functools import partial
from fractions import Fraction

def divides(a, b):
	if a == 0: raise ValueError("a == 0")
	q, remainder = divmod(b, a)
	return q if remainder == 0 and q != 1 else None

# def all_sums(numbers):
# 	sums = {Fraction(0)}
# 	for num in numbers:
# 		sums.update({s + num for s in sums})
# 	sums.remove(Fraction(0))
# 	return sums

def all_sums(numbers):
	sums = {Fraction(0): 0}
	for num in numbers:
		new_sums = {s + num: count + 1 for s, count in sums.items()}
		sums.update(new_sums)
	sums.pop(Fraction(0))
	return sums

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

def compute_looping_branches(n, l, branches_count):
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
		if (cur_n, cur_m) in looping_branches:
			looping_branches[(cur_n, cur_m)] += 1
		else:
			looping_branches[(cur_n, cur_m)] = 1
		l -= branches_count[i]
	return looping_branches

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
	return r - n_saved_splitters + merge_cost(n_looping_branches, 2)
	
def extract_cost(x, c):
	# how many splitters + mergers at minimum to extract c from x
	if c == 0: raise ValueError("c == 0")
	if c in config.conveyor_speeds: return 1
	if divides(c, x):
		n_splits = x // c
		n, m, _, n_splitters = find_n_m_l(n_splits)
		# print(n, m, n_splitters)
		splitters_count, branches_count = compute_tree_info(n, m)
		n_looping_branches_extracted, n_saved_splitters_extracted = compute_n_looping_branches(n_splits - 1, splitters_count, branches_count)
		n_looping_branches_overflow, n_saved_splitters_overflow = compute_n_looping_branches(2**n*3**m - n_splits, splitters_count, branches_count)
		return n_splitters - n_saved_splitters_extracted - n_saved_splitters_overflow + merge_cost(n_looping_branches_extracted, 1) + merge_cost(n_looping_branches_overflow, 1)
	n, m, _, n_splitters = find_n_m_l(x)
	# print(n, m, n_splitters)
	splitters_count, branches_count = compute_tree_info(n, m)
	# print(splitters_count, branches_count)
	n_looping_branches_extracted, n_saved_splitters_extracted = compute_n_looping_branches(c, splitters_count, branches_count)
	# print()
	n_looping_branches_overflow, n_saved_splitters_overflow = compute_n_looping_branches(x - c, splitters_count, branches_count)
	return n_splitters - n_saved_splitters_extracted - n_saved_splitters_overflow + merge_cost(n_looping_branches_extracted, 1) + merge_cost(n_looping_branches_overflow, 1)

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

def print_standing_text(text, length=100):
	print("\r" + " " * length + "\r" + text, end="")

def decimal_representation_info(fraction):
	denominator = fraction.denominator
	power_of_2 = 0
	power_of_5 = 0
	while denominator % 2 == 0:
		denominator //= 2
		power_of_2 += 1
	while denominator % 5 == 0:
		denominator //= 5
		power_of_5 += 1
	m = max(power_of_2, power_of_5)
	if denominator == 1: return True, m # terminating and has m digits after the decimal point
	if power_of_2 == 0 and power_of_5 == 0: return False, None # non-terminating and non-repeating
	return False, m # non-terminating and repeating at some point after m digits

def format_fractions(fractions):
	output = []
	counts = {}

	for frac in fractions:
		if frac in counts:
			counts[frac] += 1
		else:
			counts[frac] = 1

	def with_count(count, frac_str):
		return f"{count}x {frac_str}" if count > 1 else frac_str

	for frac, count in counts.items():
		terminating, m = decimal_representation_info(frac)

		if not m:
			output.append(with_count(count, str(frac)))
			continue

		if terminating:
			# Construct the terminating decimal string
			integer_part = frac.numerator // frac.denominator
			decimal_part = abs(frac.numerator) % frac.denominator

			decimal_str = ''
			for _ in range(m):
				decimal_part *= 10
				decimal_digit = decimal_part // frac.denominator
				decimal_str += str(decimal_digit)
				decimal_part %= frac.denominator

			# Combine integer and decimal parts
			output.append(with_count(count, f"{integer_part}.{decimal_str}"))
			continue

		output.append(with_count(count, str(frac)))

		# graveyard

		# decimal_part = []
		# integer_part = frac.numerator // frac.denominator
		# remainder = frac.numerator % frac.denominator
		
		# seen_remainders = {}
		# index = 0

		# while remainder != 0:
		# 	if remainder in seen_remainders:
		# 		repeat_start = seen_remainders[remainder]
		# 		non_repeating = ''.join(decimal_part[:repeat_start])
		# 		repeating = ''.join(decimal_part[repeat_start:])
		# 		output.append(with_count(count, f"{integer_part}.{non_repeating}({repeating})"))
		# 		break

		# 	seen_remainders[remainder] = index
		# 	remainder *= 10
		# 	decimal_digit = remainder // frac.denominator
		# 	decimal_part.append(str(decimal_digit))
		# 	remainder %= frac.denominator
		# 	index += 1
		# else:
		# 	output.append(with_count(count, str(frac)))

	return ' '.join(output)

def compute_minimum_possible_fraction(values):
	min_fraction = None

	for value in values:
		if value.denominator == 1:
			fraction = Fraction(1, 1)  # Treat integers as Fraction(1, 1)
		else:
			fraction = value - Fraction(value.numerator // value.denominator)  # Get the fractional part

		if min_fraction is None or fraction < min_fraction:
			min_fraction = fraction

	return min_fraction

def compute_gcd(*fractions):
	numerators = [f.numerator for f in fractions]
	denominators = [f.denominator for f in fractions]
	gcd_numerator = functools.reduce(math.gcd, numerators)
	lcm_denominator = functools.reduce(lambda x, y: x * y // math.gcd(x, y), denominators)
	return Fraction(gcd_numerator, lcm_denominator)

def get_divisors(n):
	return (x for x in range(2, n+1) if n % x == 0)

def gcd_incompatible(gcd, value):
	return value < gcd

def get_gcd_incompatible(gcd):
	return partial(gcd_incompatible, gcd)

def remove_pairs(list_a, list_b):
	count_a = Counter(list_a)
	count_b = Counter(list_b)
	for item in count_a.keys():
		if item in count_b:
			pairs_to_remove = min(count_a[item], count_b[item])
			count_a[item] -= pairs_to_remove
			count_b[item] -= pairs_to_remove
	remaining_a = []
	remaining_b = []
	for item, count in count_a.items(): remaining_a.extend([item] * count)
	for item, count in count_b.items(): remaining_b.extend([item] * count)
	return remaining_a, remaining_b

def sort_nodes(nodes):
	return sorted(nodes, key=lambda node: node.value)

def get_node_values(nodes):
	return tuple(map(lambda node: node.value, nodes))

def get_node_ids(nodes):
	return set(map(lambda node: node.node_id, nodes))

def get_short_node_ids(nodes, short=3):
	return set(map(lambda node: node.node_id[-short:], nodes))

def pop_node(node, nodes):
	for i, other in enumerate(nodes):
		if other.node_id == node.node_id:
			return nodes.pop(i)
	return None

def compute_cant_use(target_counts, sources):
	source_counts = {}
	for src in sources:
		if src.value in source_counts:
			source_counts[src.value] += 1
		else:
			source_counts[src.value] = 1
	cant_use = set()
	for src in sources:
		value = src.value
		src_count = source_counts.get(value, None)
		target_count = target_counts.get(value, None)
		if src_count and target_count and max(0, src_count - target_count) == 0:
			cant_use.add(value)
	return cant_use

def get_compute_cant_use(target_counts):
	return partial(compute_cant_use, target_counts)

# def insert_into_sorted(sorted_list, item, key=lambda x: x):
	# low, high = 0, len(sorted_list)
	# while low < high:
	# 	mid = low + (high - low) // 2
	# 	if key(item) > key(sorted_list[mid]):
	# 		low = mid + 1
	# 	else:
	# 		high = mid
	# sorted_list.insert(low, item)

def get_sim_without(value, values):
	sim = [v for v in values]
	sim.remove(value)
	return sim

def parse_fraction(value):
	if '/' in value:
		numerator, denominator = value.split('/')
		return Fraction(int(numerator), int(denominator))
	
	match = re.match(r'(\d*)\.(\d*)\((\d+)\)', value)
	if match:
		whole_part = match.group(1)
		non_repeating = match.group(2)
		repeating = match.group(3)
		
		if not whole_part:
			whole_part = '0'
		
		non_repeating_len = len(non_repeating)
		repeating_len = len(repeating)

		numerator = int(whole_part + non_repeating + repeating) - int(whole_part + non_repeating)
		denominator = (10 ** (non_repeating_len + repeating_len) - 10 ** non_repeating_len)

		return Fraction(numerator, denominator)

	if '.' in value:
		return Fraction(str(float(value))).limit_denominator()
	
	return Fraction(int(value), 1)

def debug_parsed_values(source_values, target_values):
	if source_values is not None and target_values is not None:
		print("Source values:")
		for val in source_values:
			print(f"{val} (as fraction: {val.numerator}/{val.denominator})")
		print("\nTarget values:")
		for val in target_values:
			print(f"{val} (as fraction: {val.numerator}/{val.denominator})")

def parse_user_input(user_input):
	separator = 'to'
	if len(user_input.split(" ")) < 3 or separator not in user_input:
		print(f"Usage: <source_args> {separator} <target_args>")
		return [], []

	source_part, target_part = user_input.split(separator)
	source_args = source_part.strip().split()
	target_args = target_part.strip().split()

	if not source_args:
		print("Error: At least one source value must be provided.")
		return None, None
	if not target_args:
		print("Error: At least one target value must be provided.")
		return None, None

	source_values = []
	i = 0
	while i < len(source_args):
		src = source_args[i]
		if not src.endswith('x'):
			source_value = parse_fraction(src)
			source_values.append(source_value)
			i += 1
			continue
		if len(src) < 2 or not src[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return None, None
		multiplier = int(src[:-1])
		source_value = parse_fraction(source_args[i + 1])
		source_values.extend([source_value] * multiplier)
		i += 2

	target_values = []
	i = 0
	while i < len(target_args):
		target = target_args[i]
		if not target.endswith('x'):
			target_value = parse_fraction(target)
			target_values.append(target_value)
			i += 1
			continue
		if len(target) < 2 or not target[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return None, None
		multiplier = int(target[:-1])
		if i + 1 == len(target_args):
			print("Error: You must provide a target value after Nx.")
			return None, None
		target_value = parse_fraction(target_args[i + 1])
		target_values.extend([target_value] * multiplier)
		i += 2

	# debug_parsed_values(source_values, target_values)
	return source_values, target_values

def generate_test_cases(num_cases, max_size, elements_max_size=1200*2):
	test_cases = []
	for _ in range(num_cases):
		sim = sorted([random.randint(1, elements_max_size) for _ in range(random.randint(1, max_size))])
		targets = sorted([random.randint(1, elements_max_size) for _ in range(random.randint(1, max_size))])
		test_cases.append((sim, targets))
	return test_cases

# graveyard

# class Binary:
# 	def __init__(self, n):
# 		self.n = n
# 		self._arr = [0] * n
# 		self.bit_count = 0

# 	def increment(self):
# 		# returns if it's 0 after the increment
# 		for i in range(self.n):
# 			self._arr[i] = not self._arr[i]
# 			if self._arr[i]:
# 				self.bit_count += 1
# 				return True
# 			self.bit_count -= 1
# 		return False

# 	def __iadd__(self, other):
# 		for _ in range(other - 1): self.increment()
# 		return self.increment()

# 	def __getitem__(self, index):
# 		return self._arr[index]

# 	def __setitem__(self, index, value):
# 		old_bit = self._arr[index]
# 		self._arr[index] = value
# 		self.bit_count += (value - old_bit) 

# 	def __iter__(self):
# 		return iter(self._arr)

# 	def __str__(self):
# 		return str(self._arr)

# def show(G):
# 	edge_labels = nx.get_edge_attributes(G, 'label')
# 	pos = nx.spring_layout(G)
# 	plt.figure(figsize=(12, 8))
# 	nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=10, font_weight='bold')
# 	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
# 	plt.show()

# def load_graph_from_json(file_path):
# 	with open(file_path, 'r') as f:
# 		data = json.load(f)
# 	G = nx.node_link_graph(data)
# 	return G

# def find_shortest_path_with_operations(G, src, dst):
# 	if src not in G or dst not in G or dst >= src: 
# 		return None, None
	
# 	filtered_graph = nx.DiGraph()
# 	filtered_graph.add_nodes_from(G.nodes(data=True))
# 	empty = True
	
# 	for u, v in G.edges():
# 		for op in [2, 3]:
# 			label = str(op)
# 			if G.edges[u, v]['label'] == label and v == int(u / op):
# 				filtered_graph.add_edge(u, v, label=label)
# 				empty = False
# 			if G.edges[v, u]['label'] == label and u == int(v / op):
# 				filtered_graph.add_edge(v, u, label=label)
# 				empty = False

# 	if empty: 
# 		return None, None
	
# 	filtered_graph.remove_nodes_from([node for node in filtered_graph.nodes() if filtered_graph.degree(node) == 0])

# 	try:
# 		# Find the shortest path in the filtered graph
# 		shortest_path = nx.shortest_path(filtered_graph, source=src, target=dst)
# 		operations = []
		
# 		for i in range(len(shortest_path) - 1):
# 			u, v = shortest_path[i], shortest_path[i + 1]
# 			# Get the operation label from the filtered graph
# 			label = filtered_graph.edges[u, v]['label']
# 			operations.append(f"/{label}")  # Format as /2 or /3

# 		return shortest_path, operations
# 	except nx.NetworkXNoPath:
# 		return None, None

# def get_all_pairs_operations(G):
# 	results = []
	
# 	for src in G.nodes():
# 		for dst in G.nodes():
# 			if src > dst:
# 				path, operations = find_shortest_path_with_operations(G, src, dst)
# 				if path is not None:
# 					results.append([src, dst, operations])
	
# 	return results

# if __name__ == "__main__":
# 	graph_file = "graph_data.json"
# 	G = load_graph_from_json(graph_file)
	
# 	all_operations = get_all_pairs_operations(G)
	
# 	for res in all_operations:
# 		print(res)