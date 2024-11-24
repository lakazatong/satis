import sys, os, pathlib
dirpath = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(dirpath)
sys.path.append(os.path.join(dirpath, 'src'))
if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

import itertools, time, math
from bisect import insort
from utils.fastlist import FastList
from config import config
from utils.other import remove_pairs
from fractions import Fraction
from score import ScoreCalculator
import matplotlib.pyplot as plt
import ast
from node import Node
from tree import Tree
from cost import extract_cost, divide_cost, merge_cost, find_n_m_l
from utils.fractions import fractions_to_integers
import random, numpy as np

# values = [2, 2, 2, 2, 2, 5, 5, 6, 6, 10, 50]
values = [2, 2, 2, 2, 2, 5, 5, 10]
leaves, total_cost = Node.group_values(values)
print(leaves)
# t = Tree([Node(12), Node(30), Node(50)])
# t.attach_leaves(leaves)
# print(t.pretty())
# t.save('tmp/test')

exit(0)

print(divide_cost(1200, 30))

exit(0)

# node = Node(1200)
# node.level = 0
# for child in [Node(960), Node(240)]:
# 	node.children.append(child)
# 	child.parents = [node]

# print(node.pretty())

# Node.expand_extract(node, 240)

# print(node.pretty())

# print(extract_cost(1200, 240))
# exit(0)

# for x in range(2, 1200 + 1):
# 	for d in range(1, x + 1):
# 		if x % d == 0:
# 			divide_cost(x, d)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

x = 840
# x_values = [d for d in range(4, x + 1) if x % d == 0]
# y_values = [extract_cost(x, d) for d in x_values if x % d == 0]
x_values = [c for c in range(1, x + 1)]
y_values = [extract_cost(x, c) for c in x_values]

ax.scatter(x_values, y_values, marker='o', label=f"x={x}")

ax.legend(loc="upper right", fontsize=8)
ax.set_ylabel("Cost", fontsize=12)
ax.set_xlabel("d", fontsize=12)
ax.set_title("Dot Plot for Costs Across Different x Values", fontsize=14)

plt.tight_layout()
plt.show()

exit(0)

for x in range(3, 1200 + 1):
	for c in range(x + 1):
		extract_cost(x, c)

# 7 1200 - 240

min_value = float('inf')
min_case = None
n_cases = 0

for x in range(1, 1200 + 1):
	for d in range(2, x):
		if x % d == 0:
			v = x // d
			n, m, l, _ = find_n_m_l(d)
			n_divided_value = 2**n * 3**m
			divided_value = x / n_divided_value
			to_loop_value = l * divided_value
			new_x = x + to_loop_value
			case = (x, d, v, (n, m), round(new_x, 1), l, round(divided_value, 1))
			if new_x > 1200:
				if n_divided_value < min_value:
					min_value = divided_value
					min_case = case
				print(case)
				n_cases += 1
				n_children = 2 if n > 0 else 3
				if x / n_children + to_loop_value / n_children > 1200:
					print('please no')
					exit(1)

print()
print(n_cases)
if min_case:
	print(min_case)

exit(0)

t = Tree([])

targets = [5, 5, 2, 2, 2, 2]
t._group_targets(targets)

print("-"*20)
print(targets)
print(t)
# print(t.leaves[1].pretty())

exit(0)

def group_values(L, divisions):
	grouped = False
	i = 0
	while i < len(L) - 1:
		ref = L[i]
		n = 1
		while i < len(L) - 1 and L[i + 1] == ref:
			L[i] += L.pop(i + 1)
			n += 1
			grouped = True
		if n > 1:
			divisions.append(('divide', (L[i],), (n,)))
		i += 1
	return grouped

def _find_linear_combinations(x, t, idx, current_solution):
	if t == 0: return [list(current_solution)]
	if idx == len(x): return []
	solutions = []
	max_value = t // x[idx]
	for c in range(max_value + 1):
		current_solution[idx] = c
		solutions.extend(_find_linear_combinations(x, t - c * x[idx], idx + 1, current_solution))
	return solutions

def find_linear_combinations(x, t):
	if not x or t < 0: return []
	return _find_linear_combinations(x, t, 0, [0] * len(x))

def has_sufficient_sources(coeffs, srcs_list, srcs_counts):
	for i, coeff in enumerate(coeffs):
		source = srcs_list[i]
		if srcs_counts[source] < coeff:
			return False
	return True

def apply_best_merge(sources, targets, logs):
	min_cost = float('inf')
	best_merge_sources = None
	best_merge_target = None
	best_n_sources = None
	for t in targets:
		srcs = [src for src in sources if src < t]
		if not srcs: continue
		srcs_set = list(set(srcs))
		n = len(srcs_set)
		tmp = find_linear_combinations(srcs_set, t)
		if not tmp: continue
		all_coeffs = sorted(map(lambda coeffs: (n-coeffs.count(0), coeffs), tmp), key=lambda x: -x[0])
		srcs_counts = {src: srcs.count(src) for src in srcs}
		for n_sources, coeffs in all_coeffs:
			if not has_sufficient_sources(coeffs, srcs_set, srcs_counts):
				continue
			# we pick the first combination of sources that can make that target
			# without considering which merge keeps as many "interesting" sources for later
			# the merge of a cost is only dependent on the number of sources
			if best_n_sources and n_sources == best_n_sources: break
			cost = merge_cost(sum(coeffs), 1)
			if cost >= min_cost:
				continue
			min_cost = cost
			best_merge_sources = []
			best_merge_target = t
			best_n_sources = n_sources
			for i, coeff in enumerate(coeffs):
				best_merge_sources.extend([srcs_set[i]] * coeff)
	if best_merge_sources:
		for val in best_merge_sources:
			sources.remove(val)
		targets.remove(best_merge_target)
		logs.append(('merge', best_merge_sources, tuple()))
		return True
	return False

def apply_best_extract_two_targets(sources, targets, logs):
	min_cost = float('inf')
	best_extraction = None
	best_source = None
	min_target = min(targets)
	for source in (src for src in sources if src > min_target):
		for i, t1 in enumerate(targets):
			for t2 in targets[i + 1:]:
				if source != t1 + t2 or t1 == t2:
					continue
				cost = extract_cost(source, t1)
				if cost < min_cost:
					min_cost = cost
					best_extraction = (t1, t2)
					best_source = source

	if best_extraction:
		t1, t2 = best_extraction
		sources.remove(best_source)
		targets.remove(t1)
		targets.remove(t2)
		logs.append(('extract', (best_source,), (t1, t2)))
		return True
	return False

def apply_best_extract_one_target(sources, targets, logs):
	min_cost = float('inf')
	best_t = None
	best_source = None

	for source in sources:
		for t in targets:
			if source <= t:
				continue
			cost = extract_cost(source, t)
			if cost < min_cost:
				min_cost = cost
				best_t = t
				best_source = source

	if not best_t: return False
	overflow = best_source - best_t
	if best_t == overflow: return False
	sources.remove(best_source)
	sources.append(overflow)
	targets.remove(best_t)
	logs.append(('extract', (best_source,), (best_t, overflow)))
	return True

def solve(sources, targets):
	logs, divisions = [], []
	sources = sorted(sources)
	targets = sorted(targets)

	sources_sum = sum(sources)
	targets_sum = sum(targets)
	if sources_sum < targets_sum:
		sources.append(targets_sum - sources_sum)
	if sources_sum > targets_sum:
		targets.append(sources_sum - targets_sum)

	r, unit_flow_ratio = fractions_to_integers(sources + targets)

	n_sources = len(sources)
	n_targets = len(targets)

	orig_sources = r[:n_sources]
	orig_targets = r[n_sources:]

	sources = [x for x in orig_sources]
	targets = [x for x in orig_targets]

	while True:
	
		sources, targets = remove_pairs(sources, targets)

		if (len(sources) == 0 and len(targets) != 0) or (len(sources) != 0 and len(targets) == 0):
			print('oopsie')
			exit(1)

		if len(targets) == 1:
			logs.append(('merge', tuple(sources), tuple()))
			break

		while group_values(targets, divisions):
			continue
		print(sources, targets)
		exit(0)

		if not apply_best_merge(sources, targets, logs):
			if not apply_best_extract_two_targets(sources, targets, logs):
				apply_best_extract_one_target(sources, targets, logs)

		if sources == targets:
			break

	return orig_sources, orig_targets, logs + [d for d in reversed(divisions)], unit_flow_ratio

def generate_problem():
	number_range = 100
	max_repetitions = 10
	num_sources = random.randint(1, 10)
	num_targets = random.randint(1, 10)
	repetition_probabilities = [i for i in range(1, max_repetitions + 1)]
	total_probability = sum(repetition_probabilities)
	def generate_number_with_repetitions():
		number = random.randint(1, number_range)
		repetitions = random.choices(range(1, max_repetitions + 1), weights=repetition_probabilities, k=1)[0]
		return number, repetitions
	sources = []
	targets = []
	for _ in range(num_sources):
		number, repetitions = generate_number_with_repetitions()
		sources.extend([number] * repetitions)
	for _ in range(num_targets):
		number, repetitions = generate_number_with_repetitions()
		targets.extend([number] * repetitions)
	return sources, targets

tests = [
	([Fraction(1,2), Fraction(13,12)], [Fraction(3,4), Fraction(5,6)]),
	([10, 40], [50]),
	([100], 2*[50]),
	([124], 44*[Fraction(31,11)]),
	([14], [1, 13]),
	(2*[5], [4, 6]),
	([31], 11*[Fraction(31,11)]),
	([40] + 16*[50], 2*[420]),
	([45], [5] + 2*[20]),
	(4*[7], 7*[4]),
	([5], 5*[1]),
	([50], 2*[5] + 2*[20]),
	([780], 5*[156]),
	(7*[4], 4*[7])
	# ([], []),
]

def run_operations(operations, sources):
	for name, srcs, meta in operations:
		if name == 'extract':
			sources.remove(srcs[0])
			for v in meta: sources.append(v)
		elif name == 'divide':
			sources.remove(srcs[0])
			v = srcs[0] // meta[0]
			for _ in range(meta[0]): sources.append(v)
		elif name == 'merge':
			for src in srcs:
				sources.remove(src)
			sources.append(sum(srcs))
	return sorted(sources)

def run_problem(orig_sources, orig_targets):
	print(orig_sources)
	print(orig_targets)
	sources, targets, logs, unit_flow_ratio = solve(orig_sources, orig_targets)
	print(sources)
	print(targets)
	print(logs)
	print(unit_flow_ratio)
	print()
	simulated_sources = run_operations(logs, sources)
	# print(simulated_sources)
	assert simulated_sources == targets

# run_problem([11, 11, 11, 11, 11, 11, 11, 11, 43, 43, 43, 43, 43, 43, 43, 43, 64, 64, 64, 64, 64, 64, 64, 64, 83, 83, 83, 83, 83, 83, 83, 83, 85, 85, 85, 85, 85, 85, 85], [6, 6, 6, 6, 6, 6, 6, 6, 6, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 92, 92, 92, 92, 92, 92, 92, 92, 92, 771])
# run_problem([60, 60, 60], [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 33, 33, 33, 33, 33, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 89, 89, 13, 13, 13, 13, 13, 13, 13])
# run_problem([68, 68, 68, 68, 68, 68, 68, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28], [4, 4, 4, 4, 4, 99, 99, 99, 99, 99, 42, 42, 42, 42, 42, 42, 42, 42, 42, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8])
run_problem([28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 68, 68, 68, 68, 68, 68, 32, 153], [378, 495])
# run_problem(*generate_problem())

exit(0)

for sources, targets in tests:
	run_problem(sources, targets)

exit(0)

# import ipdb

# problem_str = "14 to 1 13"

# roots = Node.unwrap(f"{problem_str}/solution0.data")
# # for root in roots:
# # 	print(root.pretty())
# # exit(0)
# # ipdb.set_trace(context=50)
# Node.expand_roots(roots)
# # for root in roots:
# # 	if root.node_id.endswith('5ad'):
# # 		print(root.children[0].children[0].children)
# # 		break
# Node.save(roots, f"{problem_str}/test")

# exit(0)

# print(merge_cost(4, 1))
print(divide_cost(780, 5))

exit(0)

# print(decimal_representation_info(Fraction(14_000_107, 7_812_500))) # terminate, 9 digits
# print(decimal_representation_info(Fraction(144_573, 96_040))) # non terminating repeating after m digits
# print(decimal_representation_info(Fraction(144_573, 2_401))) # non terminating and never repeating

x = 60
values = [extract_cost(x, i) for i in range(1, x+1)]
plt.plot(range(1, x+1), values, 'o')
plt.xlabel('i')
plt.ylabel('Extract Cost')
plt.title('Extract Cost vs. i (Dot Plot)')
plt.grid()
plt.show()

exit(0)

# targets = [4] * 7

# scoreCalculator = ScoreCalculator(targets)
# print_score = lambda sources: print(f"{sources} -> {scoreCalculator.compute(sources)}")
# # print_score([7] * 4)
# # print_score([7, 1, 6, 7, 7])
# # print_score([7, 1, 6, 14])
# print_score([7, 1, 20])
# # print_score([28])

# exit(0)

# x = 12
# for d in range(1, x+1):
# 	print(d)
# 	print(find_n_m_l(d))
# 	print(divide_cost(x, d))
# 	print()

#      0  1  2  3  4  5  6  7  8  9 10 11 12
# res = [7, 8, 8, 7, 7, 8, 5, 5, 6, 4, 5, 5, 0]
# for i in range(x+1):
# 	print()
# 	cost = divide_cost(x, x, force_l = i)
# 	print(f"{cost = }")
# 	if cost != res[i]:
# 		print(f"failed for {i}")
# 		break

#      1  2  3  4  5  6  7  8  9 10 11 12
# res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# for i in range(1, x+1):
# 	print()
# 	cost = extract_cost(x, i)
# 	print(f"{i = }")
# 	print(f"{cost = }")
# 	# if cost != res[i]:
# 	# 	print(f"failed for {i}")
# 	# 	break

# exit(0)

# # for t in targets:
# # 	print(f"minimum divisor for {t}: {min(get_divisors(t))}")

# h = lambda value: f"{value}: {maximum_value(value, targets)}"
# print(h(5))
# print(h(2))
# print(h(11))
# print(h(20))
# print(h(50))
# print(h(100))
# print(h(25))
# print(h(55))

# exit(0)

# print(divides(Fraction(11, 4), 11))
# print(divides(25, 457))
# print(divides(Fraction(1, 2), Fraction(11, 2)))

# exit(0)

def compute(x, n, m, l):
	d = 2**n*3**m
	b = x / d
	t = b + (l * b) / (d - l)
	return t

# print(compute(780, 1, 1, 1))
# print(compute(28, 0, 2, 2))


def count_merges(n, m, l):
	if l < 0: raise ValueError("cannot merge negative amount of nodes")
	if l <= 1: return l
	r = 0
	while l > 1:
		r += 1
		l -= 2
	return r

def test(x, t):
	if not divides(t, x):
		print("no?")
		return
	t_count = int(x/t)
	n, m, l, splits = find_n_m_l(t_count)
	print("\n" + "\n".join([
		f"splitting {x} {n} times in 2, then {m} times in 3",
		f"results in {t_count}x {t}",
		f"loops back {l} branches to {x}",
		f"uses {splits} splitters + {count_merges(n, m, l)} mergers"
	]))

test(7, 1)
# test(780, 156)
# test(28, 4)
# test(1200, 1)
# test(5, 0.5)
# test(5.5, 0.5)
# test(60, 1)
# test(2285, 125)
# test(457, 25)
# test(115, 5)
# test(11, Fraction(11, 4))
# test(45, 5)
# print(count_merges(60))
