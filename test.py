import random, math, itertools, cProfile, time
from utils import remove_pairs
from itertools import combinations
from collections import Counter

s = set()
s.add('alo')
s.add('belo')
print(s)

exit(0)

# user settings

allowed_divisors = [2, 3] # must be sorted
conveyor_speeds = [60, 120, 270, 480, 780, 1200] # must be sorted

# logging = False
# log_filename = "logs.txt"

# short_repr = True
# include_depth_informations = False

# solutions_filename = lambda i: f"solution{i}"
# solution_regex = re.compile(r'solution\d+\.png') # ext is always png

# internals

# concluding = False
# stop_concluding = False
# solving = False
# stop_solving = False
allowed_divisors_r = allowed_divisors[::-1]
min_sum_count, max_sum_count = allowed_divisors[0], allowed_divisors_r[0]
conveyor_speeds_r = conveyor_speeds[::-1]
conveyor_speed_limit = conveyor_speeds_r[0]
# solutions = []
# solutions_count = 0
# best_size = None
# enqueued_sources = set()
# trim_root = False
# if logging: open(log_filename, "w").close()

def test_distance():
	def compute_distance(sim, target_values):
		gcd = math.gcd(*sim, *target_values)
		def gcd_incompatible(value): return value < gcd
		filtered_conveyor_speeds = [speed for speed in conveyor_speeds if not gcd_incompatible(speed)]
		filtered_conveyor_speeds_r = filtered_conveyor_speeds[::-1]
		sim = list(sim)
		targets = target_values[:]
		distance = 0
		
		# remove common elements
		sim, targets = remove_pairs(sim, targets)

		possible_extractions = [
			(value, speed, overflow)
			for speed in filtered_conveyor_speeds_r
			for value in set(sim)
			if (overflow := value - speed) \
				and value > speed \
				and not gcd_incompatible(overflow) \
				and (speed in targets or overflow in targets)
		]

		# remove perfect extractions
		for i in range(len(possible_extractions)-1, -1, -1):
			value, speed, overflow = possible_extractions[i]
			if value not in sim: continue
			if speed == overflow:
				if len([v for v in targets if v == speed]) < 2: continue
			else:
				if speed not in targets or overflow not in targets: continue
			sim.remove(value)
			targets.remove(speed)
			targets.remove(overflow)
			distance += 1
			possible_extractions.pop(i)

		# remove unperfect extractions
		for value, speed, overflow in possible_extractions:
			if value not in sim: continue
			if speed in targets:
				sim.remove(value)
				targets.remove(speed)
				sim.append(overflow)
				distance += 2
			elif overflow in targets:
				sim.remove(value)
				targets.remove(overflow)
				sim.append(speed)
				distance += 2

		possible_divisions = sorted([
			(value, divisor, divided_value, min(divisor, sum(1 for v in targets if v == divided_value)))
			for divisor in allowed_divisors_r
			for value in set(sim)
			if (divided_value := value // divisor) \
				and value % divisor == 0 \
				and not gcd_incompatible(divided_value) \
				and divided_value in targets
		], key=lambda x: x[3]-x[1])

		# remove perfect divisions
		while possible_divisions:
			value, divisor, divided_value, divided_values_count = possible_divisions[-1]
			if divided_values_count != divisor: break
			if value not in sim or len([v for v in targets if v == divided_value]) < divisor: continue
			sim.remove(value)
			for _ in range(divided_values_count): targets.remove(divided_value)
			possible_divisions.pop()
			distance += 1
		
		# remove unperfect divisions
		for i in range(len(possible_divisions)-1, -1, -1):
			value, divisor, divided_value, divided_values_count = possible_divisions[i]
			if value not in sim or len([v for v in targets if v == divided_value]) < divided_values_count: continue
			sim.remove(value)
			for _ in range(divided_values_count): targets.remove(divided_value)
			for _ in range(divisor - divided_values_count): sim.append(divided_value)
			distance += 2

		# remove all possible merges that yield a target, prioritizing the ones that merge the most amount of values in sim
		for target in reversed(sorted(targets[:])):
			possible_merges = sorted([
				comb
				for r in range(min_sum_count, max_sum_count + 1)
				for comb in combinations(sim, r)
				if sum(comb) == target
			], key=lambda x: len(x))
			if possible_merges:
				comb = possible_merges[-1]
				for v in comb: sim.remove(v)
				targets.remove(target)
				distance += 1

		return distance + len(sim) + len(targets)

	def generate_test_cases(num_cases, size):
		test_cases = []
		for _ in range(num_cases):
			sim = [random.randint(1, size) for _ in range(random.randint(1, 100))]
			targets = [random.randint(1, size) for _ in range(random.randint(1, 100))]
			test_cases.append((sim, targets))
		return test_cases

	for sim, target_values in generate_test_cases(1000, 1200):
		compute_distance(sim[:], target_values)

# cProfile.run("test_distance()")
test_distance()

exit(0)

class Node:
	def __init__(self, value, parents=[], children=[]):
		self.value = value
		self.parents = parents
		self.children = children

min_sum_count = 2
max_sum_count = 3
sources = [Node(random.randint(0, 100)) for _ in range(26)]
source_values_length = len(sources)

def insert_into_sorted(sorted_list, item, key=lambda x: x):
	low, high = 0, len(sorted_list)
	while low < high:
		mid = low + (high - low) // 2
		if key(item) > key(sorted_list[mid]):
			low = mid + 1
		else:
			high = mid
	sorted_list.insert(low, item)

def sum_combinations(nodes, cant_use):
	n = len(nodes)
	indices = range(n)
	enqueued_sources = set()
	simulations = []
	for to_sum_count in range(min_sum_count, max_sum_count + 1):
		for indices_comb in itertools.combinations(indices, to_sum_count):
			comb = [nodes[i] for i in indices_comb]
			if any(node.value in cant_use for node in comb): continue
			if all(len(node.parents) == 1 for node in comb):
				parent = comb[0].parents[0]
				if all(node.parents[0] is parent for node in comb) and len(parent.children) == to_sum_count and (parent.parents or source_values_length == 1): continue
			sim = sorted(node.value for node in nodes if node not in comb)
			summed_value = sum(node.value for node in comb)
			# if gcd_incompatible(summed_value) or summed_value > conveyor_speed_limit: continue
			insert_into_sorted(sim, summed_value)
			sim = tuple(sim)
			if sim in enqueued_sources: continue
			enqueued_sources.add(sim)
			simulations.append((sim, list(indices_comb)))
	
	return simulations

def get_merge_sims(nodes, cant_use):
	enqueued_sources = set()
	simulations = []
	n = len(nodes)
	
	if n < 2: return simulations

	seen_sums = set()
	binary = Binary(n)
	binary[1] = True

	def get_merge_sim(to_sum_count):
		nonlocal seen_sums
		to_not_sum_indices = []
		i = 0
		
		while not binary[i]:
			to_not_sum_indices.append(i)
			i += 1
		
		src = nodes[i]
		
		if src.value in cant_use: return None
		
		to_sum_indices = [i]
		parent = src.parents[0] if src.parents else None
		same_parent = len(src.parents) == 1
		
		while i < n - 1:
			i += 1
			
			if not binary[i]:
				to_not_sum_indices.append(i)
				continue

			src = nodes[i]
			
			if src.value in cant_use: return None
			
			if len(src.parents) != 1 or not src.parents[0] is parent:
				same_parent = False
			
			to_sum_indices.append(i)

		if same_parent and to_sum_count == len(parent.children) and (parent.parents or source_values_length == 1): return None

		to_sum_values = sorted([nodes[i].value for i in to_sum_indices])
		summed_value = sum(to_sum_values)
		# if gcd_incompatible(summed_value) or summed_value > conveyor_speed_limit: return None
		
		to_sum_values = tuple(sorted(to_sum_values))
		if to_sum_values in seen_sums: return None
		seen_sums.add(to_sum_values)

		sim = tuple(sorted([nodes[i].value for i in to_not_sum_indices] + [summed_value]))
		if sim in enqueued_sources: return None
		enqueued_sources.add(sim)
		return sim, to_sum_indices

	while binary.increment():
		to_sum_count = binary.bit_count
		
		if to_sum_count < min_sum_count or to_sum_count > max_sum_count: continue

		r = get_merge_sim(to_sum_count)
		if r: simulations.append(r)
	
	return simulations

def main():
	# r1 = sorted(sum_combinations(sources, set()), key=lambda x: hash(x[0]))
	r2 = sorted(get_merge_sims(sources, set()), key=lambda x: hash(x[0]))
	# for i in range(len(r1)):
	# 	sim1 = r1[i]
	# 	sim2 = r2[i]
	# 	if sim1 != sim2:
	# 		print(sim1)
	# 		print(sim2)
	# 		print()
	# 		print("fail")
	# 		break

cProfile.run("main()")

exit(0)

queue = list(i for i in range(1, 101))

def dequeue(queue):
	n = len(queue)
	i = 1
	while True:
		tmp = 1 << (i - 1)
		prob = 1 / (tmp << 1)
		idx = round((1 - 1 / tmp) * n)
		print(round(prob * 100, 1), idx)
		i += 1
		if i > n or idx >= n: return queue.pop(-1)
		# if random.random() < prob: return queue.pop(idx)

print(dequeue(list(i for i in range(1, 101))))