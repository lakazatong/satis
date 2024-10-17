import random, math, itertools, cProfile
from Binary import Binary

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