import math, time, random, copy, traceback, itertools

from utils import remove_pairs, sort_nodes, get_node_values, get_node_ids, pop_node, insert_into_sorted, get_sim_without, log, clear_solution_files, parse_user_input
from config import config
from node import Node

class SatisSolver:

	def __init__(self):
		# independant from any problem
		self.extract_sims_cache = {}
		self.divide_sims_cache = {}
		self.merge_sims_cache = {}

		self.reset()
	
	def load(self, user_input):
		source_values, target_values = parse_user_input(user_input)
		if not source_values and not target_values: return False
		
		self.reset()
		
		self.source_values = sorted(source_values)
		self.target_values = sorted(target_values)
		self.sources_total = sum(source_values)
		self.targets_total = sum(target_values)
		
		if self.sources_total > self.targets_total:
			value = self.sources_total - self.targets_total
			insert_into_sorted(self.target_values, value)
			self.targets_total += value
			print(f"\nTargets were lacking, generated a {value} node as target")
		
		elif self.sources_total < self.targets_total:
			value = self.targets_total - self.sources_total
			insert_into_sorted(self.source_values, value)
			self.sources_total += value
			print(f"\nSources were lacking, generated a {value} node as source")

		self.source_values_length = len(self.source_values)
		self.target_values_length = len(self.target_values)
		self.target_counts = {
			value: self.target_values.count(value) for value in set(self.target_values)
		}
		self.gcd = math.gcd(*self.source_values, *self.target_values)

		self.filtered_conveyor_speeds = [speed for speed in config.conveyor_speeds if not self.gcd_incompatible(speed)]
		self.filtered_conveyor_speeds_r = self.filtered_conveyor_speeds[::-1]
		print(f"\ngcd = {self.gcd}, filtered conveyor speeds = {self.filtered_conveyor_speeds}")
		
		self.node_sources = list(map(lambda value: Node(value), self.source_values))
		if self.source_values_length > 1:
			self.sources_root = Node(sum(self.source_values))
			self.sources_root.children = self.node_sources
			for child in self.sources_root.children:
				child.parents.append(self.sources_root)
			self.trim_root = True
		else:
			self.sources_root = self.node_sources[0]
			self.trim_root = False
		
		self.sources_root.compute_size(self.trim_root)

		return True

	def gcd_incompatible(self, value):
		return value < self.gcd

	def reset(self):
		self.solving = False
		self.done_solving = False
		self.concluding = False
		self.done_concluding = False

		self.compute_distances_cache = {}

		self.trim_root = False
		
		self.solutions = []
		self.solutions_count = 0
		self.best_size = None

		if config.logging: open(config.log_filename, "w").close()

	def gcd_incompatible(self, value):
		return value < self.gcd

	def get_extract_sim(self, sources, i):
		src = sources[i]
		simulations = []
		tmp_sim = None

		for speed in self.filtered_conveyor_speeds:
			if src.value <= speed: break
			overflow = src.value - speed
			
			if self.gcd_incompatible(overflow): continue

			if not tmp_sim: tmp_sim = get_sim_without(src.value, sources)
			sim = tuple(sorted(tmp_sim + [speed, overflow]))
			simulations.append((sim, (i, speed)))
		
		return simulations

	def get_extract_sims(self, sources, source_values):
		simulations = []

		if self.solutions and sources[0].get_root().size + 2 > self.best_size: return simulations
		
		cached_simulations = self.extract_sims_cache.get(source_values)
		if cached_simulations: return cached_simulations

		seen_values = set()
		n = len(sources)

		for i in range(n):
			if not self.solving: break
			src = sources[i]
			if src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(self.get_extract_sim(sources, i))

		self.extract_sims_cache[source_values] = simulations
		return simulations

	def get_divide_sim(self, sources, i):
		src = sources[i]
		simulations = []
		tmp_sim = None
		sources_root = None

		for divisor in config.allowed_divisors:
			if not self.solving: break

			if not src.can_split(divisor): continue
			
			divided_value = src.value // divisor
			if self.gcd_incompatible(divided_value): continue

			if not tmp_sim: tmp_sim = get_sim_without(src.value, sources)
			sim = tuple(sorted(tmp_sim + [divided_value] * divisor))
			simulations.append((sim, (i, divisor)))

		return simulations

	def get_divide_sims(self, sources, source_values):
		cached_simulations = self.divide_sims_cache.get(source_values)
		if cached_simulations:
			if self.solutions:
				sources_size = sources[0].get_root().size
				return list(filter(lambda sim: sources_size + sim[1][1] <= self.best_size, cached_simulations))
			return cached_simulations

		simulations = []
		seen_values = set()
		n = len(sources)
		
		for i in range(n):
			if not self.solving: break
			src = sources[i]
			if src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(self.get_divide_sim(sources, i))
		
		self.divide_sims_cache[source_values] = simulations
		return simulations

	def get_merge_sims(self, sources, source_values):
		simulations = []

		if self.solutions and sources[0].get_root().size + 1 > self.best_size: return simulations

		cached_simulations = self.merge_sims_cache.get(source_values)
		if cached_simulations: return cached_simulations
		
		n = len(sources)
		indices = range(n)
		seen_sims = set()
		
		for to_sum_count in range(config.min_sum_count, config.max_sum_count + 1):
			if not self.solving: break
			for to_sum_indices in itertools.combinations(indices, to_sum_count):
				if not self.solving: break
				
				to_sum_indices = list(to_sum_indices)
				to_sum_nodes = [sources[i] for i in to_sum_indices]
				node_ids = get_node_ids(to_sum_nodes)

				sim = sorted(node.value for node in sources if node.node_id not in node_ids)
				summed_value = sum(node.value for node in to_sum_nodes)
				
				if self.gcd_incompatible(summed_value) or summed_value > config.conveyor_speed_limit: continue
				
				insert_into_sorted(sim, summed_value)
				sim = tuple(sim)
				
				if sim in seen_sims: continue
				seen_sims.add(sim)
				
				simulations.append((sim, (to_sum_indices, to_sum_count)))

		self.merge_sims_cache[source_values] = simulations
		return simulations

	def compute_distance(self, sim):
		if sim in self.compute_distances_cache: return self.compute_distances_cache[sim]
		original_sim = sim
		sim = list(sim)
		targets = self.target_values[:]
		distance = 0
		
		# remove common elements
		sim, targets = remove_pairs(sim, targets)

		sim_set = set(sim)
		possible_extractions = [
			(value, speed, overflow)
			for speed in self.filtered_conveyor_speeds_r
			for value in sim_set
			if (overflow := value - speed) \
				and value > speed \
				and not self.gcd_incompatible(overflow) \
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

		sim_set = set(sim)
		possible_divisions = sorted([
			(value, divisor, divided_value, min(divisor, sum(1 for v in targets if v == divided_value)))
			for divisor in config.allowed_divisors_r
			for value in sim_set
			if (divided_value := value // divisor) \
				and value % divisor == 0 \
				and not self.gcd_incompatible(divided_value) \
				and divided_value in targets
		], key=lambda x: x[3]-x[1])

		# remove perfect divisions
		for i in range(len(possible_divisions)-1, -1, -1):
			value, divisor, divided_value, divided_values_count = possible_divisions[i]
			if divided_values_count != divisor: break
			if value not in sim or len([v for v in targets if v == divided_value]) < divisor: continue
			sim.remove(value)
			for _ in range(divided_values_count): targets.remove(divided_value)
			possible_divisions.pop(i)
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
				for r in range(config.min_sum_count, config.max_sum_count + 1)
				for comb in itertools.combinations(sim, r)
				if sum(comb) == target
			], key=lambda x: len(x))
			if possible_merges:
				comb = possible_merges[-1]
				for v in comb: sim.remove(v)
				targets.remove(target)
				distance += 1

		r = distance + len(sim) + len(targets)
		self.compute_distances_cache[original_sim] = r
		return r

	def compute_cant_use(self, sources):
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
			target_count = self.target_counts.get(value, None)
			if src_count and target_count and max(0, src_count - target_count) == 0:
				cant_use.add(value)
		return cant_use

	# computes how close the sources are from the target_values
	# the lower the better
	def compute_sources_score(self, sources, past):
		n = len(sources)
		simulations = []
		source_values = get_node_values(sources)
		cant_use = self.compute_cant_use(sources)
		simulations.extend(self.get_extract_sims(sources, source_values))
		simulations.extend(self.get_divide_sims(sources, source_values))
		simulations.extend(self.get_merge_sims(sources, source_values))
		score = -1
		# it required at least one operation to get there, hence the 1 +
		if simulations: score = 1 + min(self.compute_distance(sim) for sim, _ in simulations)
		return score

	def is_solution(self, sources):
		# assume the given sources are sorted by value
		n = len(sources)
		if n != self.target_values_length: return False
		for i in range(n):
			if sources[i].value != self.target_values[i]:
				return False
		return True

	def solution_found(self, new_solution_root):
		# return if found better size
		if len(self.solutions) == 0 or new_solution_root.size < self.best_size:
			self.solutions = [new_solution_root]
			self.best_size = new_solution_root.size
			self.solutions_count = 1
			print(" " * 10 + f"\rFound {self.solutions_count} solutions of size {self.best_size}", end="")
			return True
		elif new_solution_root.size == self.best_size:
			self.solutions.append(new_solution_root)
			self.solutions_count += 1
			print(" " * 10 + f"\rFound {self.solutions_count} solutions of size {self.best_size}", end="")
			return False
		print("impossible case reached, should have been checked already")
		self.solving = False

	def solve(self):
		print(f"\nsolving: {self.source_values} to {self.target_values}\n")

		queue = []

		def purge_queue():
			nonlocal queue
			for i in range(len(queue) - 1, -1, -1):
				if not self.solving: break
				sources, *_ = queue[i]
				if sources[0].get_root().size >= self.best_size: queue.pop(i)

		def enqueue(nodes, past):
			nonlocal queue
			nodes_root = nodes[0].get_root()
			nodes_root.compute_size(self.trim_root)
			if self.is_solution(nodes):
				if self.solution_found(nodes_root): purge_queue()
				return
			score = self.compute_sources_score(nodes, past)
			if score < 0: return
			to_insert = (nodes, score, past)
			insert_into_sorted(queue, to_insert, key=lambda x: x[1])

		def dequeue():
			nonlocal queue
			n = len(queue)
			if n < 3: return queue.pop(0)
			return queue.pop(0 if random.random() < 0.8 else random.randrange(1, n))

		enqueue(self.node_sources, set())

		while self.solving and queue:
			sources, score, past = dequeue()
			
			sources_root = sources[0].get_root()
			source_values = get_node_values(sources)
			n = len(sources)
			cant_use = self.compute_cant_use(sources)
			sources_id = get_node_ids(sources)

			log_op = lambda op: log(f"\n\nFROM\n{sources_root}\nDID\n{op}")

			def copy_sources():
				_, leaves = sources_root.deepcopy()
				past_copy = copy.deepcopy(past)
				past_copy.add(source_values)
				return sort_nodes([leaf for leaf in leaves if leaf.node_id in sources_id]), past_copy

			def try_extract():
				simulations = self.get_extract_sims(sources, source_values)
				for sim, (i, speed) in simulations:
					if not self.solving: break

					if sim in past: continue

					src = sources[i]
					
					if src.value in cant_use: continue
					
					parent_values = set(get_node_values(src.parents))
					# if speed in parent_values then it would have been better to leave it as is
					# and merge all the other values to get the overflow value
					# we would get by exctracting speed amount
					# same logic applies if overflow is in parent values
					if speed in parent_values or src.value - speed in parent_values: continue

					copy, past_copy = copy_sources()
					src_copy = copy[i]
					pop_node(src_copy, copy)

					if config.logging: log_op(f"{src_copy} - {speed}")

					for node in src_copy - speed:
						insert_into_sorted(copy, node, lambda node: node.value)
					enqueue(copy, past_copy)

			def try_divide():
				simulations = self.get_divide_sims(sources, source_values)
				parents_value_sum, n_parents = None, None
				for sim, (i, divisor) in simulations:
					if not self.solving: break

					src = sources[i]

					if src.value in cant_use: continue
					
					if sim in past: continue
					
					if not parents_value_sum:
						parents_value_sum = sum(get_node_values(src.parents))
						n_parents = len(src.parents)
					if parents_value_sum == src.value and n_parents == divisor: continue

					copy, past_copy = copy_sources()
					src_copy = copy[i]
					pop_node(src_copy, copy)

					if config.logging: log_op(f"{src_copy} / {divisor}")

					for node in src_copy / divisor:
						insert_into_sorted(copy, node, lambda node: node.value)
					enqueue(copy, past_copy)

			def try_merge():
				simulations = self.get_merge_sims(sources, source_values)
				for sim, (to_sum_indices, to_sum_count) in simulations:
					if not self.solving: break

					to_sum_nodes = [sources[i] for i in to_sum_indices]

					if any(node.value in cant_use for node in to_sum_nodes): continue

					if all(len(node.parents) == 1 for node in to_sum_nodes):
						parent = to_sum_nodes[0].parents[0]
						if all(node.parents[0] is parent for node in to_sum_nodes) and len(parent.children) == to_sum_count and (parent.parents or self.source_values_length == 1): continue

					if sim in past: continue

					copy, past_copy = copy_sources()
					to_sum = [copy[i] for i in to_sum_indices]
					list(map(lambda src: pop_node(src, copy), to_sum))

					if config.logging: log_op("+".join(str(ts) for ts in to_sum))

					insert_into_sorted(copy, to_sum[0] + to_sum[1:], lambda node: node.value)
					enqueue(copy, past_copy)

			try_divide()
			if not self.solving: break
			try_extract()
			if not self.solving: break
			try_merge()

	def conclude(self):
		if not self.solutions: return
		clear_solution_files()
		print()
		for i in range(len(self.solutions)):
			if not self.concluding: break
			solution = self.solutions[i]
			if self.trim_root:
				for child in solution.children:
					child.compute_levels()
			else:
				solution.compute_levels()
			solution.visualize(config.solutions_filename(i), self.trim_root)

	# simple state machine
	def stop(self):
		if self.solving:
			self.solving = False
		elif self.done_solving and self.concluding:
			self.concluding = False

	def run(self):
		self.solving = True
		self.solve()
		self.solving = False
		self.done_solving = True

		self.concluding = True
		self.conclude()
		self.concluding = False
		self.done_concluding = True
		
		self.running = False

	def close(self):
		print("backend close called")
		pass

# graveyard

# def _simplify_merge(nodes):
# 	# Step 1: Merge nodes with the same value until all are different
# 	has_merged = False
# 	while True:
# 		merged_nodes = []
# 		done = True
# 		i = 0

# 		while i < len(nodes):
# 			current_node = nodes[i]
# 			current_value = current_node.value
# 			same_value_nodes = []

# 			i += 1
# 			while i < len(nodes) and nodes[i].value == current_value:
# 				if len(same_value_nodes) == config.allowed_divisors_r[0] - 1:
# 					break
# 				same_value_nodes.append(nodes[i])
# 				i += 1

# 			if len(same_value_nodes) > 0:
# 				merged_node = current_node.merge_up(same_value_nodes)
# 				merged_nodes.append(merged_node)
# 				done = False
# 				has_merged = True
# 			else:
# 				merged_nodes.append(current_node)

# 		if done: break

# 		merged_nodes = sort_nodes(merged_nodes)
# 		nodes = [node for node in merged_nodes]
# 	return nodes, has_merged

# def _simplify_extract(nodes):
# 	# Step 2: Extract maximum conveyor speed that fits (ignore nodes with value already equal to a conveyor speed)
# 	extracted_nodes = []
# 	for node in nodes:
# 		extracted_flag = False
# 		for speed in config.conveyor_speeds_r:
# 			if node.value == speed: break
# 			if node.value > speed:
# 				extracted_node, overflow_node = node.extract_up(speed)
# 				extracted_nodes.append(extracted_node)
# 				extracted_nodes.append(overflow_node)
# 				extracted_flag = True
# 				break
# 		if not extracted_flag:
# 			extracted_nodes.append(node)

# 	nodes = sort_nodes(extracted_nodes)
# 	return nodes

# def simplify(nodes):
# 	nodes, has_merged = _simplify_merge(nodes)
# 	nodes = _simplify_extract(nodes)
# 	while has_merged:
# 		nodes, has_merged = _simplify_merge(nodes)
# 		if not has_merged: break
# 		nodes = _simplify_extract(nodes)
# 	return nodes