import math, time, random, itertools, json, os

from utils import get_node_values, get_node_ids, clear_solution_files, parse_user_input, get_compute_cant_use, get_sim_without, remove_pairs, get_divisors, compute_minimum_possible_fraction
from bisect import insort
from config import config
from node import Node
from tree import Tree
# from simsManager import SimsManager
from fastList import FastList
from distance import find_best_merges

class SatisSolver:
	def __init__(self):
		# self.simsManager = None
		# for the SimsManager
		# self.allowed_divisors_r = reversed(sorted(list(config.allowed_divisors)))
		self.reset()
		if config.logging:
			open(config.log_filepath, "w").close()
			self.log_file_handle = open(config.log_filepath, "a", encoding="utf-8")

	def log(self, txt):
		self.log_file_handle.write(txt)
		self.log_file_handle.flush()
		os.fsync(self.log_file_handle.fileno())

	def load(self, user_input):
		try:
			source_values, target_values = parse_user_input(user_input)
		except:
			print("\nTu racontes quoi mon reuf")
			return False
		if not source_values or not target_values: return False
		
		self.reset()
		
		source_values = sorted(source_values)
		self.target_values = sorted(target_values)
		sources_total = sum(source_values)
		targets_total = sum(target_values)
		
		if sources_total > targets_total:
			value = sources_total - targets_total
			insort(self.target_values, value)
			print(f"\nTargets were lacking, generated a {value} node as target")
		
		elif sources_total < targets_total:
			value = targets_total - sources_total
			insort(source_values, value)
			print(f"\nSources were lacking, generated a {value} node as source")

		print()

		source_values_length = len(source_values)
		self.target_values_length = len(self.target_values)
		target_counts = {
			value: self.target_values.count(value) for value in set(self.target_values)
		}
		self.compute_cant_use = get_compute_cant_use(target_counts)
		self.conveyor_speed_limit = config.conveyor_speeds[-1]
		self.minimum_possible_fraction = compute_minimum_possible_fraction(self.target_values)

		print(f"{self.minimum_possible_fraction = }")

		print()

		self.tree_source = Tree([Node(value) for value in source_values])

		return True

	def reset(self):
		self.solving = False
		self.done_solving = False
		self.concluding = False
		self.done_concluding = False
		
		self.solutions = []
		self.solutions_count = 0
		self.best_size = None

	def maximum_value(self, value):
		# all_divisors = [[i for i in get_divisors(t)] for t in self.target_values]
		# r = 0
		# for i in range(self.target_values_length):
		# 	t = self.target_values[i]
		# 	if value <= t:
		# 		r += 1
		# 		continue
		return self.target_values_length

	def extract_sims(self, tree, cant_use, conveyor_speed):
		if self.solutions and tree.size + 2 > self.best_size: return
		
		source_values = tree.source_values
		seen_values = set()
		
		for i, src in enumerate(tree.sources):
			# common to all sims
			
			value = src.value
			if value in cant_use or value <= conveyor_speed: continue
			
			if value in seen_values: continue
			seen_values.add(value)
			
			overflow_value = value - conveyor_speed
			
			# specific to the problem

			values_to_add = [conveyor_speed, overflow_value]
			if any(src.past.contains(value) for value in values_to_add): continue

			sim = get_sim_without(value, source_values)
			for value in values_to_add: insort(sim, value)
			sim, sim_set = tuple(sim), set(sim)

			# if tree.past.contains(sim) or \
			# 	tree.total_seen.get(conveyor_speed, 0) + 1 > self.maximum_value(conveyor_speed) or \
			# 	tree.total_seen.get(overflow_value, 0) + 1 > self.maximum_value(overflow_value):
			# 	continue

			if tree.past.contains(sim): continue

			yield (sim, (i,))

	def divide_loop_sims(self, tree, cant_use):
		if self.solutions and tree.size + 3 > self.best_size: return

		source_values = tree.source_values
		seen_values = set()

		for i, src in enumerate(tree.sources):
			# common to all sims

			value = src.value
			if value in cant_use or value in seen_values: continue
			
			seen_values.add(value)

			conveyor_speed = next((c for c in config.conveyor_speeds if c > value), None)
			if not conveyor_speed: continue
			
			new_value = conveyor_speed / 3
			tmp = new_value * 2
			if value < tmp: continue
			overflow_value = value - tmp

			# specific to the problem
			
			if src.past.contains(new_value) or src.past.contains(overflow_value): continue

			sim = get_sim_without(value, source_values)
			insort(sim, new_value)
			insort(sim, new_value)
			insort(sim, overflow_value)

			sim = tuple(sim)

			# if tree.past.contains(sim) or \
			# 	tree.total_seen.get(new_value, 0) + 2 > self.maximum_value(new_value) or \
			# 	tree.total_seen.get(overflow_value, 0) + 1 > self.maximum_value(overflow_value):
			# 	continue

			if tree.past.contains(sim): continue

			yield (sim, (i, conveyor_speed))

	def divide_sims(self, tree, cant_use, divisor):
		if self.solutions and tree.size + divisor > self.best_size: return
		
		source_values = tree.source_values
		seen_values = set()
		
		for i, src in enumerate(tree.sources):
			# common to all sims
			
			value = src.value
			if value in cant_use or value in seen_values: continue
			seen_values.add(value)

			divided_value = value / divisor
			if divided_value.denominator != 1 and (all(divided_value.denominator != value.denominator for value in self.target_values) or divided_value < self.minimum_possible_fraction): continue

			# specific to the problem
			
			if src.past.contains(divided_value): continue

			sim = get_sim_without(value, source_values)
			for _ in range(divisor): insort(sim, divided_value)
			sim = tuple(sim)

			# if tree.past.contains(sim) or tree.total_seen.get(divided_value, 0) + divisor > self.maximum_value(divided_value): continue
			
			if tree.past.contains(sim): continue

			yield (sim, (i,))

	def merge_sims(self, tree, cant_use, to_sum_count):
		if self.solutions and tree.size + 1 > self.best_size: return
		
		sources = tree.sources
		source_values = tree.source_values
		seen_sums = set()

		for to_sum_indices in itertools.combinations(range(tree.n_sources), to_sum_count):
			# common to all sims
			
			to_sum_indices = list(to_sum_indices)
			to_sum_values = tuple(source_values[i] for i in to_sum_indices)

			if any(value in cant_use for value in to_sum_values): continue

			if to_sum_values in seen_sums: continue
			seen_sums.add(to_sum_values)

			summed_value = sum(to_sum_values)
			to_sum_nodes = [sources[i] for i in to_sum_indices]

			# specific to the problem			
			
			if summed_value > self.conveyor_speed_limit or any(src.past.contains(summed_value) for src in to_sum_nodes): continue

			sim = [value for value in source_values]
			for value in to_sum_values: sim.remove(value)
			insort(sim, summed_value)
			sim = tuple(sim)

			# if tree.past.contains(sim) or tree.total_seen.get(summed_value, 0) + 1 > self.maximum_value(summed_value): continue

			if tree.past.contains(sim): continue

			yield (sim, (to_sum_indices,))

	def compute_distance(self, sources):
		sources = list(sources)
		targets = self.target_values[:]
		distance = 0
		
		# remove common elements
		sources, targets = remove_pairs(sources, targets)

		sources_set = set(sources)
		possible_extractions = [
			(value, speed, overflow)
			for speed in config.conveyor_speeds_r
			for value in sources_set
			if (overflow := value - speed) \
				and value > speed \
				and (speed in targets or overflow in targets)
		]

		# remove perfect extractions
		for i in range(len(possible_extractions)-1, -1, -1):
			value, speed, overflow = possible_extractions[i]
			if value not in sources: continue
			if speed == overflow:
				if len([v for v in targets if v == speed]) < 2: continue
			else:
				if speed not in targets or overflow not in targets: continue
			sources.remove(value)
			targets.remove(speed)
			targets.remove(overflow)
			distance += 1
			possible_extractions.pop(i)

		# remove unperfect extractions
		for value, speed, overflow in possible_extractions:
			if value not in sources: continue
			if speed in targets:
				sources.remove(value)
				targets.remove(speed)
				sources.append(overflow)
				distance += 2
			elif overflow in targets:
				sources.remove(value)
				targets.remove(overflow)
				sources.append(speed)
				distance += 2

		sources_set = set(sources)
		possible_divisions = sorted([
			(value, divisor, divided_value, min(divisor, sum(1 for v in targets if v == divided_value)))
			for divisor in config.allowed_divisors
			for value in sources_set
			if (divided_value := value / divisor) \
				and value % divisor == 0 \
				and divided_value in targets
		], key=lambda x: x[3]-x[1])

		# remove perfect divisions
		for i in range(len(possible_divisions)-1, -1, -1):
			value, divisor, divided_value, divided_values_count = possible_divisions[i]
			if divided_values_count != divisor: break
			if value not in sources or len([v for v in targets if v == divided_value]) < divisor: continue
			sources.remove(value)
			for _ in range(divided_values_count): targets.remove(divided_value)
			possible_divisions.pop(i)
			distance += 1
		
		# remove unperfect divisions
		for i in range(len(possible_divisions)-1, -1, -1):
			value, divisor, divided_value, divided_values_count = possible_divisions[i]
			if value not in sources or len([v for v in targets if v == divided_value]) < divided_values_count: continue
			sources.remove(value)
			for _ in range(divided_values_count): targets.remove(divided_value)
			for _ in range(divisor - divided_values_count): sources.append(divided_value)
			distance += 2

		# remove all possible merges that yield a target, prioritizing the ones that merge the most amount of values in sources
		for to_sum_count in range(config.max_sum_count, config.min_sum_count-1, -1):
			all_combinations = list(itertools.combinations(sources, to_sum_count))
			combinations_count = len(all_combinations)
			all_combinations_sum = [sum(combination) for combination in all_combinations]
			all_merges = [[
				all_combinations[i]
				for i in range(combinations_count)
				if all_combinations_sum[i] == target
			] for target in reversed(sorted(targets))]
			sources_left, targets_left, n_targets_left = find_best_merges(all_merges, sources, targets)
			if sources_left: # and targets_left and n_targets_left
				sources = sources_left
				targets = targets_left
				distance += 1

		return distance + len(sources) + len(targets)

	# computes how close the sources are from the target_values
	# the lower the better
	def compute_tree_score(self, tree):
		# return distance(get_node_values(tree.sources), self.target_values)
		return self.compute_distance(get_node_values(tree.sources))
		# sources = tree.sources
		# simulations = []
		# cant_use = self.compute_cant_use(sources)
		# simulations.extend(self.extract_sims(tree, cant_use))
		# simulations.extend(self.divide_sims(tree, cant_use))
		# simulations.extend(self.merge_sims(tree, cant_use))
		# score = -1
		# # it required at least one operation to get there, hence the 1 +
		# if simulations: score = 1 + min(self.simsManager.compute_distance(sim, self.target_values) for sim, _ in simulations)
		# return score

	def is_solution(self, sources):
		# assume the given sources are sorted by value
		n = len(sources)
		if n != self.target_values_length: return False
		for i in range(n):
			if sources[i].value != self.target_values[i]:
				return False
		return True

	def solution_found(self, tree):
		# return if found better size
		if self.solutions_count == 0 or tree.size < self.best_size:
			self.solutions = [tree]
			self.best_size = tree.size
			self.solutions_count = 1
			optional_s_txt = "s" if self.solutions_count > 1 else ""
			print("\r" + " " * 100 + f"\rFound {self.solutions_count} solution{optional_s_txt} of size {self.best_size}", end="")
			return True
		elif tree.size == self.best_size:
			self.solutions.append(tree)
			self.solutions_count += 1
			optional_s_txt = "s" if self.solutions_count > 1 else ""
			print("\r" + " " * 100 + f"\rFound {self.solutions_count} solution{optional_s_txt} of size {self.best_size}", end="")
			return False
		print("impossible case reached, should have been checked already")
		self.solving = False

	# def build_optimal_solutions(self):
		# for i in range(self.solutions_count-1, -1, -1):
		# 	sol_tree = self.solutions[i]
		# 	current_sol_size = self.target_values_length
		# 	for j in range(len(sol_past)-1, -1, -1):
		# 		sp_sources = sol_past[j]
		# 		sp_source_values = get_node_values(sp_sources)
		# 		for ct_root, ct_sources, ct_past in cutted_trees:
		# 			ct_source_values = get_node_values(ct_sources)
		# 			if sp_source_values == ct_source_values and ct_root.size + current_sol_size < sol_root.size:
		# 				# we found the smallest cutted tree that can shorten the current solution

		# 				break
		# 		current_sol_size += len(sp_sources)
		# print(f"{len(self.cutted_trees) = }")

	def solve(self):
		queue = []

		def purge_queue():
			nonlocal queue
			for i in range(len(queue) - 1, -1, -1):
				if not self.solving: break
				tree, _ = queue[i]
				if tree.size >= self.best_size: queue.pop(i)

		def enqueue(tree):
			nonlocal queue
			if self.is_solution(tree.sources):
				if self.solution_found(tree): purge_queue()
				return
			score = self.compute_tree_score(tree)
			if score < 0: return
			insort(queue, (tree, score), key=lambda x: (-x[1], -x[0].size))
		
		def dequeue():
			nonlocal queue
			n = len(queue)
			if n < 3: return queue.pop()
			# favor exploration as the number of solutions grows by 5% per solution, with a maximum of 70% exploration
			# maximum exploration is reached at (70 - 30) / 5 = 8 solutions found
			# 30% exploration when no solution is found
			# by exploration I mean trees with lower scores
			exploration_prob = max(0.3, 0.70 - (5 * self.solutions_count / 100))
			return queue.pop(-1 if random.random() < exploration_prob else random.randrange(n-1))

		enqueue(self.tree_source)
		# self.processed_sources.add(self.tree_source.source_values)

		# max_size = 0

		while self.solving and queue:
			tree, _ = dequeue()
			# if tree.n_sources > max_size:
			# 	max_size = tree.n_sources
			# 	print(max_size)
			cant_use = self.compute_cant_use(tree.sources)

			def try_op(get_sims, op, get_sims_args=tuple([])):
				for _, sim_metadata in get_sims(tree, cant_use, *get_sims_args):
					if not self.solving: break
					# if sim in self.processed_sources:
					# 	insort(self.cutted_trees, tree, lambda cutted_tree: cutted_tree.size)
					# 	continue
					# self.processed_sources.add(sim)
					# tree_copy = copy.deepcopy(tree)
					tree_copy = tree.deepcopy()
					log_msg, result_nodes = op(tree_copy.sources, sim_metadata, *get_sims_args)
					tree_copy.add(result_nodes)
					if config.logging: self.log(f"\n\nFROM\n{tree}\nDID\n{log_msg}")
					enqueue(tree_copy)

			def extract(sources_copy, sim_metadata, conveyor_speed):
				i, *_ = sim_metadata
				src_copy = sources_copy[i]
				return f"{src_copy} - {conveyor_speed}" if config.logging else None, src_copy.extract(conveyor_speed)

			def divide(sources_copy, sim_metadata, divisor):
				i, *_ = sim_metadata
				src_copy = sources_copy[i]
				return f"{src_copy} / {divisor}" if config.logging else None, src_copy.divide(divisor)
			
			def divide_loop(sources_copy, sim_metadata):
				i, conveyor_speed = sim_metadata
				src_copy = sources_copy[i]
				return f"{src_copy} /loop {conveyor_speed}" if config.logging else None, src_copy.divide_loop(conveyor_speed)

			def merge(sources_copy, sim_metadata, to_sum_count):
				to_sum_indices, *_ = sim_metadata
				to_sum_nodes = [sources_copy[i] for i in to_sum_indices]
				return "\n+\n".join(str(ts) for ts in to_sum_nodes) if config.logging else None, [Node.merge(to_sum_nodes)]

			for conveyor_speed in config.conveyor_speeds:
				try_op(self.extract_sims, extract, (conveyor_speed,))
				if not self.solving: break
			
			if not self.solving: break
			
			for divisor in config.allowed_divisors:
				try_op(self.divide_sims, divide, (divisor,))
				if not self.solving: break
			
			if not self.solving: break
			
			try_op(self.divide_loop_sims, divide_loop)
			
			if not self.solving: break

			for to_sum_count in range(config.min_sum_count, config.max_sum_count + 1):
				try_op(self.merge_sims, merge, (to_sum_count,))
				if not self.solving: break

		# self.build_optimal_solutions()

	def conclude(self):
		if not self.solutions:
			print("No bitches?")
			return
		clear_solution_files()
		print()
		for i, tree in enumerate(self.solutions):
			if not self.concluding: break
			tree.visualize(config.solutions_filename(i))

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
		# if self.simsManager: self.simsManager.save_cache()
		if config.logging: self.log_file_handle.close()

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
# 		for speed in config.conveyor_speeds:
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