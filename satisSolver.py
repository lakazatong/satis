import math, time, random, copy, itertools, json, os
from contextlib import redirect_stdout

from utils import sort_nodes, get_node_values, get_node_ids, pop_node, insert_into_sorted, clear_solution_files, parse_user_input, get_gcd_incompatible, get_compute_cant_use
from config import config
from node import Node
from tree import Tree
from simsManager import SimsManager
from fastList import FastList

class SatisSolver:
	def __init__(self):
		self.simsManager = None
		# for the SimsManager
		self.allowed_divisors_r = reversed(sorted(list(config.allowed_divisors)))
		self.reset()

	def log(self, *args, **kwargs):
		with redirect_stdout(self.log_file_handle):
			print(*args, **kwargs)

	def load(self, user_input):
		self.user_input = user_input
		source_values, target_values = parse_user_input(user_input)
		if not source_values or not target_values: return False
		
		self.reset()
		
		self.source_values = sorted(source_values)
		self.target_values = sorted(target_values)
		sources_total = sum(source_values)
		targets_total = sum(target_values)
		
		if sources_total > targets_total:
			value = sources_total - targets_total
			insert_into_sorted(self.target_values, value)
			print(f"\nTargets were lacking, generated a {value} node as target")
		
		elif sources_total < targets_total:
			value = targets_total - sources_total
			insert_into_sorted(self.source_values, value)
			print(f"\nSources were lacking, generated a {value} node as source")

		self.source_values_length = len(self.source_values)
		self.target_values_length = len(self.target_values)
		target_counts = {
			value: self.target_values.count(value) for value in set(self.target_values)
		}
		self.compute_cant_use = get_compute_cant_use(target_counts)
		gcd = math.gcd(*self.source_values, *self.target_values)
		if not self.simsManager:
			self.simsManager = SimsManager(self)
			# self.simsManager.load_cache()
		self.gcd_incompatible = get_gcd_incompatible(gcd)
		# just to show, extract does "gcd_incompatible(speed)" to check instead of doing "speed in self.filtered_conveyor_speeds"
		filtered_conveyor_speeds = sorted([speed for speed in config.conveyor_speeds if not self.gcd_incompatible(speed)])
		self.conveyor_speed_limit = max(filtered_conveyor_speeds)
		# for the SimsManager
		self.filtered_conveyor_speeds_r = reversed(sorted(speed for speed in config.conveyor_speeds if not self.gcd_incompatible(speed)))

		print(f"\ngcd: {gcd}\nfiltered conveyor speeds: {", ".join(map(str, filtered_conveyor_speeds))}\n")
		
		self.tree_source = Tree([Node(value) for value in self.source_values])

		return True

	def reset(self):
		self.solving = False
		self.done_solving = False
		self.concluding = False
		self.done_concluding = False
		
		self.solutions = []
		self.solutions_count = 0
		self.best_size = None

		self.cutted_trees = []
		self.seen_sources = {}
		self.compute_distances_cache = {}

		if config.logging:
			open(config.log_filepath, "w").close()
			self.log_file_handle = open(config.log_filepath, "a", encoding="utf-8")

	def extract_sims(self, tree, cant_use):
		sources = tree.sources
		if self.solutions and tree.size() + 2 > self.best_size: return []
		filtered_simulations = []
		for info in self.simsManager.get_divide_sims(tree):
			sim, (i, speed) = info
			src = sources[i]
			if src.value in cant_use: continue
			if self.gcd_incompatible(speed): continue
			overflow = src.value - speed
			if self.gcd_incompatible(overflow): continue
			if tree.past.contains(sim): continue
			parent_values = set(get_node_values(src.parents))
			# if speed in parent_values then it would have been better to leave it as is
			# and merge all the other values to get the overflow value
			# we would get by exctracting speed amount
			# same logic applies if overflow is in parent values
			if speed in parent_values or overflow in parent_values: continue
			filtered_simulations.append(info)
		return filtered_simulations

	def divide_sims(self, tree, cant_use):
		sources = tree.sources
		simulations = self.simsManager.get_divide_sims(tree)
		if self.solutions:
			if not simulations: return []
			size = tree.size()
			# we need to filter here because the new size is unknown whereas for extract and merge
			# they are a constant +2 or +1
			# sim[1][1] is divisor aka the number of nodes added
			return list(filter(lambda sim: size + sim[1][1] <= self.best_size, simulations))
		filtered_simulations = []
		parents_value_sum, n_parents = None, None
		for info in simulations:
			sim, (i, divisor) = info
			src = sources[i]
			if src.value in cant_use: continue
			divided_value = src.value // divisor
			if self.gcd_incompatible(divided_value): continue
			if tree.past.contains(sim): continue
			if not parents_value_sum:
				parents_value_sum = sum(get_node_values(src.parents))
				n_parents = len(src.parents)
			if parents_value_sum == src.value and n_parents == divisor: continue
			filtered_simulations.append(info)
		return filtered_simulations

	def merge_sims(self, tree, cant_use):
		sources = tree.sources
		if self.solutions and tree.size() + 1 > self.best_size: return []
		filtered_simulations = []
		for info in self.simsManager.get_merge_sims(tree):
			sim, (to_sum_indices, to_sum_count) = info
			to_sum_nodes = [sources[i] for i in to_sum_indices]
			if any(node.value in cant_use for node in to_sum_nodes): continue
			summed_value = sum(node.value for node in to_sum_nodes)
			if self.gcd_incompatible(summed_value) or summed_value > self.conveyor_speed_limit: continue
			if all(len(node.parents) == 1 for node in to_sum_nodes):
				parent = to_sum_nodes[0].parents[0]
				if all(node.parents[0] is parent for node in to_sum_nodes) and len(parent.children) == to_sum_count and (parent.parents or self.source_values_length == 1): continue
			if tree.past.contains(sim): continue
			filtered_simulations.append(info)
		return filtered_simulations

	# computes how close the sources are from the target_values
	# the lower the better
	def compute_tree_score(self, tree):
		sources = tree.sources
		simulations = []
		cant_use = self.compute_cant_use(sources)
		simulations.extend(self.extract_sims(tree, cant_use))
		simulations.extend(self.divide_sims(tree, cant_use))
		simulations.extend(self.merge_sims(tree, cant_use))
		score = -1
		# it required at least one operation to get there, hence the 1 +
		if simulations: score = 1 + min(self.simsManager.compute_distance(sim, self.target_values) for sim, _ in simulations)
		return score

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
		if self.solutions_count == 0 or tree.size() < self.best_size:
			self.solutions = [tree]
			self.best_size = tree.size()
			self.solutions_count = 1
			print(" " * 10 + f"\rFound {self.solutions_count} solutions of size {self.best_size}", end="")
			return True
		elif tree.size() == self.best_size:
			self.solutions.append(tree)
			self.solutions_count += 1
			print(" " * 10 + f"\rFound {self.solutions_count} solutions of size {self.best_size}", end="")
			return False
		print("impossible case reached, should have been checked already")
		self.solving = False

	def build_optimal_solutions(self):
		for i in range(self.solutions_count-1, -1, -1):
			sol_tree = self.solutions[i]
			current_sol_size = self.target_values_length
			for j in range(len(sol_past)-1, -1, -1):
				sp_sources = sol_past[j]
				sp_source_values = get_node_values(sp_sources)
				for ct_root, ct_sources, ct_past in cutted_trees:
					ct_source_values = get_node_values(ct_sources)
					if sp_source_values == ct_source_values and ct_root.size + current_sol_size < sol_root.size:
						# we found the smallest cutted tree that can shorten the current solution

						break
				current_sol_size += len(sp_sources)

	def solve(self):
		queue = []

		def purge_queue():
			nonlocal queue
			for i in range(len(queue) - 1, -1, -1):
				if not self.solving: break
				tree, _ = queue[i]
				if tree.size() >= self.best_size: queue.pop(i)

		def enqueue(tree):
			nonlocal queue
			if self.is_solution(tree.sources):
				if self.solution_found(tree): purge_queue()
				return
			score = self.compute_tree_score(tree)
			if score < 0: return
			insert_into_sorted(queue, (tree, score), key=lambda x: x[1])

		def dequeue():
			nonlocal queue
			n = len(queue)
			if n < 3: return queue.pop(0)
			return queue.pop(0 if random.random() < 0.8 else random.randrange(1, n))

		enqueue(self.tree_source)

		while self.solving and queue:
			tree, _ = dequeue()
			sources = tree.sources
			# source_values = tree.source_values

			# if source_values in self.seen_sources:
			# 	insert_into_sorted(self.cutted_trees, tree, lambda cutted_tree: cutted_tree.size())
			# 	continue
			
			# seen_sources.add(source_values)
			cant_use = self.compute_cant_use(sources)

			def try_op(get_sims, op):
				for _, sim_metadata in get_sims(tree, cant_use):
					if not self.solving: break
					tree_copy = copy.deepcopy(tree)
					log_msg, result_nodes = op(tree_copy.sources, sim_metadata)
					tree_copy.add(result_nodes)
					if config.logging: self.log(f"\n\nFROM\n{tree}\nDID\n{log_msg}")
					enqueue(tree_copy)

			def extract(sources_copy, sim_metadata):
				i, speed = sim_metadata
				src_copy = sources_copy[i]
				return f"{src_copy} - {speed}", src_copy - speed

			def divide(sources_copy, sim_metadata):
				i, divisor = sim_metadata
				src_copy = sources_copy[i]
				return f"{src_copy} / {divisor}", src_copy / divisor

			def merge(sources_copy, sim_metadata):
				to_sum_indices, _ = sim_metadata
				to_sum = [sources_copy[i] for i in to_sum_indices]
				return "\n+\n".join(str(ts) for ts in to_sum), [to_sum[0] + to_sum[1:]]

			try_op(self.extract_sims, extract)
			if not self.solving: break
			try_op(self.divide_sims, divide)
			if not self.solving: break
			try_op(self.merge_sims, merge)

		# self.build_optimal_solutions()

	def conclude(self):
		if not self.solutions: return
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