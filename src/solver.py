import os

from utils.solver import get_sim_without
from utils.text import print_standing_text

from cost import extract_cost, divide_cost, merge_cost, split_cost
from bisect import insort
from config import config
from node import Node

class SatisSolver:
	def __init__(self):
		self.reset()
		if config.logging: self.log_file_handle = open(config.log_filepath, "a", encoding="utf-8")

	def log(self, txt):
		self.log_file_handle.write(txt)
		self.log_file_handle.flush()
		os.fsync(self.log_file_handle.fileno())

	def load(self, user_input):
		from utils.solver import parse_user_input
		import traceback
		try:
			source_values, target_values = parse_user_input(user_input)
		except:
			print("\nTu racontes quoi mon reuf")
			print(traceback.format_exc(), end="")
			return False
		if not source_values or not target_values: return False

		from score import ScoreCalculator
		from tree import Tree
		from utils.fractions import format_fractions, fractions_to_integers
		from utils.solver import get_compute_cant_use, get_gcd_incompatible
		from utils.math import compute_gcd
		from utils.other import remove_pairs

		self.reset()

		source_values = sorted(source_values)
		target_values = sorted(target_values)
		sources_total = sum(source_values)
		targets_total = sum(target_values)
		
		if sources_total < targets_total:
			value = targets_total - sources_total
			insort(source_values, value)
			print(f"\nSources were lacking, generated a {value} node as source")
		elif sources_total > targets_total:
			value = sources_total - targets_total
			insort(target_values, value)
			print(f"\nTargets were lacking, generated a {value} node as target")

		source_values, target_values = remove_pairs(source_values, target_values)

		if not source_values:
			assert not target_values
			# no problem to solve
			return False

		self.problem_str = format_fractions(source_values) + " to " + format_fractions(target_values)

		r, self.unit_flow_ratio = fractions_to_integers(source_values + target_values)
		
		n_sources = len(source_values)

		source_values = r[:n_sources]
		self.target_values = r[n_sources:]
		
		self.leaves, self.size_offset = Node.group_values(self.target_values)
		self.target_values = Node.get_node_values(self.leaves)

		self.n_targets = len(self.target_values)

		self.scoreCalculator = ScoreCalculator(self.target_values, self)
		
		self.min_target = min(self.target_values)
		target_counts = {
			value: self.target_values.count(value) for value in set(self.target_values)
		}
		self.compute_cant_use = get_compute_cant_use(target_counts)
		self.conveyor_speed_limit = config.conveyor_speeds[-1]

		self.tree_source = Tree([Node(value) for value in source_values])

		self.best_size = self.best_size_upper_bond() + self.size_offset
		gcd = compute_gcd(*source_values, *self.target_values)
		self.gcd_incompatible = get_gcd_incompatible(gcd)
		print(f"\nSolutions' size upper bound: {self.best_size}, {gcd = }\n")

		return True

	def best_size_upper_bond(self):
		from utils import remove_pairs
		sources = self.tree_source.source_values
		r = 0
		r = merge_cost(len(sources), 1)
		summed_value = sum(sources)
		for i in range(self.n_targets):
			target = self.target_values[i]
			r += extract_cost(summed_value, target)
			summed_value -= target
		return r

	def reset(self):
		self.solving = False
		self.done_solving = False
		self.concluding = False
		self.done_concluding = False
		
		self.solutions = []
		self.solutions_count = 0

		self.score_cache = {}

		if config.logging:
			open(config.log_filepath, "w").close()

	def extract_sims(self, tree, cant_use):
		source_values = tree.source_values
		seen_extractions = set()
		for i, src in enumerate(tree.sources):
			for conveyor_speed in range(1, (src.value - 1) // 2 + 1):
				if not self.solving: return

				value = src.value

				if (value, conveyor_speed) in seen_extractions: continue
				seen_extractions.add((value, conveyor_speed))

				if value in cant_use: continue
				
				if self.gcd_incompatible(conveyor_speed): continue

				cost = extract_cost(value, conveyor_speed)

				if tree.size + cost > self.best_size: continue
				
				overflow_value = value - conveyor_speed

				if overflow_value == conveyor_speed: continue
				
				if self.gcd_incompatible(overflow_value): continue
				values_to_add = [conveyor_speed, overflow_value]
				if any(src.past.contains(value) for value in values_to_add): continue

				sim = get_sim_without(value, source_values)
				for value in values_to_add: insort(sim, value)
				sim, sim_set = tuple(sim), set(sim)

				if tree.past.contains(sim) or any(t.past.contains(sim) for t in self.solutions): continue

				yield (sim, (i, conveyor_speed), cost)

	def divide_sims(self, tree, cant_use):
		source_values = tree.source_values
		seen_divisions = set()
		for i, src in enumerate(tree.sources):
			for divisor in (d for d in range(2, src.value + 1) if src.value % d == 0):
				if not self.solving: return
				
				value = src.value
				
				if (value, divisor) in seen_divisions: continue
				seen_divisions.add((value, divisor))

				if value in cant_use: continue

				cost = divide_cost(value, divisor)
				if tree.size + cost > self.best_size: continue

				divided_value = value // divisor

				if src.past.contains(divided_value) or self.gcd_incompatible(divided_value): continue

				sim = get_sim_without(value, source_values)
				for _ in range(divisor): insort(sim, divided_value)
				sim = tuple(sim)

				if tree.past.contains(sim) or any(t.past.contains(sim) for t in self.solutions): continue

				yield (sim, (i, divisor), cost)

	def merge_sims(self, tree, cant_use):
		import itertools

		sources = tree.sources
		source_values = tree.source_values

		for to_sum_count in range(2, tree.n_sources+1):
			cost = merge_cost(to_sum_count, 1)
			
			if tree.size + cost > self.best_size: continue
			
			seen_sums = set()
			
			for to_sum_indices in itertools.combinations(range(tree.n_sources), to_sum_count):
				if not self.solving: return
				
				to_sum_indices = list(to_sum_indices)
				to_sum_values = tuple(source_values[i] for i in to_sum_indices)

				if to_sum_values in seen_sums: continue
				seen_sums.add(to_sum_values)

				if any(value in cant_use for value in to_sum_values): continue

				summed_value = sum(to_sum_values)

				if self.gcd_incompatible(summed_value): continue
				to_sum_nodes = [sources[i] for i in to_sum_indices]
				if summed_value > self.conveyor_speed_limit or any(src.past.contains(summed_value) for src in to_sum_nodes): continue

				sim = [value for value in source_values]
				for value in to_sum_values: sim.remove(value)
				insort(sim, summed_value)
				sim = tuple(sim)

				if tree.past.contains(sim) or any(t.past.contains(sim) for t in self.solutions): continue

				yield (sim, (to_sum_indices, to_sum_count), cost)

	def split_sims(self, tree, cant_use):
		cost = split_cost()
		if tree.size + cost > self.best_size: return

		source_values = tree.source_values
		seen_values = set()

		for i, src in enumerate(tree.sources):
			if not self.solving: return

			value = src.value
			if value in seen_values: continue
			seen_values.add(value)

			if value in cant_use: continue

			conveyor_speed = next((c for c in config.conveyor_speeds if c > value), None)
			if not conveyor_speed: continue
			
			# all are from the game and all are divisible by 3
			new_value = conveyor_speed // 3
			tmp = new_value << 1
			if value < tmp: continue
			overflow_value = value - tmp

			if self.gcd_incompatible(new_value) or self.gcd_incompatible(overflow_value): continue
			if src.past.contains(new_value) or src.past.contains(overflow_value): continue

			sim = get_sim_without(value, source_values)
			insort(sim, new_value)
			insort(sim, new_value)
			insort(sim, overflow_value)

			sim = tuple(sim)

			if tree.past.contains(sim) or any(t.past.contains(sim) for t in self.solutions): continue

			yield (sim, (i, conveyor_speed), cost)

	def is_solution(self, sources):
		# assume the given sources are sorted by value
		n = len(sources)
		if n != self.n_targets: return False
		for i in range(n):
			if sources[i].value != self.target_values[i]:
				return False
		return True

	def solve(self):
		queue = []

		def purge_queue():
			nonlocal queue
			for i in range(len(queue) - 1, -1, -1):
				if not self.solving: return
				tree, _ = queue[i]
				if tree.size >= self.best_size: queue.pop(i)

		def solution_found(tree):
			# return if found better size
			if self.solutions_count == 0 or tree.size < self.best_size:
				self.solutions = [tree]
				self.best_size = tree.size
				self.solutions_count = 1
				optional_s_txt = "s" if self.solutions_count > 1 else ""
				print_standing_text(f"Found {self.solutions_count} solution{optional_s_txt} of size <= {self.best_size + self.size_offset}")
				return True
			elif tree.size == self.best_size:
				if any(t == tree for t in self.solutions): return False
				self.solutions.append(tree)
				self.solutions_count += 1
				optional_s_txt = "s" if self.solutions_count > 1 else ""
				print_standing_text(f"Found {self.solutions_count} solution{optional_s_txt} of size <= {self.best_size + self.size_offset}")
				return False
			print("impossible case reached, should have been checked already")
			self.stop()

		def enqueue(tree):
			nonlocal queue
			if not self.solving: return
			if self.is_solution(tree.sources):
				if solution_found(tree): purge_queue()
				return
			score = 0
			# if self.solutions_count == 0:
			score = self.score_cache.get(tree.source_values, None) or \
				self.scoreCalculator.compute(tree.source_values)
			insort(queue, (tree, score), key=lambda x: (x[1], -x[0].size))
		
		def dequeue():
			nonlocal queue
			if not self.solving: return
			n = len(queue)
			if n < 3 or self.solutions_count > 0: return queue.pop()
			import random
			# favor exploration as the number of solutions grows by 5% per solution, with a maximum of 70% exploration
			# maximum exploration is reached at (70 - 30) / 5 = 8 solutions found
			# 30% exploration when no solution is found
			# by exploration I mean trees with higher distances to targets
			exploration_prob = max(0.3, 0.70 - (5 * self.solutions_count / 100))
			return queue.pop(-1 if random.random() < exploration_prob else random.randrange(n-1))

		enqueue(self.tree_source)

		while self.solving and queue:
			tree, _ = dequeue()

			# tree.quick_solve(self.target_values)

			cant_use = self.compute_cant_use(tree.sources)

			def try_op(get_sims, op):
				for _, sim_metadata, cost in get_sims(tree, cant_use):
					if not self.solving: return
					tree_copy = tree.deepcopy()
					log_msg, result_nodes = op(tree_copy.sources, *sim_metadata)
					tree_copy.add(result_nodes, cost)
					if config.logging: self.log(f"\n\nFROM\n{tree.pretty()}\nDID\n{log_msg}")
					enqueue(tree_copy)

			def extract(sources_copy, i, conveyor_speed):
				src_copy = sources_copy[i]
				return f"{src_copy} - {conveyor_speed}" if config.logging else None, src_copy.extract(conveyor_speed)

			def divide(sources_copy, i, divisor):
				src_copy = sources_copy[i]
				return f"{src_copy} / {divisor}" if config.logging else None, src_copy.divide(divisor)
			
			def split(sources_copy, i, conveyor_speed):
				src_copy = sources_copy[i]
				return f"{src_copy} // {conveyor_speed}" if config.logging else None, src_copy.split(conveyor_speed)

			def merge(sources_copy, to_sum_indices, to_sum_count):
				to_sum_nodes = [sources_copy[i] for i in to_sum_indices]
				return "\n+\n".join(ts.pretty() for ts in to_sum_nodes) if config.logging else None, [Node.merge(to_sum_nodes)]

			if max(tree.source_values) <= self.min_target:
				try_op(self.merge_sims, merge)
				if not self.solving: return
			else:
				try_op(self.extract_sims, extract)
				if not self.solving: return
				try_op(self.divide_sims, divide)
				if not self.solving: return
				try_op(self.split_sims, split)
				if not self.solving: return
				try_op(self.merge_sims, merge)
				if not self.solving: return

	def clear_solution_files(self):
		for filename in os.listdir(self.problem_str):
			if config.solution_regex.match(filename):
				os.remove(os.path.join(self.problem_str, filename))

	def conclude(self):
		print()
		print()
		if not self.solutions:
			print("No bitches?\n")
			return
		if os.path.isdir(self.problem_str):
			self.clear_solution_files()
		else:
			os.makedirs(self.problem_str)
		if self.solutions_count > 1:
			for i, tree in enumerate(self.solutions):
				if not self.concluding: break
				print_standing_text(f"Saving solutions... {i+1}/{self.solutions_count}")
				tree.attach_leaves(self.leaves)
				tree.save(os.path.join(self.problem_str, config.solutions_filename(i)), self.unit_flow_ratio)
		else:
			print("Saving solution...")
			tree = self.solutions[0]
			tree.attach_leaves(self.leaves)
			tree.save(os.path.join(self.problem_str, config.solutions_filename(0)), self.unit_flow_ratio)
		print()

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

	# def maximum_value(self, value):
	# 	# all_divisors = [[i for i in get_divisors(t)] for t in self.target_values]
	# 	# r = 0
	# 	# for i in range(self.n_targets):
	# 	# 	t = self.target_values[i]
	# 	# 	if value <= t:
	# 	# 		r += 1
	# 	# 		continue
	# 	return self.n_targets

	# def build_optimal_solutions(self):
		# for i in range(self.solutions_count-1, -1, -1):
		# 	sol_tree = self.solutions[i]
		# 	current_sol_size = self.n_targets
		# 	for j in range(len(sol_past)-1, -1, -1):
		# 		sp_sources = sol_past[j]
		# 		sp_source_values = Node.get_node_values(sp_sources)
		# 		for ct_root, ct_sources, ct_past in cutted_trees:
		# 			ct_source_values = Node.get_node_values(ct_sources)
		# 			if sp_source_values == ct_source_values and ct_root.size + current_sol_size < sol_root.size:
		# 				# we found the smallest cutted tree that can shorten the current solution

		# 				break
		# 		current_sol_size += len(sp_sources)
		# print(f"{len(self.cutted_trees) = }")

# def compute_distance(self, sources):
# 	sources = list(sources)
# 	targets = self.target_values[:]
# 	distance = 0

# 	# remove common elements
# 	sources, targets = remove_pairs(sources, targets)

# 	sources_set = set(sources)
# 	possible_extractions = [
# 		(value, speed, overflow)
# 		for speed in config.allowed_extractors_r
# 		for value in sources_set
# 		if (overflow := Fraction(value - speed, 1)) \
# 			and value > speed and value.denominator == 1 and value.numerator % speed == 0 \
# 			and (speed in targets or overflow in targets)
# 	]

# 	# remove perfect extractions
# 	for i in range(len(possible_extractions)-1, -1, -1):
# 		value, speed, overflow = possible_extractions[i]
# 		if value not in sources: continue
# 		if speed == overflow:
# 			if len([v for v in targets if v == speed]) < 2: continue
# 		else:
# 			if speed not in targets or overflow not in targets: continue
# 		sources.remove(value)
# 		targets.remove(speed)
# 		targets.remove(overflow)
# 		distance += 1
# 		possible_extractions.pop(i)

# 	# remove unperfect extractions
# 	for value, speed, overflow in possible_extractions:
# 		if value not in sources: continue
# 		if speed in targets:
# 			sources.remove(value)
# 			targets.remove(speed)
# 			sources.append(overflow)
# 			distance += 2
# 		elif overflow in targets:
# 			sources.remove(value)
# 			targets.remove(overflow)
# 			sources.append(speed)
# 			distance += 2

# 	sources_set = set(sources)
# 	possible_divisions = sorted([
# 		(value, divisor, divided_value, min(divisor, sum(1 for v in targets if v == divided_value)))
# 		for divisor in config.allowed_divisors
# 		for value in sources_set
# 		if (divided_value := Fraction(value, divisor)) \
# 			and divided_value in targets \
# 			and divided_value.denominator == 1 or (any(divided_value.denominator == value.denominator for value in self.target_values) and divided_value >= self.minimum_possible_fraction)
# 	], key=lambda x: x[3]-x[1])

# 	# remove perfect divisions
# 	for i in range(len(possible_divisions)-1, -1, -1):
# 		value, divisor, divided_value, divided_values_count = possible_divisions[i]
# 		if divided_values_count != divisor: break
# 		if value not in sources or len([v for v in targets if v == divided_value]) < divisor: continue
# 		sources.remove(value)
# 		for _ in range(divided_values_count): targets.remove(divided_value)
# 		possible_divisions.pop(i)
# 		distance += 1

# 	# remove unperfect divisions
# 	for i in range(len(possible_divisions)-1, -1, -1):
# 		value, divisor, divided_value, divided_values_count = possible_divisions[i]
# 		if value not in sources or len([v for v in targets if v == divided_value]) < divided_values_count: continue
# 		sources.remove(value)
# 		for _ in range(divided_values_count): targets.remove(divided_value)
# 		for _ in range(divisor - divided_values_count): sources.append(divided_value)
# 		distance += 2

# 	# remove all possible merges that yield a target, prioritizing the ones that merge the most amount of values in sources
# 	for to_sum_count in range(config.max_sum_count, config.min_sum_count-1, -1):
# 		all_combinations = list(itertools.combinations(sources, to_sum_count))
# 		combinations_count = len(all_combinations)
# 		all_combinations_sum = [sum(combination) for combination in all_combinations]
# 		all_merges = [[
# 			all_combinations[i]
# 			for i in range(combinations_count)
# 			if all_combinations_sum[i] == target
# 		] for target in reversed(sorted(targets))]
# 		sources_left, targets_left, n_targets_left = find_best_merges(all_merges, sources, targets)
# 		if sources_left: # and targets_left and n_targets_left
# 			sources = sources_left
# 			targets = targets_left
# 			distance += 1

# 	return distance + len(sources) + len(targets)

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
# 		for speed in config.allowed_extractors_r:
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