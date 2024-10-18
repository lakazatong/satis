import os, sys, re, math, time, signal, pathlib, threading, io, cProfile, random, copy, traceback, itertools
from contextlib import redirect_stdout
from itertools import combinations

from utils import remove_pairs, sort_nodes, get_node_values, get_node_ids, pop_node, insert_into_sorted
from config import allowed_divisors, conveyor_speeds, logging, log_filename, solutions_filename, solution_regex
from node import Node

if sys.platform == 'win32':
	path = pathlib.Path(r'C:\Program Files\Graphviz\bin')
	if path.is_dir() and str(path) not in os.environ['PATH']:
		os.environ['PATH'] += f';{path}'

concluding = False
stop_concluding = False
solving = False
stop_solving = False
allowed_divisors_r = allowed_divisors[::-1]
min_sum_count, max_sum_count = allowed_divisors[0], allowed_divisors_r[0]
conveyor_speeds_r = conveyor_speeds[::-1]
conveyor_speed_limit = conveyor_speeds_r[0]
solutions = []
solutions_count = 0
best_size = None
enqueued_sources = set()
trim_root = False
if logging: open(log_filename, "w").close()

def log(*args, **kwargs):
	with open(log_filename, "a", encoding="utf-8") as f:
		with redirect_stdout(f):
			print(*args, **kwargs)

def clear_solution_files():
	for filename in os.listdir('.'):
		if solution_regex.match(filename):
			os.remove(filename)

def conclude():
	global concluding, stop_concluding
	if concluding or stop_concluding: return
	concluding = True
	stop_concluding = False
	if solutions:
		clear_solution_files()
		print()
		for i in range(len(solutions)):
			if stop_concluding: break
			solution = solutions[i]
			if trim_root:
				for child in solution.children:
					child.compute_levels()
			else:
				solution.compute_levels()
			solution.visualize(solutions_filename(i), trim_root)
	concluding = False

def get_sim_without(sources, value):
	sim = []
	found = False
	for src in sources:
		if src.value == value and not found:
			found = True
		else:
			sim.append(src.value)
	return sim

def solution_found(new_solution_root):
	# return if found better size
	global solutions, best_size, solutions_count
	# global solutions, best_size
	# print(new_solution_root)
	if len(solutions) == 0 or new_solution_root.size < best_size:
		solutions = [new_solution_root]
		best_size = new_solution_root.size
		solutions_count = 1
		# print(f"\n^^^ New solution of size {best_size} found ^^^\n")
		print(" " * 10 + f"\rFound {solutions_count} solutions of size {best_size}", end="")
		return True
	elif new_solution_root.size == best_size:
		solutions.append(new_solution_root)
		solutions_count += 1
		# print(f"\n^^^ Another solution of size {best_size} found ^^^\n")
		print(" " * 10 + f"\rFound {solutions_count} solutions of size {best_size}", end="")
		return False
	print("impossible case reached, should have been checked already")
	exit(1)

def _solve(source_values, target_values, starting_node_sources=None):
	global solving, enqueued_sources
	print(f"\nsolving: {sorted(source_values)} to {sorted(target_values)}\n")

	target_values = sorted(target_values)
	source_values_length = len(source_values)
	target_values_length = len(target_values)
	target_counts = {
		value: target_values.count(value) for value in set(target_values)
	}
	gcd = math.gcd(*source_values, *target_values)
	# node_targets = list(map(lambda value: Node(value), target_values))
	# print('\n'.join([str(src) for src in copy]))

	def gcd_incompatible(value):
		return value < gcd

	filtered_conveyor_speeds = [speed for speed in conveyor_speeds if not gcd_incompatible(speed)]
	filtered_conveyor_speeds_r = filtered_conveyor_speeds[::-1]
	print(f"gcd = {gcd}, filtered conveyor speeds = {filtered_conveyor_speeds}", end="\n"*2)

	node_sources = None
	if starting_node_sources:
		node_sources = starting_node_sources
	else:
		node_sources = list(map(lambda value: Node(value), source_values))
		if source_values_length > 1:
			root = Node(sum(source_values))
			root.children = node_sources
			for child in root.children:
				child.parents.append(root)

	queue = []

	def get_extract_sim(sources, i, past):
	# def get_extract_sim(sources, i):
		global best_size, enqueued_sources
		src = sources[i]
		simulations = []
		tmp_sim = None
		parent_values = get_node_values(src.parents)

		if solutions:
			sources_root = src.get_root()
			if sources_root.size + 2 > best_size: return simulations

		for speed in filtered_conveyor_speeds:
			if src.value <= speed: break
			
			# if so then it would have been better to leave it as is
			# and merge all the other values to get the overflow value
			# we would get by exctracting speed amount
			if speed in parent_values: continue
			
			overflow_value = src.value - speed
			if gcd_incompatible(overflow_value): continue

			tmp_sim = tmp_sim if tmp_sim else get_sim_without(sources, src.value)
			sim = tuple(sorted(tmp_sim + [speed, overflow_value]))
			if sim in past: continue
			# if sim in enqueued_sources: continue
			simulations.append((sim, (i, speed)))
		
		return simulations

	def get_extract_sims(sources, cant_use, past):
	# def get_extract_sims(sources, cant_use):
		simulations = []
		seen_values = set()
		n = len(sources)

		for i in range(n):
			if stop_solving: break
			src = sources[i]
			if src.value in cant_use or src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(get_extract_sim(sources, i, past))
			# simulations.extend(get_extract_sim(sources, i))

		return simulations

	def get_divide_sim(sources, i, past):
	# def get_divide_sim(sources, i):
		src = sources[i]
		n_parents = len(src.parents)
		simulations = []
		parents_value_sum = None
		tmp_sim = None

		for divisor in allowed_divisors:
			if stop_solving: break
			if solutions:
				sources_root = src.get_root()
				if sources_root.size + divisor > best_size: break

			if not src.can_split(divisor): continue

			if not parents_value_sum: parents_value_sum = sum(get_node_values(src.parents))
			if parents_value_sum == src.value and n_parents == divisor: continue
			
			divided_value = int(src.value / divisor)
			if gcd_incompatible(divided_value): continue

			if not tmp_sim: tmp_sim = get_sim_without(sources, src.value)
			sim = tuple(sorted(tmp_sim + [divided_value] * divisor))
			if sim in past: continue
			# if sim in enqueued_sources: continue
			simulations.append((sim, (i, divisor)))

		return simulations

	def get_divide_sims(sources, cant_use, past):
	# def get_divide_sims(sources, cant_use):
		simulations = []
		seen_values = set()
		n = len(sources)
		
		for i in range(n):
			if stop_solving: break
			src = sources[i]
			if src.value in cant_use or src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(get_divide_sim(sources, i, past))
			# simulations.extend(get_divide_sim(sources, i))
		
		return simulations

	# def get_merge_sims(nodes, cant_use):
	def get_merge_sims(nodes, cant_use, past):
		simulations = []
		if solutions:
			nodes_root = sources[0].get_root()
			if nodes_root.size + 1 > best_size: return simulations
		n = len(nodes)
		indices = range(n)
		seen_sims = set()
		for to_sum_count in range(min_sum_count, max_sum_count + 1):
			if stop_solving: break
			for indices_comb in itertools.combinations(indices, to_sum_count):
				if stop_solving: break
				comb = [nodes[i] for i in indices_comb]
				if any(node.value in cant_use for node in comb): continue
				if all(len(node.parents) == 1 for node in comb):
					parent = comb[0].parents[0]
					if all(node.parents[0] is parent for node in comb) and len(parent.children) == to_sum_count and (parent.parents or source_values_length == 1): continue
				sim = sorted(node.value for node in nodes if node not in comb)
				summed_value = sum(node.value for node in comb)
				if gcd_incompatible(summed_value) or summed_value > conveyor_speed_limit: continue
				insert_into_sorted(sim, summed_value)
				sim = tuple(sim)
				if sim in past or sim in seen_sims: continue
				# if sim in enqueued_sources: continue
				# enqueued_sources.add(sim)
				simulations.append((sim, list(indices_comb)))
				seen_sims.add(sim)

		return simulations

	def compute_cant_use(sources):
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

	computed_distances = {}

	def compute_distance(sim):
		nonlocal computed_distances
		if sim in computed_distances: return computed_distances[sim]
		original_sim = sim
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
				for r in range(min_sum_count, max_sum_count + 1)
				for comb in combinations(sim, r)
				if sum(comb) == target
			], key=lambda x: len(x))
			if possible_merges:
				comb = possible_merges[-1]
				for v in comb: sim.remove(v)
				targets.remove(target)
				distance += 1

		r = distance + len(sim) + len(targets)
		computed_distances[original_sim] = r
		return r

	def is_solution(sources):
		n = len(sources)
		if n != target_values_length: return False
		for i in range(n):
			if sources[i].value != target_values[i]:
				return False
		return True
		# # Link the simplified targets' trees with the current one
		# for target in node_targets:
		#   for child in target.children:
		#       child.parents = []
		# for i in range(n):
		#   src = sources[i]
		#   src.children = node_targets[i].children
		#   for child in src.children:
		#       src.parents.append(child)

	# computes how close the sources are from the target_values
	# the lower the better
	def compute_sources_score(sources, past):
	# def compute_sources_score(sources):
		n = len(sources)
		simulations = []
		cant_use = compute_cant_use(sources)
		simulations.extend(get_extract_sims(sources, cant_use, past))
		simulations.extend(get_divide_sims(sources, cant_use, past))
		simulations.extend(get_merge_sims(sources, cant_use, past))
		# simulations.extend(get_extract_sims(sources, cant_use))
		# simulations.extend(get_divide_sims(sources, cant_use))
		# simulations.extend(get_merge_sims(sources, cant_use))
		score = -1
		# it required at least one operation to get there, hence the 1 +
		if simulations: score = 1 + min(compute_distance(sim) for sim, _ in simulations)
		return score

	def purge_queue():
		nonlocal queue
		for i in range(len(queue) - 1, -1, -1):
			if stop_solving: break
			sources, *_ = queue[i]
			if sources[0].get_root().size >= best_size: queue.pop(i)

	# def enqueue(nodes, past, operation):
	def enqueue(nodes, past):
	# def enqueue(nodes):
		nodes_root = nodes[0].get_root()
		nodes_root.compute_size(trim_root)
		if is_solution(nodes):
			if solution_found(nodes_root): purge_queue()
			# solution_found(nodes_root)
			# log(operation)
			return
		score = compute_sources_score(nodes, past)
		# score = compute_sources_score(nodes)
		if score < 0: return
		# to_insert = (nodes, score, past, operation)
		to_insert = (nodes, score, past)
		# to_insert = (nodes, score)
		insert_into_sorted(queue, to_insert, key=lambda x: x[1])
		# queue.append(nodes)
		# enqueued_sources.add(tuple(get_node_values(nodes)))

	def dequeue():
		nonlocal queue
		# return queue.pop(0)
		n = len(queue)
		
		# 50% chance to pick the item at 0% of the queue
		# 25% chance to pick the item at 50% of the queue
		# 12.5% chance to pick the item at 75% of the queue
		# 6.25% chance to pick the item at 87.5% of the queue
		# ... (kind of)
		# i = 1
		# while True:
		# 	tmp = 1 << (i - 1)
		# 	prob = 1 / (tmp << 1)
		# 	idx = round((1 - 1 / tmp) * n)
		# 	i += 1
		# 	if i > n or idx >= n: return queue.pop(-1)
		# 	if random.random() < prob: return queue.pop(idx)
		
		# 80% to pick the first
		# otherwise all other are equally probable
		if n < 3: return queue.pop(0)
		return queue.pop(0 if random.random() < 0.8 else random.randrange(1, n))

	# will be popped just after, no need to compute the score here
	# tho, everything that goes into the queue is exected to have its size computed
	# as well as the sources' values sorted
	node_sources[0].get_root().compute_size(trim_root)
	# queue.append((sort_nodes(node_sources), 1 << 16, set(), None))
	queue.append((sort_nodes(node_sources), 1 << 16, set()))
	# queue.append((sort_nodes(node_sources), 1 << 16, []))
	# queue.append((sort_nodes(node_sources), 1 << 16))
	# queue.append((sort_nodes(node_sources)))
	# lowest_score = 1000
	# steps = -1

	while not stop_solving and queue:
		sources, score, past = dequeue()
		sources_root = sources[0].get_root()
		source_values = get_node_values(sources)
		# if source_values == [5, 325, 325]:
			# print("ONE", past)
			# print("ONE")

		n = len(sources)
		cant_use = compute_cant_use(sources)
		sources_id = get_node_ids(sources)

		# steps += 1
		# if steps + 1 == 0:
		#   exit(0)
		# print(f"step {steps}")
		# print(sources_root.size)

		def copy_sources():
			_, leaves = sources_root.deepcopy()
			past_copy = copy.deepcopy(past)
			past_copy.add(tuple(source_values))
			# past_copy.append(tuple(source_values))
			return sort_nodes([leaf for leaf in leaves if leaf.node_id in sources_id]), past_copy
			# return sort_nodes([leaf for leaf in leaves if leaf.node_id in sources_id])

		def try_extract():
			simulations = get_extract_sims(sources, cant_use, past)
			# simulations = get_extract_sims(sources, cant_use)
			for _, (i, speed) in simulations:
				if stop_solving: break
				copy, past_copy = copy_sources()
				# copy = copy_sources()
				src_copy = copy[i]
				pop_node(src_copy, copy)

				if logging:
					log("\n\nFROM")
					log(sources_root)
					log("DID")
					log(f"{src_copy} - {speed}")

				# operation = f"\n\nFROM\n{sources_root}\nDID\n{src_copy} - {speed}"

				for node in src_copy - speed:
					insert_into_sorted(copy, node, lambda node: node.value)
				# enqueue(copy, past_copy, operation)
				enqueue(copy, past_copy)
				# enqueue(copy)

		def try_divide():
			simulations = get_divide_sims(sources, cant_use, past)
			# simulations = get_divide_sims(sources, cant_use)
			for _, (i, divisor) in simulations:
				if stop_solving: break
				copy, past_copy = copy_sources()
				# copy = copy_sources()
				src_copy = copy[i]
				pop_node(src_copy, copy)

				if logging:
					log("\n\nFROM")
					log(sources_root)
					log("DID")
					log(f"{src_copy} / {divisor}")

				# operation = f"\n\nFROM\n{sources_root}\nDID\n{src_copy} / {divisor}"

				for node in src_copy / divisor:
					insert_into_sorted(copy, node, lambda node: node.value)
				# enqueue(copy, past_copy, operation)
				enqueue(copy, past_copy)
				# enqueue(copy)

		def try_merge():
			simulations = get_merge_sims(sources, cant_use, past)
			# simulations = get_merge_sims(sources, cant_use)
			for _, to_sum_indices in simulations:
				if stop_solving: break
				copy, past_copy = copy_sources()
				# copy = copy_sources()
				to_sum = [copy[i] for i in to_sum_indices]
				list(map(lambda src: pop_node(src, copy), to_sum))

				if logging:
					log("\n\nFROM")
					log(sources_root)
					log("DID")
					log("+".join(str(ts) for ts in to_sum))

				# operation = f"\n\nFROM\n{sources_root}\nDID\n{"+".join(str(ts) for ts in to_sum)}"

				insert_into_sorted(copy, to_sum[0] + to_sum[1:], lambda node: node.value)
				# enqueue(copy, past_copy, operation)
				enqueue(copy, past_copy)
				# enqueue(copy)

		try_divide()
		if stop_solving: break
		try_extract()
		if stop_solving: break
		try_merge()
	
	solving = False

def solve(source_values, target_values):
	global solving, stop_solving, trim_root
	sources_total = sum(source_values)
	targets_total = sum(target_values)
	if sources_total > targets_total:
		value = sources_total - targets_total
		target_values.append(value)
		print(f"\nTargets are lacking, generating a {value} node as target")
	elif sources_total < targets_total:
		value = targets_total - sources_total
		source_values.append(value)
		print(f"\nSources are lacking, generating a {value} node as source")
	trim_root = len(source_values) > 1
	stop_solving = False
	stop_concluding = False
	solving = True
	def _catching_solve():
		global solving
		try:
			_solve(source_values, target_values)
		except:
			print(traceback.format_exc(), end="")
			solving = False
	solve_thread = threading.Thread(target=_catching_solve, daemon=True)
	solve_thread.start()
	# keep this thread alive to catch ctrl + c
	try:
		while solving: time.sleep(0.25)
	except KeyboardInterrupt:
		pass
	solve_thread.join()

def main(user_input):
	separator = 'to'
	if len(user_input.split(" ")) < 3 or separator not in user_input:
		print(f"Usage: <source_args> {separator} <target_args>")
		return 0

	source_part, target_part = user_input.split(separator)
	source_args = source_part.strip().split()
	target_args = target_part.strip().split()

	if not source_args:
		print("Error: At least one source value must be provided.")
		return 1

	if not target_args:
		print("Error: At least one target value must be provided.")
		return 1

	sources = []
	i = 0
	while i < len(source_args):
		src = source_args[i]
		if not src.endswith('x'):
			source_value = int(src)
			sources.append(source_value)
			i += 1
			continue
		if len(src) < 2 or not src[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return 1
		multiplier = int(src[:-1])
		source_value = int(source_args[source_args.index(src) + 1])
		for _ in range(multiplier):
			sources.append(source_value)
		i += 2

	targets = []
	i = 0
	while i < len(target_args):
		target = target_args[i]
		if not target.endswith('x'):
			target_value = int(target)
			targets.append(target_value)
			i += 1
			continue
		if len(target) < 2 or not target[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return 1
		multiplier = int(target[:-1])
		if i + 1 == len(target_args):
			print("Error: You must provide a target value after Nx.")
			return 1
		target_value = int(target_args[i + 1])
		for _ in range(multiplier):
			targets.append(target_value)
		i += 2

	solve(sources, targets)
	conclude()

user_input = None
idle = True
stop_event = threading.Event()
input_lock = threading.Lock()

def exit(code):
	stop_event.set()
	sys.exit(code)

def handler(signum, frame):
	global stop_solving, stop_concluding
	# print("\nreceived ctrl+c")
	if idle:
		# print("\nwas idle")
		print()
		exit(0)
	else:
		if concluding:
			# print("\nwas concluding")
			stop_concluding = True
		elif not stop_solving:
			print("Stopping...")
			stop_solving = True

signal.signal(signal.SIGINT, handler)

def input_thread_callback():
	global user_input, idle
	while not stop_event.is_set():
		try:
			while not idle and not stop_event.is_set():
				pass
			if stop_event.is_set():
				break
			with input_lock:
				user_input = input("\nSatisfactory Solver> ")
			idle = False
		except EOFError:
			break

def test():
	# node655 = Node(655)

	# node650 = Node(650)
	# node5 = Node(5)

	# node325_1 = Node(325)
	# node325_2 = Node(325)

	# node330 = Node(330)

	# node120 = Node(120)
	# node205 = Node(205)
	
	# node450 = Node(450)

	# node150_1 = Node(150)
	# node150_2 = Node(150)
	# node150_3 = Node(150)

	# node655.children = [node650, node5]
	# node650.children = [node325_1, node325_2]
	# node5.children = [node330]
	# node325_1.children = [node330]
	# node325_2.children = [node205, node120]
	# node330.children = [node450]
	# node120.children = [node450]
	# node450.children = [node150_1, node150_2, node150_3]

	# tmp = str(node655)
	# tmp = re.sub("parents=\\[\\]", "parents=[.*]", tmp)
	# for c in "()[]":
	# 	tmp = tmp.replace(c, "\\" + c)
	# tmp = re.sub("short_node_id=(.*?),", "short_node_id=.*,", tmp)
	# pattern = re.sub("value=(.*?),", "value=.*,", tmp)
	
	# content = None
	# with open('logs.txt', 'r') as file:
	# 	content = file.read()

	# results = re.findall(pattern, content)

	# print(results)
	# cProfile.run('solve([475, 85, 100], [45, 55, 100])')
	# cProfile.run('solve([5, 650], [150, 150, 150, 205])')
	cProfile.run('solve([40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50], [420, 420])')
	pass

if __name__ == '__main__':
	test()
	exit(0)
	input_thread = threading.Thread(target=input_thread_callback, daemon=True)
	input_thread.start()

	while not stop_event.is_set():
		if user_input is None:
			continue
		with input_lock:
			if user_input in ["exit", "quit", "q"]:
				stop_event.set()
				break
			main(user_input)
			idle = True
			user_input = None

	input_thread.join()

# graveyard

# def _simplify_merge(nodes):
# 	global allowed_divisors_r
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
# 				if len(same_value_nodes) == allowed_divisors_r[0] - 1:
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
# 	global conveyor_speeds_r
# 	# Step 2: Extract maximum conveyor speed that fits (ignore nodes with value already equal to a conveyor speed)
# 	extracted_nodes = []
# 	for node in nodes:
# 		extracted_flag = False
# 		for speed in conveyor_speeds_r:
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