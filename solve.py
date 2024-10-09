# import argparse
import networkx as nx
import matplotlib.pyplot as plt
import uuid
import pydot
import sys
import pathlib
import os
import tempfile
if sys.platform == 'win32':
	path = pathlib.Path(r'C:\Program Files\Graphviz\bin')
	if path.is_dir() and str(path) not in os.environ['PATH']:
		os.environ['PATH'] += f';{path}'
import pygraphviz
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph
import math
import time
import signal
import random
import itertools

timings = {
	"total": 0,
	"copy_sources": 0,
	"try_divide": 0,
	"try_extract": 0,
	"try_merge": 0,
	"get_sim_without": 0,
	"enqueue": 0,
}

def print_timings():
	total_time = timings["total"]
	for key, val in timings.items():
		if key == "total": continue
		print(f"{key}: {val / total_time:.4f}")

def handler(signum, frame):
	print("Stopping and printing timing averages...")
	print_timings()
	exit(0)

signal.signal(signal.SIGINT, handler)

def time_block(key, fn, *args, **kwargs):
	start_time = time.time()
	result = fn(*args, **kwargs)
	timings[key] += time.time() - start_time
	return result

def set_time(key, start_time):
	timings[key] += time.time() - start_time

allowed_divisors = [2, 3]
allowed_divisors_r = allowed_divisors[::-1]
conveyor_speeds = [60, 120, 270, 480, 780, 1200]
conveyor_speeds_r = conveyor_speeds[::-1]
conveyor_speed_limit = conveyor_speeds_r[0]
short_repr = False

# def safe_add_parent(parent, node):
# 	if node is parent or node.has_parent(parent):
# 		print("self parent")
# 		print(node)
# 		exit(1)
# 	node.parents.append(parent)

def visualize(src, nodes):
	src.children = nodes
	for node in nodes:
		node.parents.append(src)
	src.visualize()

def sort_nodes(nodes):
	return sorted(nodes, key=lambda node: node.value)

def get_values(nodes):
	return list(map(lambda node: node.value, nodes))

def get_nodes_id(nodes, short=3):
	return list(map(lambda node: node.node_id[-short:], nodes))

def pop(node, nodes):
	for i, other in enumerate(nodes):
		if other.node_id == node.node_id:
			return nodes.pop(i)
	return None

def can_split(value, divisor):
	global allowed_divisors
	if not divisor in allowed_divisors: return False
	return value % divisor == 0

def increment(binary_array):
	for i in range(len(binary_array)):
		binary_array[i] = not binary_array[i]
		if binary_array[i]: break

def insert_into_sorted(sorted_list, item, key=lambda x: x):
	low, high = 0, len(sorted_list)
	while low < high:
		mid = low + (high - low) // 2
		if key(item) > key(sorted_list[mid]):
			low = mid + 1
		else:
			high = mid
	sorted_list.insert(low, item)

class Node:
	def __init__(self, value, node_id=None):
		self.value = value
		self.node_id = node_id if node_id is not None else str(uuid.uuid4())
		self.depth = 1
		self.tree_height = 1
		self.level = None
		self.size = None
		self.parents = []
		self.children = []

	def __repr__(self):
		if short_repr:
			return f"{"\t" * (self.depth - 1)}{self.value}({self.node_id[-3:]})"
		r = f"{"\t" * (self.depth - 1)}Node(value={self.value}, short_node_id={self.node_id[-3:]}, depth={self.depth}, tree_height={self.tree_height}, level={self.level}, size={self.size}, parents={get_nodes_id(self.parents)}, children=["
		if self.children:
			r += "\n"
			for child in self.children:
				r += str(child)
			r += "\t" * (self.depth - 1)
		r += "])\n"
		return r

	def get_leaves(self):
		if not self.children:
			return [self]
		leaves = []
		for child in self.children:
			leaves.extend(child.get_leaves())
		return leaves

	def get_root(self):
		cur = self
		while cur.parents:
			cur = cur.parents[0]
		return cur

	def _compute_size(self, visited):
		# the returned boolean is there to indicate if the returned size is unseen for parents
		if self.node_id in visited:
			return self.size, False
		visited.add(self.node_id)
		self.size = 1
		unseen_size = self.size
		for child in self.children:
			child_size, child_unvisited = child._compute_size(visited)
			if child_unvisited:
				unseen_size += child_size
			self.size += child_size
		return unseen_size, True

	def compute_size(self):
		self.get_root()._compute_size(set())
		return self.size

	def _deepcopy(self):
		new_node = Node(self.value, node_id=self.node_id)
		new_node.depth = self.depth
		new_node.tree_height = self.tree_height
		new_node.level = self.level
		new_node.size = self.size
		leaves = []
		if self.children:
			for child in self.children:
				new_child, child_leaves = child._deepcopy()
				new_node.children.append(new_child)
				new_child.parents.append(new_node)
				leaves += child_leaves
		else:
			leaves.append(new_node)
		return new_node, leaves

	def deepcopy(self):
		return self.get_root()._deepcopy()

	def find(self, node_id):
		if self.node_id == node_id: return self
		for child in self.children:
			result = child.find(node_id)
			if result: return result
		return None

	def has_parent(self, parent):
		for p in self.parents:
			if p is parent or p.has_parent(parent): return True
		return False

	def _compute_depth_and_tree_height(self, parent_depth):
		max_child_tree_height = 0
		for child in self.children:
			child.depth = 1 + parent_depth
			_, child_tree_height = child._compute_depth_and_tree_height(child.depth)
			if child_tree_height > max_child_tree_height:
				max_child_tree_height = child_tree_height
		self.tree_height = max_child_tree_height + 1
		return self.depth, self.tree_height

	# def set_max_tree_height(self, max_tree_height):
	# 	self.max_tree_height = max_tree_height
	# 	for child in self.children:
	# 		child.set_max_tree_height(max_tree_height)

	def compute_level(self, max_tree_height):
		self.level = max_tree_height - \
		(max(map(lambda node: node.tree_height, self.children)) if self.children else 0)
		for child in self.children:
			child.compute_level(max_tree_height)

	def compute_depth_informations(self):
		self.depth = 1 + (self.parents[0].depth if self.parents else 0)
		self._compute_depth_and_tree_height(self.depth)
		# self.set_max_tree_height(self.tree_height)
		self.compute_level(self.tree_height)

	def populate(self, G):
		G.add_node(self.node_id, label=str(self.value), level=self.level)
		for child in self.children:
			G.add_edge(self.node_id, child.node_id)
			child.populate(G)

	def visualize(self):
		try:
			G = nx.DiGraph()
			self.populate(G)

			A = to_agraph(G)

			for node in G.nodes:
				level = G.nodes[node]['level']
				A.get_node(node).attr['rank'] = f'{level}'

			for level in set(nx.get_node_attributes(G, 'level').values()):
				subgraph = A.add_subgraph(
					[n for n, attr in G.nodes(data=True) if attr['level'] == level],
					rank='same'
				)

			A.graph_attr['rankdir'] = 'TB'

			with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
				A.layout(prog='dot')
				A.draw(tmpfile.name, format='png')
				plt.figure(figsize=(10, 7))
				img = plt.imread(tmpfile.name)
				plt.imshow(img)
				plt.axis('off')
				plt.show()
		except:
			print("call compute_depth_informations before trying to visualize")
			exit(1)

	def merge_down(self, other):
		new_value = self.value + sum(get_values(other))
		new_node = Node(new_value)
		self.children.append(new_node)
		new_node.parents.append(self)
		for node in other:
			node.children.append(new_node)
			new_node.parents.append(node)
		return new_node

	def merge_up(self, other):
		new_value = self.value + sum(get_values(other))
		new_node = Node(new_value)
		self.parents.append(new_node)
		new_node.children.append(self)
		for node in other:
			node.parents.append(new_node)
			new_node.children.append(node)
		return new_node

	def can_split(self, divisor):
		return can_split(self.value, divisor)

	def split_down(self, divisor):
		new_value = int(self.value / divisor)
		new_nodes = [Node(new_value) for _ in range(divisor)]
		for node in new_nodes:
			self.children.append(node)
			node.parents.append(self)
		return new_nodes

	def split_up(self, divisor):
		new_value = int(self.value / divisor)
		new_nodes = [Node(new_value) for _ in range(divisor)]
		for node in new_nodes:
			self.parents.append(node)
			node.children.append(self)
		return new_nodes

	def extract_down(self, value):
		extracted_node = Node(value)
		overflow_value = self.value - value
		overflow_node = Node(overflow_value)
		self.children.append(extracted_node)
		self.children.append(overflow_node)
		extracted_node.parents.append(self)
		overflow_node.parents.append(self)
		return [extracted_node, overflow_node]

	def extract_up(self, value):
		extracted_node = Node(value)
		overflow_value = self.value - value
		overflow_node = Node(overflow_value)
		self.parents.append(extracted_node)
		self.parents.append(overflow_node)
		extracted_node.children.append(self)
		overflow_node.children.append(self)
		return [extracted_node, overflow_node]

	def __add__(self, other):
		if isinstance(other, list) and all(isinstance(node, Node) for node in other):
			return self.merge_down(other)
		elif isinstance(other, Node):
			return self.merge_down([other])		
		raise ValueError("Operand must be a Node")

	def __truediv__(self, divisor):
		if isinstance(divisor, (int, float)):
			return self.split_down(divisor)
		raise ValueError(f"Divisor must be an integer (one of these: {"/".join(allowed_divisors)})")

	def __sub__(self, amount):
		if isinstance(amount, (int, float)):
			return self.extract_down(amount)
		raise ValueError("Amount must be a number")

def _simplify_merge(nodes):
	global allowed_divisors_r
	# Step 1: Merge nodes with the same value until all are different
	has_merged = False
	while True:
		merged_nodes = []
		done = True
		i = 0

		while i < len(nodes):
			current_node = nodes[i]
			current_value = current_node.value
			same_value_nodes = []

			i += 1
			while i < len(nodes) and nodes[i].value == current_value:
				if len(same_value_nodes) == allowed_divisors_r[0] - 1:
					break
				same_value_nodes.append(nodes[i])
				i += 1

			if len(same_value_nodes) > 0:
				merged_node = current_node.merge_up(same_value_nodes)
				merged_nodes.append(merged_node)
				done = False
				has_merged = True
			else:
				merged_nodes.append(current_node)

		if done: break

		merged_nodes = sort_nodes(merged_nodes)
		nodes = [node for node in merged_nodes]
	return nodes, has_merged

def _simplify_extract(nodes):
	global conveyor_speeds_r
	# Step 2: Extract maximum conveyor speed that fits (ignore nodes with value already equal to a conveyor speed)
	extracted_nodes = []
	for node in nodes:
		extracted_flag = False
		for speed in conveyor_speeds_r:
			if node.value == speed: break
			if node.value > speed:
				extracted_node, overflow_node = node.extract_up(speed)
				extracted_nodes.append(extracted_node)
				extracted_nodes.append(overflow_node)
				extracted_flag = True
				break
		if not extracted_flag:
			extracted_nodes.append(node)

	nodes = sort_nodes(extracted_nodes)
	return nodes

def simplify(nodes):
	nodes, has_merged = _simplify_merge(nodes)
	nodes = _simplify_extract(nodes)
	while has_merged:
		nodes, has_merged = _simplify_merge(nodes)
		if not has_merged: break
		nodes = _simplify_extract(nodes)
	return nodes

def _get_sim_without(sources, value):
	sim = []
	found = False
	for src in sources:
		if src.value == value and not found:
			found = True
		else:
			sim.append(src.value)
	return sim

def get_sim_without(sources, value):
	return time_block("get_sim_without", _get_sim_without, sources, value)

def _solve(source_values, target_values):
	print(f"\nsolving for {target_values}\n")
	steps = -1

	enqueued_sims = set()
	target_values = sorted(target_values)
	target_counts = {
		value: target_values.count(value) for value in set(target_values)
	}
	gcd = math.gcd(*target_values)
	node_sources = list(map(lambda value: Node(value), source_values))
	# node_targets = list(map(lambda value: Node(value), target_values))
	# print('\n'.join([str(src) for src in copy]))

	def gcd_incompatible(value):
		return value < gcd or value % gcd != 0

	filtered_conveyor_speeds = [speed for speed in conveyor_speeds if not gcd_incompatible(speed)]
	filtered_conveyor_speeds_r = filtered_conveyor_speeds[::-1]
	print(f"gcd = {gcd}, filtered_conveyor_speeds = {filtered_conveyor_speeds}")

	if len(node_sources) > 1:
		root = Node(sum(source_values))
		root.children = node_sources
		for child in root.children:
			child.parents.append(root)

	queue = []
	solution = None
	cant_use = {}

	def get_extract_sim(sources, i):
		nonlocal solution
		src = sources[i]
		simulations = []
		tmp_sim = None
		parent_values = get_values(src.parents)
		
		for speed in filtered_conveyor_speeds:
			# if so then it would have been better to leave it as is
			# and merge all the other values to get the overflow value
			# we would get by exctracting speed amount
			if speed in parent_values:
				print("sad that we got there")
				exit(1)
				# return None
			
			if src.value <= speed: break

			if solution:
				if not sources[0].size: sources[0].compute_size()
				if sources[0].size + 2 >= solution.size: continue
			
			overflow_value = src.value - speed
			if gcd_incompatible(overflow_value): continue

			tmp_sim = tmp_sim if tmp_sim else get_sim_without(sources, src.value)
			sim = tuple(sorted(tmp_sim + [speed, overflow_value]))
			if sim in enqueued_sims: continue
			simulations.append((sim, (i, speed)))
		
		return simulations

	def get_extract_sims(sources):
		nonlocal cant_use
		simulations = []
		seen_values = set()
		
		for i in range(len(sources)):
			src = sources[i]
			if cant_use[src.value] or src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(get_extract_sim(sources, i))
		
		return simulations

	def get_divide_sim(sources, i):
		nonlocal solution
		global allowed_divisors
		src = sources[i]
		simulations = []
		tmp_sim = None
		
		for divisor in allowed_divisors:
			if not src.can_split(divisor): continue

			if sum(get_values(src.parents)) == src.value and len(src.parents) == divisor: return

			if solution:
				if not sources[0].size: sources[0].compute_size()
				if sources[0].size + divisor >= solution.size: continue
			
			divided_value = int(src.value / divisor)
			if gcd_incompatible(divided_value): continue

			tmp_sim = tmp_sim if tmp_sim else get_sim_without(sources, src.value)
			sim = tuple(sorted(tmp_sim + [divided_value] * divisor))
			if sim in enqueued_sims: continue
			simulations.append((sim, (i, divisor)))
		
		return simulations

	def get_divide_sims(sources):
		nonlocal cant_use
		simulations = []
		seen_values = set()
		
		for i in range(len(sources)):
			if cant_use[src.value] or src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(get_divide_sim(sources, i))
		
		return simulations

	def get_merge_sim(sources, flags, to_sum_count, seen_sums):
		nonlocal cant_use
		to_not_sum_indices = []
		i = 0
		
		while not flags[i]:
			to_not_sum_indices.append(i)
			i += 1
		
		src = sources[i]
		
		if cant_use[src.value]: return None
		
		to_sum_indices = [i]
		parent = src.parents[0]
		same_parent = len(src.parents) == 1
		
		while i < n - 1:
			i += 1
			
			if not flags[i]:
				to_not_sum_indices.append(i)
				continue
			
			src = sources[i]
			
			if cant_use[src.value]: return None
			
			if len(src.parents) != 1 or not src.parents[0] is parent:
				same_parent = False
			
			to_sum_indices.append(i)

		if same_parent and to_sum_count == len(src.parents[0].children):
			# can happen that the parent was the artificial root created to unify all sources
			# in this case only we don't skip
			if parent.parents or len(source_values) == 1: return None

		to_sum_values = [sources[i].value for i in to_sum_indices]
		summed_value = sum(to_sum_values)
		if gcd_incompatible(summed_value) or summed_value > conveyor_speed_limit: return None
		
		to_sum_values = sorted(to_sum_values)
		if to_sum_values in seen_sums: return None
		seen_sums.add(to_sum_values)

		sim = tuple(sorted([sources[i].value for i in to_not_sum_indices] + [summed_value]))
		if sim in enqueued_sims: return None
		
		return sim, to_sum_indices

	def get_merge_sims(sources):
		nonlocal solution
		simulations, n = [], len(sources)
		
		if n < 2: return simulations
		
		seen_sums = {}
		binary = [False] * n
		binary[0], binary[1] = True, True
		
		for _ in range(3, 1 << n): # 2^n - 1
			to_sum_count = sum(binary)
			
			if to_sum_count < 2 or to_sum_count > 3: continue
			if solution and n - to_sum_count + 1 >= solution.size: continue

			r = get_merge_sim(sources, binary, to_sum_count, seen_sums)
			if r: simulations.append(r)
			
			increment(binary)
		
		return simulations

	def compute_distance(sim):
		nonlocal target_values, filtered_conveyor_speeds_r
		global allowed_divisors, allowed_divisors_r
		sim = list(sim)
		targets = target_values[:]
		distance = 0
		
		for value in sim[:]:
			if value in targets:
				sim.remove(value)
				targets.remove(value)

		# remove all perfect extractions
		for speed in filtered_conveyor_speeds_r:
			if not speed in targets: continue

			for value in sim:
				if value <= speed: continue
				overflow = value - speed
				if gcd_incompatible(overflow) or not overflow in targets: continue
				sim.remove(value)
				targets.remove(speed)
				targets.remove(overflow)
				distance += 1
		
		# remove all perfect extractions
		for speed in filtered_conveyor_speeds:
			if not speed in targets: continue

			for value in sim:
				if value <= speed: continue
				overflow = value - speed
				if gcd_incompatible(overflow): continue
				sim.remove(value)
				sim.append(overflow)
				targets.remove(speed)
				distance += 2

		def try_divide(value, divisor):
			nonlocal targets
			if value % divisor != 0: return None, 0
			
			divided_value = value // divisor
			
			if gcd_incompatible(divided_value): return None, 0

			matches, remaining_targets = 0, targets[:]

			for divided_value in extras[:]:
				if divided_value in remaining_targets:
					matches += 1
					remaining_targets.remove(divided_value)

			return divided_value, matches

		# remove all perfect divisions
		for divisor in allowed_divisors_r:
			while True:
				done = True				
				for value in sim[:]:
					_, matches = try_divide(value, divisor)
					if not matches or matches != divisor: continue
					sim.remove(best_value)
					for _ in range(matches): targets.remove(divided_value)
					distance += 1
					done = False
				if done: break				
		
		# remove all divisions that have at least one match
		for divisor in allowed_divisors:
			while True:
				done = True
				for value in sim[:]:
					divided_value, matches = try_divide(value, divisor)
					if not matches: continue
					if matches == divisor:
						print("impossible case reached, all perfect divisions were removed already")
						exit(1)
					sim.remove(best_value)
					extras = divisor - matches
					for _ in range(matches): targets.remove(divided_value)
					for _ in range(extras): sim.append(divided_value)
					distance += 1 + extras
					done = False
				if done: break

		for target in targets[:]:
			found = False
			for to_sum_count in allowed_divisors_r:
				for to_sum_values in itertools.combinations(sim, to_sum_count):
					if sum(to_sum_values) != target: continue
					
					targets.remove(target)
					for val in to_sum_values: sim.remove(val)

					distance += 1
					found = True
					break
				if found: break
		
		return distance + len(sim) + len(targets)

	# computes how close the sources are from the target_values
	# the lower the better
	def compute_sources_score(sources):
		n = len(sources)
		simulations = []
		simulations.extend(get_extract_sims(sources))
		simulations.extend(get_divide_sims(sources))
		simulations.extend(get_merge_sims(sources))
		return min(compute_distance(sim) for sim, _ in simulations)

	def _enqueue(sources):
		nonlocal queue
		insert_into_sorted(queue, (sources, compute_sources_score(sources)), key=lambda x: x[1])

	def enqueue(sources):
		time_block("enqueue", _enqueue, sources)

	# will be popped just after, no need to compute the score here
	queue.append((node_sources, 1 << 16))
	lowest_score = 1000

	while queue:
		start_total = time.time()
		tmp, score = queue.pop(0)
		sources = sort_nodes(tmp)
		if score < lowest_score:
			root = sources[0].get_root()
			root.compute_depth_informations()
			print(f"lowest score = {score}, tree =\n{root}")
			lowest_score = score
		# print(f"step {abs(steps)}")

		steps -= 1
		if steps + 1 == 0:
			print("stopping")
			print_timings()
			exit(0)
		if (-steps) % 1000 == 0:
			print(f"step {abs(steps)}")

		def _copy_sources():
			_, leaves = sources[0].deepcopy()
			return sort_nodes(leaves)

		def copy_sources():
			return time_block("copy_sources", _copy_sources)

		n = len(sources)

		if n == len(target_values):
			matches = True
			for i in range(n):
				if sources[i].value != target_values[i]:
					matches = False
					break
			if matches:
				

				# Link the simplified targets' trees with the current one
				# for target in node_targets:
				# 	for child in target.children:
				# 		child.parents = []
				# for i in range(n):
				# 	src = sources[i]
				# 	src.children = node_targets[i].children
				# 	for child in src.children:
				# 		src.parents.append(child)

				new_solution = sources[0].get_root()
				new_solution._compute_size(set())
				if solution is None or new_solution.size < solution.size:
					solution = new_solution
				else:
					print("impossible case reached, should have been checked already")
					exit(1)
				solution.compute_depth_informations()
				print(f"one solution found of size {solution.size}\n")
				print(solution)
				solution.visualize()

		source_counts = {}
		for src in sources:
			if src.value in source_counts:
				source_counts[src.value] += 1
			else:
				source_counts[src.value] = 1

		cant_use = {}
		for src in sources:
			value = src.value
			src_count = source_counts.get(value, None)
			target_count = target_counts.get(value, None)
			cant_use[value] = max(0, src_count - target_count) == 0 if src_count and target_count else False

		def _try_extract():
			nonlocal sources
			for sim, (i, speed) in get_extract_sims(sources):
				copy = copy_sources()
				src_copy = copy[i]
				pop(src_copy, copy)
				enqueue(copy + (src_copy - speed))
				enqueued_sims.add(sim)

		def try_extract():
			time_block("try_extract", _try_extract)
		
		def _try_divide():
			nonlocal sources
			for sim, (i, divisor) in get_divide_sims(sources):
				copy = copy_sources()
				src_copy = copy[i]
				pop(src_copy, copy)
				enqueue(copy + (src_copy / divisor))
				enqueued_sims.add(sim)

		def try_divide():
			time_block("try_divide", _try_divide)

		def _try_merge():
			nonlocal sources
			for sim, to_sum_indices in get_merge_sims(sources):
				copy = copy_sources()
				to_sum = [copy[i] for i in to_sum_indices]
				list(map(lambda src: pop(src, copy), to_sum))
				enqueue(copy + [to_sum[0] + to_sum[1:]])
				enqueued_sims.add(sim)

		def try_merge():
			time_block("try_merge", _try_merge)

		try_divide()
		try_extract()
		try_merge()

		timings["total"] += time.time() - start_total
	
	return solution

def solve(source_values, target_values):
	sources_total = sum(source_values)
	targets_total = sum(target_values)
	if sources_total > targets_total:
		target_values.append(sources_total - targets_total)
	elif sources_total < targets_total:
		source_values.append(targets_total - sources_total)
	return _solve(source_values, target_values)

def main():
	separator = 'to'
	if len(sys.argv) < 3 or separator not in ' '.join(sys.argv[1:]):
		print(f"Usage: python solve.py <source_args> {separator} <target_args>")
		exit(1)

	source_part, target_part = ' '.join(sys.argv[1:]).split(separator)
	source_args = source_part.strip().split()
	target_args = target_part.strip().split()

	if not source_args:
		print("Error: At least one source value must be provided.")
		exit(1)

	if not target_args:
		print("Error: At least one target value must be provided.")
		exit(1)

	sources = []
	i = 0
	while i < len(source_args):
		src = source_args[i]
		if not src.endswith('x'):
			source_value = int(src)
			if source_value % 5 != 0:
				print("Error: all values must be multiples of 5")
				exit(1)
			sources.append(source_value)
			i += 1
			continue
		if len(src) < 2 or not src[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			exit(1)
		multiplier = int(src[:-1])
		source_value = int(source_args[source_args.index(src) + 1])
		if source_value % 5 != 0:
			print("Error: all values must be multiples of 5")
			exit(1)
		for _ in range(multiplier):
			sources.append(source_value)
		i += 2

	targets = []
	i = 0
	while i < len(target_args):
		target = target_args[i]
		if not target.endswith('x'):
			target_value = int(target)
			if target_value % 5 != 0:
				print("Error: all values must be multiples of 5")
				exit(1)
			targets.append(target_value)
			i += 1
			continue
		if len(target) < 2 or not target[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			exit(1)
		multiplier = int(target[:-1])
		if i + 1 == len(target_args):
			print("Error: You must provide a target value after Nx.")
			exit(1)
		target_value = int(target_args[i + 1])
		if target_value % 5 != 0:
			print("Error: all values must be multiples of 5")
			exit(1)
		for _ in range(multiplier):
			targets.append(target_value)
		i += 2

	sol = solve(sources, targets)
	print(f"\n Smallest solution found (size = {sol.size}):\n")
	print(sol)
	sol.visualize()

def test():
	A = Node(100)
	B, C = A - 60
	D = B + C
	A.compute_depth_informations()
	A.compute_size()
	print(A)

if __name__ == '__main__':
	# test()
	main()