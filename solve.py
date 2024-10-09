import argparse
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

timings = {
	"total": 0,
	"copy_sources": 0,
	"try_divide": 0,
	"try_extract": 0,
	"try_merge": 0,
	"get_sim_without": 0,
	"get_to_sums1": 0,
	"get_to_sums2": 0,
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

allowed_divisions = [3, 2]
conveyor_speeds = [60, 120, 270, 480, 780, 1200]
conveyor_speeds_r = sorted(conveyor_speeds, reverse=True)
conveyor_speed_limit = conveyor_speeds[-1]
short_repr = False

def safe_add_parent(parent, node):
	if node is parent or node.has_parent(parent):
		print("self parent")
		print(node)
		exit(1)
	node.parents.append(parent)

def visualize(src, nodes):
	src.children = nodes
	for node in nodes:
		safe_add_parent(src, node)
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

def increment(binary_array):
	for i in range(len(binary_array)):
		binary_array[i] = not binary_array[i]
		if binary_array[i]: break

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
		r = f"{"\t" * (self.depth - 1)}Node(value={self.value}, short_node_id={self.node_id[-3:]}, depth={self.depth}, tree_height={self.tree_height}, level={self.level}, size={self.size}, children=["
		if self.children:
			for child in self.children:
				r += "\n" + str(child)
			r += "\n" + "\t" * (self.depth - 1) + "])"
		else:
			r += "])"
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

	def _compute_size(self):
		size = 1
		if self.children:
			size += sum(child._compute_size() for child in self.children)
		self.size = size
		return size

	def compute_size(self):
		return self.get_root()._compute_size()

	def _deepcopy(self):
		new_node = Node(self.value, node_id=self.node_id)
		new_node.depth = self.depth
		new_node.tree_height = self.tree_height
		new_node.level = self.level
		new_node.children = [child._deepcopy() for child in self.children]
		for child in new_node.children:
			safe_add_parent(new_node, child)
		return new_node

	def deepcopy(self):
		deep_copied_root = self.get_root()._deepcopy()
		return deep_copied_root.find(self.node_id)

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

	def compute_depth_and_tree_height(self, parent):
		self.depth = 1 + parent.depth if parent else 1
		max_child_tree_height = 0
		for child in self.children:
			child_depth, child_tree_height = child.compute_depth_and_tree_height(self)
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
		self.compute_depth_and_tree_height(self.parents[0] if self.parents else None)
		# self.set_max_tree_height(self.tree_height)
		self.compute_level(self.tree_height)

	def add_edges(self, G):
		G.add_node(self.node_id, label=str(self.value), level=self.level)
		for child in self.children:
			child.add_edges(G)

	def visualize(self):
		self.compute_depth_informations()
		G = nx.DiGraph()
		self.add_edges(G)

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

	def _merge(self, down, other):
		# print("merge called")
		new_value = self.value + sum(get_values(other))
		new_node = Node(new_value)
		if down:
			safe_add_parent(self, new_node)
			self.children.append(new_node)
			for node in other:
				safe_add_parent(node, new_node)
				node.children.append(new_node)
		else:
			safe_add_parent(new_node, self)
			new_node.children.append(self)
			for node in other:
				safe_add_parent(new_node, node)
				new_node.children.append(node)
		return new_node

	def merge_down(self, other):
		return self._merge(True, other)

	def merge_up(self, other):
		return self._merge(False, other)

	def can_split(self, divisor):
		if divisor not in allowed_divisions: return False
		new_value = self.value / divisor
		if new_value.is_integer(): return True
		return False

	def _split(self, down, divisor):
		# print("split called")
		if not self.can_split(divisor): return None
		new_value = int(self.value / divisor)
		new_nodes = [Node(new_value) for _ in range(divisor)]
		if down:
			for node in new_nodes:
				safe_add_parent(self, node)
				self.children.append(node)
		else:
			for node in new_nodes:
				safe_add_parent(node, self)
				node.children.append(self)
		return new_nodes

	def split_down(self, divisor):
		return self._split(True, divisor)

	def split_up(self, divisor):
		return self._split(False, divisor)

	def _extract(self, down, value):
		# print("extract called")
		if value not in conveyor_speeds:
			raise ValueError(f"Extracted value must be one of {conveyor_speeds}.")
		
		if value > self.value:
			raise ValueError("Cannot extract more than the node's value.")
		
		extracted_node = Node(value)
		overflow_value = self.value - value
		overflow_node = Node(overflow_value)
		
		if down:
			safe_add_parent(self, extracted_node)
			safe_add_parent(self, overflow_node)
			self.children.append(extracted_node)
			self.children.append(overflow_node)
		else:
			safe_add_parent(extracted_node, self)
			safe_add_parent(overflow_node, self)
			extracted_node.children.append(self)
			overflow_node.children.append(self)
		
		return [extracted_node, overflow_node]

	def extract_down(self, value):
		return self._extract(True, value)

	def extract_up(self, value):
		return self._extract(False, value)

	def __add__(self, other):
		if isinstance(other, list) and all(isinstance(node, Node) for node in other):
			return self.merge_down(other)
		elif isinstance(other, Node):
			return self.merge_down([other])		
		raise ValueError("Operand must be a Node")

	def __truediv__(self, divisor):
		if isinstance(divisor, (int, float)):
			return self.split_down(divisor)
		raise ValueError(f"Divisor must be an integer (one of these: {"/".join(allowed_divisions)})")

	def __sub__(self, amount):
		if isinstance(amount, (int, float)):
			return self.extract_down(amount)
		raise ValueError("Amount must be a number")

def _simplify_merge(nodes):
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
				if len(same_value_nodes) == max(allowed_divisions) - 1:
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

def has_seen(sim):
	current_config = tuple(sorted(sim))
		
	if current_config in seen_configs:
		if logging:
			print()
			print(f"Skipping already seen configuration: {current_config}")
		return True
	
	seen_configs.add(current_config)
	return False

def refill_queue(queue, divide_queue, extract_queue, merge_queue):
	queues = [divide_queue, extract_queue, merge_queue]
	while any(queues):
		queue.append(random.choice([q for q in queues if q]).pop(0))

steps = -1
logging = False
seen_configs = set()

def _solve(source_values, target_values):
	global steps

	target_counts = {
		value: target_values.count(value) for value in set(target_values)
	}
	gcd = math.gcd(*target_values)
	node_sources = list(map(lambda value: Node(value), source_values))
	# node_targets = list(map(lambda value: Node(value), target_values))
	# print('\n'.join([str(src) for src in copy]))

	def invalid_value(value):
		return value < gcd or value > conveyor_speed_limit

	if len(node_sources) > 1:
		root = Node(sum(source_values))
		root.children = node_sources
		for child in root.children:
			safe_add_parent(root, child)

	queue = [node_sources]
	extract_queue = []
	divide_queue = []
	merge_queue = []
	solution = None

	while queue:
		start_total = time.time()
		sources = queue.pop(0)

		steps -= 1
		if steps + 1 == 0:
			print("stopping")
			print_timings()
			exit(0)
		if (-steps) % 1000 == 0:
			print(f"step {abs(steps)}")

		sources = sort_nodes(sources)
		
		def _copy_sources():
			copy = sources[0].get_root()._deepcopy()
			return list(map(lambda src: copy.find(src.node_id), sources))

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
				# 		safe_add_parent(src, child)

				new_solution = sources[0].get_root()
				new_solution._compute_size()
				if solution is None or new_solution.size < solution.size:
					solution = new_solution
				solution.compute_depth_informations()
				print(f"one solution found of size {solution.size}")
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

		def _get_sim_without(value):
			nonlocal sources
			sim = []
			found = False
			for src in sources:
				if src.value == value and not found:
					found = True
				else:
					sim.append(src.value)
			return sim

		def get_sim_without(value):
			return time_block("get_sim_without", _get_sim_without, value)

		def _try_extract(i):
			nonlocal sources
			src = sources[i]
			sim = None
			parent_values = get_values(src.parents)
			for speed in conveyor_speeds:
				
				# if so then it would have been better to leave it as is
				# and merge all the other values to get the overflow value
				# we would get by exctracting speed amount
				if speed in parent_values: return
				
				if solution:
					if not sources[0].size: sources[0].compute_size()
					if sources[0].size + 2 >= solution.size: continue
				
				if src.value <= speed: break
				
				extracted_value = src.value - speed
				overflow_value = src.value - extracted_value
				if extracted_value < gcd or overflow_value < gcd: continue

				sim = sim if sim else get_sim_without(src.value)
				if has_seen(sim + [extracted_value, overflow_value]): continue
				
				copy = copy_sources()
				src_copy = copy[i]
				pop(src_copy, copy)
				extract_queue.append(copy + (src_copy - speed))

		def try_extract(i):
			time_block("try_extract", _try_extract, i)

		def _try_divide(i):
			nonlocal sources
			src = sources[i]
			sim = None
			for divisor in allowed_divisions:

				if not src.can_split(divisor): continue

				if sum(get_values(src.parents)) == src.value and len(src.parents) == divisor: return

				if solution:
					if not sources[0].size: sources[0].compute_size()
					if sources[0].size + divisor >= solution.size: continue
				
				divided_value = int(src.value / divisor)
				if divided_value < gcd: continue

				sim = sim if sim else get_sim_without(src.value)
				if has_seen(sim + [divided_value] * divisor): continue
				
				copy = copy_sources()
				src_copy = copy[i]
				pop(src_copy, copy)
				divide_queue.append(copy + (src_copy / divisor))

		def try_divide(i):
			time_block("try_divide", _try_divide, i)

		def _try_merge(sources, flags):
			nonlocal cant_use
			
			to_sum_count = sum(flags)
			if to_sum_count <= 1: return
			if solution and n - to_sum_count + 1 >= solution.size: return

			to_not_sum_indices = []
			i = 0
			while not flags[i]:
				to_not_sum_indices.append(i)
				i += 1
			src = sources[i]
			if cant_use[src.value]: return
			to_sum_indices = [i]

			# if len(src.parents) == 0:
			# 	print("\nimpossible case reached, 0 parent while trying to merge")
			# 	print(src)
			# 	exit(1)
			parent = src.parents[0]
			same_parent = len(src.parents) == 1
			while i < n - 1:
				i += 1
				if not flags[i]:
					to_not_sum_indices.append(i)
					continue
				src = sources[i]
				if cant_use[src.value]: return
				# if len(src.parents) == 0:
				# 	print("\nimpossible case reached, 0 parent while trying to merge")
				# 	print(src)
				# 	exit(1)
				if len(src.parents) != 1 or not src.parents[0] is parent:
					same_parent = False
				to_sum_indices.append(i)
			if same_parent and to_sum_count == len(src.parents[0].children):
				# if all(d != to_sum_count for d in allowed_divisions):
				# 	print("\nimpossible case reached, detected a split of neither 2 or 3")
				# 	print(sources)
				# 	exit(1)
				return

			# if to_sum_count != len(to_sum_indices):
			# 	print("wtf")
			# 	exit(1)

			summed_value = sum(sources[i].value for i in to_sum_indices)
			if invalid_value(summed_value): return
			sim = [sources[i].value for i in to_not_sum_indices] + [summed_value]
			if has_seen(sim): return

			copy = copy_sources()
			to_sum = [copy[i] for i in to_sum_indices]
			list(map(lambda src: pop(src, copy), to_sum))
			merge_queue.append(copy + [to_sum[0] + to_sum[1:]])

		def try_merge(sources, flags):
			time_block("try_merge", _try_merge, sources, flags)

		for i in range(n):
			if cant_use[sources[i].value]: continue
			try_divide(i)
			try_extract(i)

		if n >= 2:
			binary_start = 3
			max_num = (1 << n) - 1 # 2^n - 1
			binary = [False] * n
			binary[0] = True
			binary[1] = True
			for _ in range(binary_start, max_num + 1):
				try_merge(sources, binary)
				increment(binary)
		
		refill_queue(queue, divide_queue, extract_queue, merge_queue)
		timings["total"] += time.time() - start_total
	
	return solution

def solve(source_values, target_values):
	sources_total = sum(source_values)
	targets_total = sum(target_values)
	if sources_total > targets_total:
		target_values.append(sources_total - sources_total)
	elif sources_total < targets_total:
		raise ValueError("the sum of targets is greater than the sum of sources")
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
	sol.visualize()

def test():
	ids = get_nodes_id([Node(20) for _ in range(3)])
	print(ids[0] == ids[1] and ids[1] == ids[2])
	exit(0)

if __name__ == '__main__':
	# test()
	main()