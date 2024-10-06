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

timings = {
	"total": 0,
	"copy_sources": 0,
	"try_divide": 0,
	"try_extract": 0,
	"try_merge": 0,
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
	timings[key] += (time.time() - start_time)
	return result

allowed_divisions = [3, 2]
conveyor_speeds = [60, 120, 270, 480, 780, 1200]
conveyor_speeds_r = sorted(conveyor_speeds, reverse=True)
short_repr = False

def safe_add_parent(parent, node):
	if node is parent:
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

class Node:
	def __init__(self, value, node_id=None):
		self.value = value
		self.node_id = node_id if node_id is not None else str(uuid.uuid4())
		self.depth = -1
		self.tree_height = -1
		self.level = -1
		self.size = None
		self.parents = []
		self.children = []

	def __repr__(self):
		if short_repr:
			return f"{"\t" * (self.depth - 1)}{self.value}({self.node_id[-3:]})"
		r = f"{"\t" * (self.depth - 1)}Node(value={self.value}, short_node_id={self.node_id[-3:]}, depth={self.depth}, tree_height={self.tree_height}, level={self.level}, children=["
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
		self.size = 1 + sum(child._compute_size() for child in self.children)
		return self.size

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
		G.add_node(self.node_id, label=str(self.value), level=self.level)  # Include level in attributes
		for child in self.children:
			G.add_edge(self.node_id, child.node_id)
			child.add_edges(G)

	def visualize(self):
		self.compute_depth_informations()
		print(self)
		G = nx.DiGraph()
		self.add_edges(G)

		A = to_agraph(G)

		# Enforce levels as ranks in Graphviz
		for node in G.nodes:
			level = G.nodes[node]['level']
			A.get_node(node).attr['rank'] = f'{level}'

		# Create subgraphs for nodes with the same level
		for level in set(nx.get_node_attributes(G, 'level').values()):
			subgraph = A.add_subgraph(
				[n for n, attr in G.nodes(data=True) if attr['level'] == level],
				rank='same'
			)

		A.graph_attr['rankdir'] = 'TB'  # Bottom to Top layout

		# Render the graph
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
				merged_node = current_node.merge_up(*same_value_nodes)
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

steps = -1
logging = False
seen_configs = set()

def has_seen(sim):
	current_config = tuple(sorted(sim))
		
	if current_config in seen_configs:
		if logging:
			print()
			print(f"Skipping already seen configuration: {current_config}")
		return True
	
	seen_configs.add(current_config)
	return False

def _solve(initial_sources, target_infos):
	global steps

	queue = [initial_sources]
	solution = None
	solution_size = 0
	target_values = target_infos["values"]
	target_counts = target_infos["counts"]
	gcd = target_infos["gcd"]

	while queue:
		start_total = time.time()
		sources = queue.pop(0)

		steps -= 1
		if steps + 1 == 0:
			print("stopping")
			exit(0)
		if (-steps) % 1000 == 0:
			print(f"still trying... (step {-steps})")

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
				print("one solution found", sources, target_values)
				new_solution = sources[0].get_root()
				new_solution._compute_size()
				if solution is None or new_solution.size < solution.size:
					solution = new_solution

		source_counts = {}
		for src in sources:
			if src.value in source_counts:
				source_counts[src.value] += 1
			else:
				source_counts[src.value] = 1

		can_use = {}
		for src in sources:
			value = src.value
			src_count = source_counts.get(value, None)
			target_count = target_counts.get(value, None)
			can_use[value] = max(0, src_count - target_count) > 0 if src_count and target_count else True

		def get_sim_without(value):
			nonlocal sources
			sim = []
			i = 0
			while i < n:
				src = sources[i]
				if src.value == value: break
				sim.append(src.value)
				i += 1
			for j in range(i + 1, n):
				sim.append(sources[j].value)
			return sim

		def _try_divide(src):
			nonlocal sources
			if sum(get_values(src.parents)) == src.value: return
			sim = None
			for divisor in allowed_divisions:
				
				if not src.can_split(divisor): continue
				
				divided_value = int(src.value / divisor)
				if divided_value < gcd: continue

				sim = sim if sim else get_sim_without(src.value)
				if has_seen(sim + [divided_value] * divisor): continue
				
				if solution:
					if not sources[0].size: sources[0].compute_size()
					if sources[0].size + divisor >= solution.size: continue
				
				copy = copy_sources()
				src = copy[i]
				pop(src, copy)
				queue.append(copy + (src / divisor))

		def try_divide(src):
			time_block("try_divide", _try_divide, src)

		def _try_extract(src):
			nonlocal sources
			if sum(get_values(src.parents)) == src.value: return
			sim = None
			for speed in conveyor_speeds:
				
				if src.value <= speed: break
				
				extracted_value = src.value - speed
				overflow_value = src.value - extracted_value
				if extracted_value < gcd or overflow_value < gcd: continue

				sim = sim if sim else get_sim_without(src.value)
				if has_seen(sim + [extracted_value, overflow_value]): continue
				
				if solution:
					if not sources[0].size: sources[0].compute_size()
					if sources[0].size + 2 >= solution.size: continue
				
				copy = copy_sources()
				src = copy[i]
				pop(src, copy)
				queue.append(copy + (src - speed))

		def try_extract(src):
			time_block("try_extract", _try_extract, src)

		def _try_merge(sources, to_sum_indices, to_not_sum_indices):
			nonlocal can_use
			
			if len(to_sum_indices) <= 1: return
			
			src = sources[to_sum_indices[0]]
			
			if can_use[src.value] == 0: return

			if solution:
				simulated_size = len(to_not_sum_indices) + 1
				if simulated_size >= solution.size: return

			if len(src.parents) == 0:
				print("\nimpossible case reached, 0 parent while trying to merge")
				print(src)
				exit(1)
			parent = src.parents[0]
			ignore = False
			same_parent = len(src.parents) == 1
			for i in to_sum_indices[1:]:
				src = sources[i]
				if can_use[src.value] == 0:
					ignore = True
					break
				if len(src.parents) == 0:
					print("\nimpossible case reached, 0 parent while trying to merge")
					print(src)
					exit(1)
				if len(src.parents) != 1 or not src.parents[0] is parent:
					same_parent = False
			if ignore: return
			if same_parent and len(to_sum_indices) == len(src.parents[0].children):
				if all(d != len(to_sum_indices) for d in allowed_divisions):
					print("\nimpossible case reached, detected a split of neither 2 or 3")
					print(sources)
					exit(1)
				return

			summed_value = sum(get_values(sources[i] for i in to_sum_indices))
			sim = [summed_value]
			for i in to_not_sum_indices: sim.append(sources[i].value)
			if has_seen(sim): return

			copy = copy_sources()
			to_sum = [copy[i] for i in to_sum_indices]
			if summed_value < gcd:
				print("impossible case reached, sum of merged nodes is lower than gcd")
				exit(1)
			list(map(lambda src: pop(src, copy), to_sum))
			queue.append(copy + ([to_sum[0] + to_sum[1:]] if len(to_sum) > 0 else []))

		def try_merge(sources, to_sum_indices, to_not_sum_indices):
			time_block("try_merge", _try_merge, sources, to_sum_indices, to_not_sum_indices)
		
		for i in range(n):
			src = sources[i]
			if can_use[src.value] == 0: continue
			try_divide(src)
			try_extract(src)
			for num in range(3, 2**n):
				to_sum_indices, to_not_sum_indices = [], []
				for i in range(n):
					if (num >> i) & 1:
						to_sum_indices.append(i)
					else:
						to_not_sum_indices.append(i)
				try_merge(sources, to_sum_indices, to_not_sum_indices)
		
		timings["total"] += (time.time() - start_total)
	
	return solution

def solve(src, targets):
	total = sum(targets)
	if src > total:
		targets.append(src - total)
	elif src < total:
		raise ValueError("the sum of targets is greater than the source", targets)
	src = Node(src)
	# targets = simplify(list(map(lambda target: Node(target), sorted(targets))))
	targets = sorted(targets)
	# visualize(src, targets)
	return _solve([src], {
		"values": targets,
		"counts": {
			value: targets.count(value) for value in set(targets)
		},
		"gcd": math.gcd(*targets)
	})

def main():
	parser = argparse.ArgumentParser(description='Solve a Satisfactory problem')
	parser.add_argument('src', type=int, help='The source value')
	parser.add_argument('targets', nargs='+', help='Target values, which can be numbers or Nx format')

	args = parser.parse_args()

	src = args.src
	if src % 5 != 0:
		print("Error: all values must be multiples of 5")
		exit(1)
	targets = []

	i = 0
	while i < len(args.targets):
		target = args.targets[i]
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
		if i + 1 == len(args.targets):
			print("Error: You must provide a target value after Nx.")
			exit(1)
		target_value = int(args.targets[i + 1])
		if target_value % 5 != 0:
			print("Error: all values must be multiples of 5")
			exit(1)
		for _ in range(multiplier):
			targets.append(target_value)
		i += 2

	if not targets:
		print("Error: At least one target value must be provided.")
		exit(1)

	sol = solve(src, targets)
	sol.visualize()

def test():
	ids = get_nodes_id([Node(20) for _ in range(3)])
	print(ids[0] == ids[1] and ids[1] == ids[2])
	exit(0)

if __name__ == '__main__':
	# test()
	main()