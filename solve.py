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

allowed_divisions = [3, 2]
conveyor_speeds = [60, 120, 270, 480, 780, 1200]
conveyor_speeds_r = sorted(conveyor_speeds, reverse=True)

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

def get_nodes_id(nodes):
	return list(map(lambda node: node.node_id, nodes))

class Node:
	def __init__(self, value, node_id=None):
		self.value = value
		self.node_id = node_id if node_id is not None else str(uuid.uuid4())
		self.depth = -1
		self.tree_height = -1
		self.level = -1
		self.parents = []
		self.children = []

	def __repr__(self):
		r = f"{"\t" * (self.depth - 1)}Node(value={self.value}, short_node_id={self.node_id[-3:]}, depth={self.depth}, tree_height={self.tree_height}, level={self.level}, children=["
		if self.children:
			for child in self.children:
				r += "\n" + str(child)
			r += "\n" + "\t" * (self.depth - 1) + "])"
		else:
			r += "])"
		return r

	def get_root(self):
		cur = self
		while cur.parents:
			cur = cur.parents[0]
		return cur

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

steps = 0
logging = True
seen_configs = set()

def _solve(sources, targets):
	global steps, seen_configs
	steps -= 1

	# Sort sources once and convert the current configuration to a tuple for duplicate checks
	sources = sort_nodes(sources)
	current_config = tuple(src.value for src in sources)
	
	# Check if this configuration has already been seen
	if current_config in seen_configs:
		if logging:
			print(f"Skipping already seen configuration: {current_config}")
		return []
	
	# Add current configuration to the seen set
	seen_configs.add(current_config)

	if logging:
		print()
		print(sources)
		print("sources' children:")
		for src in sources:
			print(src.value, src.children)
		print("sources' parents:")
		for src in sources:
			print(src.value, get_values(src.parents))

	n = len(sources)

	# Match sources with targets
	if n == len(targets):
		for i in range(n):
			if sources[i].value != targets[i]:
				# Doesn't match targets, consider this a fail
				return []
		# Link the simplified targets' trees with the current one
		# for target in targets:
		# 	for child in target.children:
		# 		child.parents = []
		# for i in range(n):
		# 	src = sources[i]
		# 	src.children = targets[i].children
		# 	for child in src.children:
		# 		safe_add_parent(src, child)
		return [sources[0].get_root()]

	def get_other(i):
		return [src.deepcopy() for src in (sources[:i] + sources[i + 1:])]

	def try_divide(i, r):
		global steps
		src = sources[i]
		if sum(get_values(src.parents)) == src.value: return
		for divisor in allowed_divisions:
			if src.can_split(divisor):
				other = get_other(i)
				if logging: print(f"solving with {src} / {divisor} and {other}")
				if not steps:
					print("stopping")
					exit(0)
				divided = src.deepcopy() / divisor
				r.extend(_solve(divided + other, targets))

	def try_extract(i, r):
		global steps
		src = sources[i]
		if sum(get_values(src.parents)) == src.value: return
		for speed in conveyor_speeds:
			if src.value <= speed: break
			other = get_other(i)
			if logging: print(f"solving with {src} - {speed} and {other}")
			if not steps:
				print("stopping")
				exit(0)
			r.extend(_solve((src.deepcopy() - speed) + other, targets))

	r = []

	for i in range(n):
		try_divide(i, r)
		try_extract(i, r)

	# if n == 1:
	# 	# can't merge
	# 	return r

	for num in range(0 if r else 1, 2**n):
		flags = [
			(
				(num >> i) & 1
				and sum(get_values(sources[i].parents)) != sources[i].value
			)
			for i in range(n)
		]
		to_sum = [sources[i].deepcopy() for i in range(n) if flags[i]]
		other = [sources[i].deepcopy() for i in range(n) if not flags[i]]
		if logging: print(f"solving with {' + '.join(str(ts) for ts in to_sum)} and {other}")
		if not steps:
			print("stopping")
			exit(0)
		r.extend(_solve([to_sum[0] + to_sum[1:]] + other if len(to_sum) > 0 else other, targets))
	
	return r

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
	return _solve([src], targets)

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

	res = solve(src, targets)
	for r in res:
		r.visualize()

def test():
	ids = get_nodes_id([Node(20) for _ in range(3)])
	print(ids[0] == ids[1] and ids[1] == ids[2])
	exit(0)

if __name__ == '__main__':
	# test()
	main()