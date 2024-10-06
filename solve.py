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

conveyor_speeds = [60, 120, 270, 480, 780, 1200]
conveyor_speeds_r = sorted(conveyor_speeds, reverse=True)

def visualize(src, nodes):
	src.children = nodes
	src.visualize()

class Node:
	def __init__(self, value):
		self.value = value
		self.children = []
		self.depth = None
		self.tree_height = None
		self.level = None
		self.id = str(uuid.uuid4())

	def __repr__(self):
		r = f"{"\t" * (self.depth - 1)}Node(value={self.value}, depth={self.depth}, tree_height={self.tree_height}, level={self.level}, children=["
		if self.children:
			for child in self.children:
				r += "\n" + str(child)
			r += "\n" + "\t" * (self.depth - 1) + "])"
		else:
			r += "])"
		return r

	def compute_depth_and_tree_height(self, parent=None):
		self.depth = 1 + parent.depth if parent else 1
		for child in self.children:
			child_depth = child.compute_depth_and_tree_height(self)
		self.tree_height = max(map(lambda node: node.tree_height, self.children)) + 1 if self.children else 1
		return self.depth

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
		self.compute_depth_and_tree_height()
		# self.set_max_tree_height(self.tree_height)
		self.compute_level(self.tree_height)

	def add_edges(self, G):
		G.add_node(self.id, label=str(self.value), level=self.level)  # Include level in attributes
		for child in self.children:
			G.add_edge(self.id, child.id)
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

	def merge(self, down, *other):
		new_value = self.value + sum(node.value for node in other)
		new_node = Node(new_value)
		if down:
			self.children.append(new_node)
			for node in other:
				node.children.append(new_node)
		else:
			new_node.children.append(self)
			for node in other:
				new_node.children.append(node)
		return new_node

	def can_split(self, divisor):
		if divisor not in (2, 3): return False
		new_value = self.value / divisor
		if new_value.is_integer(): return True
		return False

	def split(self, down, divisor):
		if not self.can_split(divisor): return []
		new_value = int(self.value / divisor)
		new_nodes = [Node(new_value) for _ in range(divisor)]
		if down:
			self.children.extend(new_nodes)
		else:
			for node in new_nodes:
				node.children.append(self)
		return new_nodes

	def extract(self, down, value):
		if value not in conveyor_speeds:
			raise ValueError(f"Extracted value must be one of {conveyor_speeds}.")
		
		if value > self.value:
			raise ValueError("Cannot extract more than the node's value.")
		
		extracted_node = Node(value)
		overflow_value = self.value - value
		overflow_node = Node(overflow_value)
		
		if down:
			self.children.append(extracted_node)
			self.children.append(overflow_node)
		else:
			extracted_node.children.append(self)
			overflow_node.children.append(self)
		
		return extracted_node, overflow_node

	def __add__(self, *other):
		if all(isinstance(o, Node) for o in other):
			return self.merge(True, *other)
		raise ValueError("Operand must be a Node")

	def __truediv__(self, divisor):
		if isinstance(divisor, (int, float)):
			return self.split(divisor)
		raise ValueError("Divisor must be an integer (2 or 3)")

	def __sub__(self, amount):
		if isinstance(amount, (int, float)):
			return self.extract(amount)
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
				if len(same_value_nodes) == 2:
					break
				same_value_nodes.append(nodes[i])
				i += 1

			if len(same_value_nodes) > 0:
				merged_node = current_node.merge(False, *same_value_nodes)
				merged_nodes.append(merged_node)
				done = False
				has_merged = True
			else:
				merged_nodes.append(current_node)

		if done: break

		merged_nodes = sorted(merged_nodes, key=lambda node: node.value)
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
				extracted_node, overflow_node = node.extract(False, speed)
				extracted_nodes.append(extracted_node)
				extracted_nodes.append(overflow_node)
				extracted_flag = True
				break
		if not extracted_flag:
			extracted_nodes.append(node)

	nodes = sorted(extracted_nodes, key=lambda node: node.value)
	return nodes

def simplify(nodes):
	nodes, has_merged = _simplify_merge(nodes)
	nodes = _simplify_extract(nodes)
	while has_merged:
		nodes, has_merged = _simplify_merge(nodes)
		if not has_merged: break
		nodes = _simplify_extract(nodes)
	return nodes

def _solve(sources, targets):
	if len(sources) == 1:
		# can't merge
		src = sources[0]
		if src.can_split(3):
			if src.can_split(2):
				return [src]
			else:
				return [src]
		else:
			if src.can_split(2):
				return [src]
			else:
				# can't split
				# must extract
				return [src]
	return sources

def solve(src, targets):
	total = sum(targets)
	if src > total:
		targets.append(src - total)
	elif src < total:
		raise ValueError("the sum of targets is greater than the source", targets)
	src = Node(src)
	targets = simplify(list(map(lambda target: Node(target), sorted(targets))))
	visualize(src, targets)
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
	# for r in res:
	# 	r.visualize()

if __name__ == '__main__':
	main()