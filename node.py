import uuid, tempfile, traceback, io, sys, pathlib, os
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph

from utils import get_node_values, get_short_node_ids
from config import config

if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

class Node:
	def __init__(self, value, node_id=None):
		if value < 0: raise ValueError("negative value")
		self.value = value
		self.node_id = node_id if node_id is not None else str(uuid.uuid4())
		# self.depth = None
		self.size = None
		self.tree_height = None
		self.level = None
		self.parents = []
		self.children = []

	def to_string(self, short_repr=config.short_repr):
		if short_repr: return f"{self.value}({self.node_id[-3:]})"
		r = "Node("
		r += f"value={self.value}, "
		r += f"short_node_id={self.node_id[-3:]}, "
		if config.include_depth_informations:
			# r += f"depth={self.depth}, "
			r += f"size={self.size}, "
			r += f"tree_height={self.tree_height}, "
			r += f"level={self.level}, "
		r += f"parents={get_short_node_ids(self.parents)})"
		return r

	def str(self, stack):
		depth = len(stack)
		stack.append(self)
		r = ""
		for i in range(depth):
			if stack[i + 1].node_id != self.node_id:
				r += (" " if stack[i].children and stack[i].children[-1].node_id == stack[i + 1].node_id else "│") + "  "
			else:
				r += ("└" if stack[i].children[-1].node_id == self.node_id else "├") + "─►"
		r += self.to_string() + "\n"
		for child in self.children: r += child.str(stack)
		stack.pop()
		return r

	def __repr__(self):
		stack = [self]
		r = self.to_string() + "\n"
		for child in self.children: r += child.str(stack)
		return r[:-1] # ignore last \n

	def get_root(self):
		cur = self
		while cur.parents:
			cur = cur.parents[0]
		return cur

	def _deepcopy(self, copied_nodes, leaves):
		stack = [(self, None)]
		while stack:
			current_node, parent_node = stack.pop()
			if current_node.node_id in copied_nodes:
				new_node = copied_nodes[current_node.node_id]
			else:
				new_node = Node(current_node.value, node_id=current_node.node_id)
				new_node.size = current_node.size
				new_node.tree_height = current_node.tree_height
				new_node.level = current_node.level
				copied_nodes[current_node.node_id] = new_node
				if not current_node.children:
					leaves.append(new_node)
				else:
					stack.extend([(child, new_node) for child in current_node.children])
			if parent_node:
				new_node.parents.append(parent_node)
				parent_node.children.append(new_node)
		return new_node

	def deepcopy(self):
		leaves = []
		r = self._deepcopy({}, leaves)
		return r, leaves

	def compute_size(self, trim_root):
		queue = [self]
		visited = set()
		self.size = -1 if trim_root else 0
		while queue:
			cur = queue.pop()
			if cur.node_id in visited: continue
			visited.add(cur.node_id)
			self.size += 1
			for child in cur.children:
				queue.append(child)

	def compute_levels(self):
		stack = [(self, 0)]
		visited = set()
		nodes = []
		while stack:
			node, state = stack.pop()
			if state == 0:
				if node.node_id not in visited:
					visited.add(node.node_id)
					stack.append((node, 1))
					for child in node.children:
						stack.append((child, 0))
			else:
				if node.children:
					max_children_tree_height = max(child.tree_height for child in node.children)
					node.tree_height = max_children_tree_height + 1
					node.level = - max_children_tree_height
				else:
					node.tree_height = 1
					node.level = 0
				nodes.append(node)
		for node in nodes:
			node.level += self.tree_height

	def populate(self, G):
		G.add_node(self.node_id, label=str(self.value), level=self.level)
		for child in self.children:
			G.add_edge(self.node_id, child.node_id)
			child.populate(G)

	def visualize(self, filename, trim_root):
		try:
			G = nx.DiGraph()
			if trim_root and self.children:
				for child in self.children:
					child.populate(G)
			else:
				self.populate(G)

			A = to_agraph(G)
			for node in A.nodes():
				level = G.nodes[node]["level"]
				A.get_node(node).attr["rank"] = f"{level}"

			for level in set(nx.get_node_attributes(G, "level").values()):
				A.add_subgraph(
					[n for n, attr in G.nodes(data=True) if attr["level"] == level],
					rank="same"
				)

			# Invert colors
			A.graph_attr["bgcolor"] = "black"
			for node in A.nodes():
				node.attr["color"] = "white"
				node.attr["fontcolor"] = "white"
				node.attr["style"] = "filled"
				node.attr["fillcolor"] = "black"

			for edge in A.edges():
				edge.attr["color"] = "white"

			print(f"\nGenerating {filename}...", end="")
			A.layout(prog="dot")
			img_stream = io.BytesIO()
			A.draw(img_stream, format=config.solutions_filename_extension)
			img_stream.seek(0)
			filepath = f"{filename}.{config.solutions_filename_extension}"
			with open(filepath, "wb") as f:
				f.write(img_stream.getvalue())
			print(f"done, solution saved at '{filepath}'")
		except Exception as e:
			print(traceback.format_exc(), end="")
			return

	def merge_down(self, other):
		new_value = self.value + sum(get_node_values(other))
		new_node = Node(new_value)
		self.children.append(new_node)
		new_node.parents.append(self)
		for node in other:
			node.children.append(new_node)
			new_node.parents.append(node)
		return new_node

	def merge_up(self, other):
		new_value = self.value + sum(get_node_values(other))
		new_node = Node(new_value)
		self.parents.append(new_node)
		new_node.children.append(self)
		for node in other:
			node.parents.append(new_node)
			new_node.children.append(node)
		return new_node

	def can_split(self, divisor):
		if not divisor in config.allowed_divisors: return False
		return self.value % divisor == 0

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
		raise ValueError("Operand must be a Node[] or Node")

	def __truediv__(self, divisor):
		if isinstance(divisor, (int, float)):
			return self.split_down(divisor)
		raise ValueError(f"Divisor must be an integer (one of these: {"/".join(config.allowed_divisors)})")

	def __sub__(self, amount):
		if isinstance(amount, (int, float)):
			return self.extract_down(amount)
		raise ValueError("Amount must be a number")

	# graveyard

	# def get_leaves(self):
	# 	if not self.children:
	# 		return [self]
	# 	leaves = []
	# 	for child in self.children:
	# 		leaves.extend(child.get_leaves())
	# 	return leaves

	# def find(self, node_id):
	# 	if self.node_id == node_id: return self
	# 	for child in self.children:
	# 		result = child.find(node_id)
	# 		if result: return result
	# 	return None

	# def has_parent(self, parent):
	# 	for p in self.parents:
	# 		if p is parent or p.has_parent(parent): return True
	# 	return False

	# def compute_depth(self):
	# 	queue = [(self, 1)]
	# 	while queue:
	# 		cur, depth = queue.pop(0)
	# 		cur.depth = depth
	# 		for child in self.children: queue.append((child, depth + 1))

	# def _compute_depth_and_tree_height(self):
	# 	self.depth = 1 + (max(parent.depth for parent in self.parents) if self.parents else 0)
	# 	max_child_tree_height = 0
	# 	for child in self.children:
	# 		_, child_tree_height = child._compute_depth_and_tree_height()
	# 		if child_tree_height > max_child_tree_height:
	# 			max_child_tree_height = child_tree_height
	# 	self.tree_height = max_child_tree_height + 1
	# 	return self.depth, self.tree_height