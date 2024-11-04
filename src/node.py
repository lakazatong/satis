import uuid

from treelike import TreeLike
from utils import get_node_values, get_short_node_ids
from fastList import FastList
from config import config
from fractions import Fraction

class Node(TreeLike):
	@property
	def children(self):
		return self._children

	@children.setter
	def children(self, value: list['TreeLike']):
		self._children = value
	
	def __init__(self, value, parent_past=None, node_id=None):
		if value < 0: raise ValueError("negative value")
		super().__init__()
		if not isinstance(value, (Fraction, int)): raise ValueError(f"not Fraction / int ({type(value)})")
		self.value = value
		self.node_id = node_id if node_id is not None else str(uuid.uuid4())
		self.level = None
		self.past = FastList(value)
		if parent_past: self.past.extend(parent_past)
		self.parents = []
		self._children = []
		
		if config.short_repr:
			self.repr_keys = False
			self.repr_whitelist.add('node_id')
		else:
			self.repr_keys = True
			self.repr_whitelist.update(('value', 'node_id', 'parents'))
			if config.include_level_in_logs:
				self.repr_whitelist.add('level')

	def repr_self(self):
		return str(self.value) if config.short_repr else 'Node'

	def repr_node_id(self):
		return 'node_id', self.node_id[-3:]

	def repr_parents(self):
		return 'parents', get_short_node_ids(self.parents)

	def __eq__(self, other):
		if not isinstance(other, Node):
			return NotImplemented
		return self.node_id == other.node_id

	def _deepcopy(self, copied_nodes):
		stack = [(self, None)]
		while stack:
			node, parent = stack.pop()
			
			if node.node_id in copied_nodes:
				new_node = copied_nodes[node.node_id]
				if parent: new_node.parents.append(parent)
			else:
				new_node = Node(node.value, node_id=node.node_id)
				new_node.level = node.level
				new_node.past.extend(self.past)
				copied_nodes[node.node_id] = new_node
				new_node.parents = [parent] if parent else []
				stack.extend((child, new_node) for child in node.children)
			
			if parent: parent.children.append(new_node)
		return copied_nodes[self.node_id]

	def deepcopy(self):
		return self._deepcopy({})

	def populate(self, G, seen_ids):
		seen_ids.add(self.node_id)
		G.add_node(self.node_id, label=str(self.value), level=self.level)
		for child in self._children:
			G.add_edge(self.node_id, child.node_id)
			if child.node_id not in seen_ids: child.populate(G, seen_ids)

	def extract(self, value):
		overflow_value = self.value - value
		new_nodes = [Node(value, self.past) for value in sorted([value, overflow_value])]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes

	def divide(self, divisor):
		divided_value = Fraction(self.value, divisor)
		new_nodes = [Node(divided_value, self.past) for _ in range(divisor)]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes
	
	def divide_loop(self, conveyor_speed):
		new_value = Fraction(conveyor_speed, 3)
		overflow_value = self.value - new_value * 2
		new_nodes = [Node(value, self.past) for value in sorted([new_value, new_value, overflow_value])]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes

	@staticmethod
	def merge(nodes):
		summed_value = sum(node.value for node in nodes)
		new_node = Node(summed_value)
		for node in nodes:
			node.children.append(new_node)
			new_node.parents.append(node)
			new_node.past.extend(node.past)
		return new_node

	def simplify_info(self):
		original_children_ids = {child.node_id for child in self._children}
		stack, deepest_node, max_depth = [(grandchild, 0) for child in self._children for grandchild in child.children], None, -1
		while stack:
			node, depth = stack.pop()
			if original_children_ids.issubset(node.reachable_from):
				if depth > max_depth:
					deepest_node, max_depth = node, depth
			for child in node.children:
				stack.append((child, depth + 1))
		return deepest_node

	# graveyard

	# def get_root(self):
	# 	cur = self
	# 	while cur.parents:
	# 		cur = cur.parents[0]
	# 	return cur

	# def get_leaves(self):
	# 	if not self._children:
	# 		return [self]
	# 	leaves = []
	# 	for child in self._children:
	# 		leaves.extend(child.get_leaves())
	# 	return leaves

	# def find(self, node_id):
	# 	if self.node_id == node_id: return self
	# 	for child in self._children:
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
	# 		for child in self._children: queue.append((child, depth + 1))

	# def _compute_depth_and_tree_height(self):
	# 	self.depth = 1 + (max(parent.depth for parent in self.parents) if self.parents else 0)
	# 	max_child_tree_height = 0
	# 	for child in self._children:
	# 		_, child_tree_height = child._compute_depth_and_tree_height()
	# 		if child_tree_height > max_child_tree_height:
	# 			max_child_tree_height = child_tree_height
	# 	self.tree_height = max_child_tree_height + 1
	# 	return self.depth, self.tree_height

		# def compute_size(self):
	# 	queue = [self]
	# 	visited = set()
	# 	self.size = 0
	# 	while queue:
	# 		cur = queue.pop()
	# 		if cur.node_id in visited: continue
	# 		visited.add(cur.node_id)
	# 		self.size += 1
	# 		for child in cur.children:
	# 			queue.append(child)

	# def compute_levels(self):
	# 	stack = [(self, 0)]
	# 	visited = set()
	# 	nodes = []
	# 	while stack:
	# 		node, state = stack.pop()
	# 		if state == 0:
	# 			if node.node_id not in visited:
	# 				visited.add(node.node_id)
	# 				stack.append((node, 1))
	# 				for child in node.children:
	# 					stack.append((child, 0))
	# 		else:
	# 			if node.children:
	# 				max_children_tree_height = max(child.tree_height for child in node.children)
	# 				node.tree_height = max_children_tree_height + 1
	# 				node.level = - max_children_tree_height
	# 			else:
	# 				node.tree_height = 1
	# 				node.level = 0
	# 			nodes.append(node)
	# 	for node in nodes:
	# 		node.level += self.tree_height

	# def extract_up(self, value):
	# 	extracted_node = Node(value)
	# 	overflow_value = self.value - value
	# 	overflow_node = Node(overflow_value)
	# 	self.parents.append(extracted_node)
	# 	self.parents.append(overflow_node)
	# 	extracted_node.children.append(self)
	# 	overflow_node.children.append(self)
	# 	return [extracted_node, overflow_node]

	# def split_up(self, divisor):
	# 	new_value = int(self.value / divisor)
	# 	new_nodes = [Node(new_value) for _ in range(divisor)]
	# 	for node in new_nodes:
	# 		self.parents.append(node)
	# 		node.children.append(self)
	# 	return new_nodes

	# def merge_up(self, other):
	# 	new_value = self.value + sum(get_node_values(other))
	# 	new_node = Node(new_value)
	# 	self.parents.append(new_node)
	# 	new_node.children.append(self)
	# 	for node in other:
	# 		node.parents.append(new_node)
	# 		new_node.children.append(node)
	# 	return new_node