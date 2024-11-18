import uuid

from treelike import TreeLike
from utils import get_node_values, get_short_node_ids
from fastList import FastList
from config import config

class Node(TreeLike):
	@property
	def children(self):
		return self._children

	@children.setter
	def children(self, value: list['TreeLike']):
		self._children = value
	
	def __init__(self, value, parent_past=None, node_id=None, level=None):
		if value < 0: raise ValueError("negative value")
		super().__init__()
		if not isinstance(value, int): raise ValueError(f"not int ({type(value)} {value})")
		self.value = value
		self.node_id = node_id if node_id is not None else str(uuid.uuid4())
		self.level = level
		self.past = FastList(value)
		if parent_past: self.past.extend(parent_past)
		self.parents = []
		self._children = []
		self._expands = []
		self.levels_to_add = 0
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
				new_node._expands = [_expand for _expand in node._expands]
				copied_nodes[node.node_id] = new_node
				new_node.parents = [parent] if parent else []
				stack.extend((child, new_node) for child in node.children)
			
			if parent: parent.children.append(new_node)
		return copied_nodes[self.node_id]

	def deepcopy(self):
		return self._deepcopy({})

	def populate(self, G, seen_ids, unit_flow_ratio):
		from fractions import Fraction
		seen_ids.add(self.node_id)
		G.add_node(self.node_id, label=str(Fraction(self.value, unit_flow_ratio)) + ", " + str(self.level), level=self.level)
		for child in self._children:
			G.add_edge(self.node_id, child.node_id)
			if child.node_id not in seen_ids: child.populate(G, seen_ids, unit_flow_ratio)

	def extract(self, value):
		if value not in config.conveyor_speeds:
			self._expands.append((Node.expand_extract, tuple()))
		overflow_value = self.value - value
		new_nodes = [Node(value, self.past) for value in sorted([value, overflow_value])]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes

	def divide(self, divisor):
		if divisor != 2 and divisor != 3:
			self._expands.append((Node.expand_divide, (divisor,)))
		divided_value = self.value // divisor
		new_nodes = [Node(divided_value, self.past) for _ in range(divisor)]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes

	def split(self, conveyor_speed):
		if conveyor_speed not in config.conveyor_speeds:
			print("impossible case reached, splitting a non conveyor speed")
			exit(1)
		new_value = conveyor_speed // 3
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
		n = len(nodes)
		if n <= 1:
			print("impossible case reached, merging 0 or 1 node")
			exit(1)
		if n > 3:
			new_node._expands.append((Node.expand_merge, tuple()))
		for node in nodes:
			node.children.append(new_node)
			new_node.parents.append(node)
			new_node.past.extend(node.past)
		return new_node

	def _min_level(self, seen_ids):
		seen_ids.add(self.node_id)
		r = self.level
		for child in self.children:
			if child.node_id in seen_ids: continue
			r = min(r, child._min_level(seen_ids))
		return r

	def min_level(self):
		seen_ids = set
		return self._min_level(seen_ids)

	def _apply_levels_update(self, seen_ids):
		seen_ids.add(self.node_id)
		self.level += self.levels_to_add
		for child in self.children:
			if child.node_id in seen_ids: continue
			child._apply_levels_update(seen_ids)

	def apply_levels_update(self):
		seen_ids = set()
		self._apply_levels_update(seen_ids)

	def _tag_levels_update(self, threshold, amount, seen_ids):
		seen_ids.add(self.node_id)
		if self.level >= threshold:
			self.levels_to_add += amount
		for child in self.children:
			if child.node_id in seen_ids: continue
			child._tag_levels_update(threshold, amount, seen_ids)

	def tag_levels_update(self, threshold, amount):
		seen_ids = set()
		self._tag_levels_update(threshold, amount, seen_ids)

	@staticmethod
	def expand_split(node):
		return 0, 0

	@staticmethod
	def expand_extract(node):
		return 0, 0

	@staticmethod
	def expand_divide(node, d):
		from utils import find_n_m_l, compute_branches_count, compute_looping_branches
		n, m, l, n_splitters = find_n_m_l(d)
		branches_count = compute_branches_count(n, m)
		looping_branches = compute_looping_branches(n, m, l, branches_count)
		values = [node.value // d]
		for _ in range(m): values.append(values[-1] * 3)
		for _ in range(n): values.append(values[-1] * 2)
		values = [x for x in reversed(values)]
		print(f"{n = }\n{m = }\n{l = }\n{n_splitters = }\n{looping_branches = }\n{values = }")
		merged_node = Node(values[0])
		merged_node.parents = [node]
		original_children = node.children
		node.children = [merged_node]
		cur_level = node.level
		cur_nodes = [merged_node]
		new_nodes = []
		
		for i in range(1, n+1):
			cur_level += 1
			n_looping_branches, n_ignore_branches = looping_branches.get((i, 0), (0, 0))
			total_to_ignore = n_looping_branches + n_ignore_branches
			if i < n or m != 0:
				for _ in range(2**i - total_to_ignore):
					new_nodes.append(Node(values[i], level=cur_level))
			else:
				new_nodes = original_children

			for j in range(len(cur_nodes)-1):
				cur = cur_nodes[j]
				for k in range(2*j, 2*(j+1)):
					child = new_nodes[k]
					cur.children.append(child)
					child.parents = [cur]

			last = cur_nodes[-1]
			j = len(cur_nodes) - 1
			for k in range(2*j, 2*(j+1) - n_looping_branches):
				child = new_nodes[k]
				last.children.append(child)
				child.parents = [last]
			
			for _ in range(n_looping_branches):
				last.children.append(merged_node)
				merged_node.parents.append(last)
			
			cur_nodes, new_nodes = new_nodes, []
		
		for i in range(1, m+1):
			cur_level += 1
			n_looping_branches, n_ignore_branches = looping_branches.get((n, i), (0, 0))
			total_to_ignore = n_looping_branches + n_ignore_branches
			if i < m:
				for _ in range(2**n*3**i - total_to_ignore):
					new_nodes.append(Node(values[n + i], level=cur_level))
			else:
				new_nodes = original_children

			# print(f"\n{i = }\n{n_looping_branches = }\n{n_ignore_branches = }\n{total_to_ignore = }\n{cur_nodes = } {len(cur_nodes)}\n{new_nodes = } {len(new_nodes)}")

			for j in range(len(cur_nodes)-1):
				cur = cur_nodes[j]
				for k in range(3*j, 3*(j+1)):
					child = new_nodes[k]
					cur.children.append(child)
					child.parents = [cur]

			last = cur_nodes[-1]
			j = len(cur_nodes) - 1
			for k in range(3*j, 3*(j+1) - n_looping_branches):
				child = new_nodes[k]
				last.children.append(child)
				child.parents = [last]
			
			for _ in range(n_looping_branches):
				last.children.append(merged_node)
				merged_node.parents.append(last)

			cur_nodes, new_nodes = new_nodes, []

		cur_level += 1
		return cur_nodes[0].level, cur_level - node.level

	@staticmethod
	def expand_merge(node):
		cur_level = node.level
		nodes_to_merge = node.parents
		for n in nodes_to_merge:
			n.children = []
		while len(nodes_to_merge) > 3:
			merged_node = Node.merge([nodes_to_merge.pop(), nodes_to_merge.pop(), nodes_to_merge.pop()])
			merged_node.level = cur_level
			nodes_to_merge.append(merged_node)
			cur_level += 1
		for n in nodes_to_merge:
			n.children = [node]
		node.parents = nodes_to_merge
		# all nodes with level >= node.level must have their levels increased by cur_level - node.level
		return node.level, cur_level - node.level

	def expand(self, seen_ids):
		seen_ids.add(self.node_id)
		levels_updates = []
		for _expand, args in self._expands:
			levels_updates.append(_expand(self, *args))
		for child in self.children:
			if child.node_id not in seen_ids:
				levels_updates.extend(child.expand(seen_ids))
		return levels_updates

	# graveyard

	# def simplify_info(self):
	# 	original_children_ids = {child.node_id for child in self._children}
	# 	stack, deepest_node, max_depth = [(grandchild, 0) for child in self._children for grandchild in child.children], None, -1
	# 	while stack:
	# 		node, depth = stack.pop()
	# 		if original_children_ids.issubset(node.reachable_from):
	# 			if depth > max_depth:
	# 				deepest_node, max_depth = node, depth
	# 		for child in node.children:
	# 			stack.append((child, depth + 1))
	# 	return deepest_node

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