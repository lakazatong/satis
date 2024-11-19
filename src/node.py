import uuid, traceback, networkx as nx, io

from networkx.drawing.nx_agraph import to_agraph
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
			self.repr_whitelist.add('_expands')
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

	@staticmethod
	def wrap(root_node, path):
		seen_ids = set()

		def traverse(node, output):
			
			if node.node_id in seen_ids: return
			seen_ids.add(node.node_id)
			
			node_tuple = (
				node.value,
				node.node_id,
				node.level,
				[parent.node_id for parent in node.parents],
				[child.node_id for child in node.children],
				[(code, args) for code, _, args in node._expands]
			)
			output.append(node_tuple)
			for child in node.children:
				if child.node_id in seen_ids: continue
				traverse(child, output)
			for parent in node.parents:
				if parent.node_id in seen_ids: continue
				traverse(parent, output)

		output = []
		traverse(root_node, output)
		with open(path, "w+", encoding="utf-8") as f:
			f.write("[\n" + ",\n".join("\t" + str(out) for out in output) + "\n]")

	@staticmethod
	def expand_from_code(code):
		match code:
			case 0: return Node.expand_extract		
			case 1: return Node.expand_divide		
			case 2: return Node.expand_merge		
			case 3: return Node.expand_split		

	@staticmethod
	def unwrap(path):
		import ast
		data = None
		with open(path, "r", encoding="utf-8") as f:
			data = ast.literal_eval(f.read())
		node_map = {}

		for value, node_id, level, *_ in data:
			node_map[node_id] = Node(value, node_id=node_id, level=level)

		for value, node_id, level, parents, children, expands in data:
			node = node_map[node_id]
			node.parents = [node_map[parent_id] for parent_id in parents]
			node.children = [node_map[child_id] for child_id in children]
			node._expands = [(code, Node.expand_from_code(code), args) for code, args in expands]

		return [node for node in node_map.values() if node.level == 0]

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

		if self.node_id in seen_ids: return
		seen_ids.add(self.node_id)
		
		# G.add_node(self.node_id, label=str(Fraction(self.value, unit_flow_ratio)), level=self.level)
		G.add_node(self.node_id, label=str(Fraction(self.value, unit_flow_ratio)) + ", " + self.repr_node_id()[1], level=self.level)
		for child in self._children:
			G.add_edge(self.node_id, child.node_id)
			if child.node_id in seen_ids: continue
			child.populate(G, seen_ids, unit_flow_ratio)

	def extract(self, value):
		if value not in config.conveyor_speeds:
			self._expands.append((0, Node.expand_extract, (value,)))
		overflow_value = self.value - value
		new_nodes = [Node(v, self.past) for v in sorted([value, overflow_value])]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes

	def divide(self, divisor):
		if divisor != 2 and divisor != 3:
			self._expands.append((1, Node.expand_divide, (divisor,)))
		divided_value = self.value // divisor
		new_nodes = [Node(divided_value, self.past) for _ in range(divisor)]
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
			new_node._expands.append((2, Node.expand_merge, tuple()))
		for node in nodes:
			node.children.append(new_node)
			new_node.parents.append(node)
			new_node.past.extend(node.past)
		return new_node

	def split(self, conveyor_speed):
		if conveyor_speed not in config.conveyor_speeds:
			print("impossible case reached, splitting a non conveyor speed")
			exit(1)
		new_value = conveyor_speed // 3
		overflow_value = self.value - new_value * 2
		new_nodes = [Node(v, self.past) for v in sorted([new_value, new_value, overflow_value])]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		self._expands.append((3, Node.expand_split, (conveyor_speed,)))
		return new_nodes

	def min_level(self, seen_ids):
		r = self.level
		
		if self.node_id in seen_ids: return r
		seen_ids.add(self.node_id)

		for child in self.children:
			if child.node_id in seen_ids: continue
			r = min(r, child.min_level(seen_ids))
		
		return r

	def tag_levels_update(self, threshold, amount, seen_ids):
		
		if self.node_id in seen_ids: return
		seen_ids.add(self.node_id)

		if self.level >= threshold:
			self.levels_to_add += amount
		for child in self.children:
			if child.node_id in seen_ids: continue
			child.tag_levels_update(threshold, amount, seen_ids)

	def apply_levels_update(self, seen_ids):
		
		if self.node_id in seen_ids: return
		seen_ids.add(self.node_id)

		self.level += self.levels_to_add
		self.levels_to_add = 0
		for child in self.children:
			if child.node_id in seen_ids: continue
			child.apply_levels_update(seen_ids)

	@staticmethod
	def expand_extract(node, conveyor_speed):
		# print(f"expand_extract {node}")
		from utils import find_n_m_l, compute_branches_count, compute_looping_branches
		n, m, l, n_splitters = find_n_m_l(node.value)
		branches_count = compute_branches_count(n, m)
		looping_branches = compute_looping_branches(n, m, l, branches_count)
		extract_branches = compute_looping_branches(n, m, conveyor_speed, branches_count)
		overflow_branches = compute_looping_branches(n, m, node.value - conveyor_speed, branches_count)
		values = [1]
		for _ in range(m): values.append(values[-1] * 3)
		for _ in range(n): values.append(values[-1] * 2)
		values = [x for x in reversed(values)]
		# print(f"{n = }\n{m = }\n{l = }\n{n_splitters = }\n{looping_branches = }\n{extract_branches = }\n{overflow_branches = }\n{values = }")
		cur_level = node.level + 1
		merged_node = Node(values[0], level=cur_level)
		merged_node.levels_to_add = -(n + m)
		merged_node.parents = [node]
		extract_node = overflow_node = None
		if node.children[0].value == conveyor_speed:
			extract_node = node.children[0]
			overflow_node = node.children[1]
		else:
			extract_node = node.children[1]
			overflow_node = node.children[0]
		extract_node.parents = []
		overflow_node.parents = []
		original_children = node.children
		node.children = [merged_node]
		cur_nodes = [merged_node]
		new_nodes = []
		
		for i in range(1, n+1):
			cur_level += 1
			n_looping_branches, a = looping_branches.get((i, 0), (0, 0))
			n_extract_branches, b = extract_branches.get((i, 0), (0, 0))
			n_overflow_branches, c = overflow_branches.get((i, 0), (0, 0))
			# print(f"{n_looping_branches = }\n{n_extract_branches = }\n{n_overflow_branches = }")
			n_reroute_branches = n_looping_branches + n_extract_branches + n_overflow_branches
			n_ignore_branches = a + b + c # their individual meaning are not important, only their sum is, hence the short namings
			total_to_ignore = n_reroute_branches + n_ignore_branches
			if i < n or m != 0:
				for j in range(2**i - total_to_ignore):
					new_node = Node(values[i], level=cur_level)
					new_node.levels_to_add = -(n + m)
					cur = cur_nodes[(n_reroute_branches + j) // 2]
					cur.children.append(new_node)
					new_node.parents = [cur]
					new_nodes.append(new_node)
			else:
				new_nodes = original_children

			for j in range(n_looping_branches):
				cur = cur_nodes[j // 2]
				cur.children.append(merged_node)
				merged_node.parents.append(cur)

			for j in range(n_extract_branches):
				cur = cur_nodes[(n_looping_branches + j) // 2]
				cur.children.append(extract_node)
				extract_node.parents.append(cur)

			for j in range(n_overflow_branches):
				cur = cur_nodes[(n_looping_branches + n_extract_branches + j) // 2]
				cur.children.append(overflow_node)
				overflow_node.parents.append(cur)

			cur_nodes, new_nodes = new_nodes, []
		
		for i in range(1, m+1):
			cur_level += 1
			n_looping_branches, a = looping_branches.get((n, i), (0, 0))
			n_extract_branches, b = extract_branches.get((n, i), (0, 0))
			n_overflow_branches, c = overflow_branches.get((n, i), (0, 0))
			n_reroute_branches = n_looping_branches + n_extract_branches + n_overflow_branches
			n_ignore_branches = a + b + c # their individual meaning are not important, only their sum is, hence the short namings
			total_to_ignore = n_reroute_branches + n_ignore_branches
			if i < m:
				for j in range(2**n*3**i - total_to_ignore):
					new_node = Node(values[n + i], level=cur_level)
					new_node.levels_to_add = -(n + m)
					cur = cur_nodes[(n_reroute_branches + j) // 3]
					cur.children.append(new_node)
					new_node.parents = [cur]
					new_nodes.append(new_node)
			else:
				new_nodes = original_children

			for j in range(n_looping_branches):
				cur = cur_nodes[j // 3]
				cur.children.append(merged_node)
				merged_node.parents.append(cur)

			for j in range(n_extract_branches):
				cur = cur_nodes[(n_looping_branches + j) // 3]
				cur.children.append(extract_node)
				extract_node.parents.append(cur)

			for j in range(n_overflow_branches):
				cur = cur_nodes[(n_looping_branches + n_extract_branches + j) // 3]
				cur.children.append(overflow_node)
				overflow_node.parents.append(cur)

			cur_nodes, new_nodes = new_nodes, []

		# temporary
		# if len(merged_node.parents) > 3:
		# 	merged_node._expands.append((2, Node.expand_merge, tuple()))

		if len(extract_node.parents) > 3:
			extract_node._expands.append((2, Node.expand_merge, tuple()))

		if len(overflow_node.parents) > 3:
			overflow_node._expands.append((2, Node.expand_merge, tuple()))

		return node.level + 1, n + m

	@staticmethod
	def expand_divide(node, d):
		# print(f"expand_divide {node}")
		from utils import find_n_m_l, compute_branches_count, compute_looping_branches
		n, m, l, n_splitters = find_n_m_l(d)
		branches_count = compute_branches_count(n, m)
		looping_branches = compute_looping_branches(n, m, l, branches_count)
		values = [node.value // d]
		for _ in range(m): values.append(values[-1] * 3)
		for _ in range(n): values.append(values[-1] * 2)
		values = [x for x in reversed(values)]
		# print(f"{n = }\n{m = }\n{l = }\n{n_splitters = }\n{looping_branches = }\n{values = }")
		cur_level = node.level + 1
		merged_node = Node(values[0], level=cur_level)
		merged_node.levels_to_add = -(n + m)
		merged_node.parents = [node]
		original_children = node.children
		node.children = [merged_node]
		cur_nodes = [merged_node]
		new_nodes = []
		
		for i in range(1, n+1):
			cur_level += 1
			n_looping_branches, n_ignore_branches = looping_branches.get((i, 0), (0, 0))
			total_to_ignore = n_looping_branches + n_ignore_branches
			if i < n or m != 0:
				for j in range(2**i - total_to_ignore):
					new_node = Node(values[i], level=cur_level)
					new_node.levels_to_add = -(n + m)
					cur = cur_nodes[(n_looping_branches + j) // 2]
					cur.children.append(new_node)
					new_node.parents = [cur]
					new_nodes.append(new_node)
			else:
				new_nodes = original_children
				for j, new_node in enumerate(new_nodes):
					new_node.levels_to_add = -(n + m)
					cur = cur_nodes[(n_looping_branches + j) // 2]
					cur.children.append(new_node)
					new_node.parents = [cur]

			for j in range(n_looping_branches):
				cur = cur_nodes[j // 2]
				cur.children.append(merged_node)
				merged_node.parents.append(cur)
			
			cur_nodes, new_nodes = new_nodes, []
		
		for i in range(1, m+1):
			cur_level += 1
			n_looping_branches, n_ignore_branches = looping_branches.get((n, i), (0, 0))
			total_to_ignore = n_looping_branches + n_ignore_branches
			if i < m:
				for j in range(2**n*3**i - total_to_ignore):
					new_node = Node(values[n + i], level=cur_level)
					new_node.levels_to_add = -(n + m)
					cur = cur_nodes[(n_looping_branches + j) // 3]
					cur.children.append(new_node)
					new_node.parents = [cur]
					new_nodes.append(new_node)
			else:
				new_nodes = original_children
				for j, new_node in enumerate(new_nodes):
					new_node.levels_to_add = -(n + m)
					cur = cur_nodes[(n_looping_branches + j) // 3]
					cur.children.append(new_node)
					new_node.parents = [cur]

			# print(f"\n{i = }\n{n_looping_branches = }\n{n_ignore_branches = }\n{total_to_ignore = }\n{cur_nodes = } {len(cur_nodes)}\n{new_nodes = } {len(new_nodes)}")

			for j in range(n_looping_branches):
				cur = cur_nodes[j // 3]
				cur.children.append(merged_node)
				merged_node.parents.append(cur)

			cur_nodes, new_nodes = new_nodes, []

		# temporary
		# if len(merged_node.parents) > 3:
		# 	merged_node._expands.append((2, Node.expand_merge, tuple()))

		return node.level, n + m

	@staticmethod
	def expand_merge(node):
		# print(f"expand_merge {node}")
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

	@staticmethod
	def expand_split(node, conveyor_speed):
		new_value = conveyor_speed // 3
		new_node = Node(new_value << 1, level=node.level + 1)
		new_node.levels_to_add = -1
		new_node.children = [new_node]
		new_node.parents = [node]
		original_children = node.children
		node.children = [new_node]
		for child in original_children:
			if child.value == new_value:
				new_node.children.append(child)
				child.parents = [new_node]
			else:
				node.children.append(child)
		return node.level + 1, 1

	def expand(self, seen_ids):

		if self.node_id in seen_ids: return []
		seen_ids.add(self.node_id)

		levels_updates = []
		while self._expands:
			_, _expand, args = self._expands.pop(0)
			levels_updates.append(_expand(self, *args))

		for child in self.children:
			if child.node_id in seen_ids: continue
			levels_updates.extend(child.expand(seen_ids))
		
		return levels_updates

	@staticmethod
	def expand_roots(roots):
		min_level_after_zero = 2**32
		seen_ids = set()
		for root in roots:
			for child in root.children:
				min_level_after_zero = min(min_level_after_zero, child.min_level(seen_ids))

		seen_ids = set()
		levels_updates = [(1, 1 - min_level_after_zero)]
		for root in roots:
			levels_updates.extend(root.expand(seen_ids))
		
		for threshold, amount in levels_updates:
			seen_ids = set()
			for root in roots:
				root.tag_levels_update(threshold, amount, seen_ids)
		
		seen_ids = set()
		for root in roots:
			root.apply_levels_update(seen_ids)

	@staticmethod
	def save(roots, filename, unit_flow_ratio=1):
		try:
			G = nx.MultiDiGraph()

			seen_ids = set()
			for root in roots:
				root.level = 0
				root.populate(G, seen_ids, unit_flow_ratio)

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

			A.layout(prog="dot")
			img_stream = io.BytesIO()
			A.draw(img_stream, format=config.solutions_filename_extension)
			img_stream.seek(0)
			filepath = f"{filename}.{config.solutions_filename_extension}"
			with open(filepath, "wb") as f:
				f.write(img_stream.getvalue())

			Node.wrap(roots[0], f"{filename}.data")
		except Exception as e:
			print(traceback.format_exc(), end="")
			return

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