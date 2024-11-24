from treelike import TreeLike
from config import config
from cost import find_n_m_l, compute_branches_count, compute_looping_branches
from fractions import Fraction

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
		if not isinstance(value, int) and not isinstance(value, Fraction): raise ValueError(f"not int or Fraction ({type(value)} {value})")
		import uuid
		from utils.fastlist import FastList
		self.value = value
		self.node_id = node_id if node_id is not None else uuid.uuid4()
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
			self.repr_whitelist.add('level')
		else:
			self.repr_keys = True
			self.repr_whitelist.update(('value', 'node_id', 'parents'))
			if config.include_level_in_logs:
				self.repr_whitelist.add('level')

	def __hash__(self):
		return hash(self.node_id)

	def repr_self(self):
		return str(self.value) if config.short_repr else 'Node'

	def repr_node_id(self):
		return 'node_id', str(self.node_id)[-3:]

	def repr_parents(self):
		return 'parents', Node.get_short_node_ids(self.parents)

	def repr__expands(self):
		def expand_code_to_str(code):
			if code == 0: return 'extract'
			if code == 1: return 'divide'
			if code == 2: return 'merge'
			if code == 3: return 'split'
			print('expand_code_to_str: impossible case reached')
			exit(1)
		return 'expands', [expand_code_to_str(code) for code, *_ in self._expands]

	@staticmethod
	def wrap(root_node, path):
		seen = set()

		def traverse(node, output):
			
			if node in seen: return
			seen.add(node)
			
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
				if child in seen: continue
				traverse(child, output)
			for parent in node.parents:
				if parent in seen: continue
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

	def _deepcopy(self, copied_nodes):
		stack = [(self, None)]
		while stack:
			node, parent = stack.pop()
			
			if node in copied_nodes:
				new_node = copied_nodes[node]
				if parent: new_node.parents.append(parent)
			else:
				new_node = Node(node.value, node_id=node.node_id)
				new_node.level = node.level
				new_node.past.extend(self.past)
				new_node._expands = [_expand for _expand in node._expands]
				copied_nodes[node] = new_node
				new_node.parents = [parent] if parent else []
				stack.extend((child, new_node) for child in node.children)
			
			if parent: parent.children.append(new_node)
		return copied_nodes[self]

	def deepcopy(self):
		return self._deepcopy({})

	def populate(self, G, seen, unit_flow_ratio):
		if self in seen: return
		seen.add(self)
		G.add_node(self, label=str(Fraction(self.value, unit_flow_ratio)), level=self.level)
		# G.add_node(self.node_id, label=str(Fraction(self.value, unit_flow_ratio)) + ", " + self.repr_node_id()[1], level=self.level)
		for child in self._children:
			G.add_edge(self, child)
			if child in seen: continue
			child.populate(G, seen, unit_flow_ratio)

	def extract(self, value):
		if value not in config.conveyor_speeds:
			self._expands.append((0, Node.expand_extract, (value,)))
		overflow_value = self.value - value
		new_nodes = [Node(v, parent_past=self.past, level=self.level + 1) for v in sorted([value, overflow_value])]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes

	def divide(self, divisor):
		if divisor != 2 and divisor != 3:
			self._expands.append((1, Node.expand_divide, (divisor,)))
		divided_value = self.value // divisor
		new_nodes = [Node(divided_value, parent_past=self.past, level=self.level + 1) for _ in range(divisor)]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		return new_nodes

	@staticmethod
	def merge(nodes):
		# nodes won't have any children when solving within the abstract world
		# but it will be important when this function will be called when expanding merges
		summed_value = sum(Fraction(node.value, len(node.children)) if node.children else node.value for node in nodes)
		new_node = Node(summed_value, level=max(node.level for node in nodes) + 1)
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
		new_nodes = [Node(v, parent_past=self.past, level=self.level + 1) for v in sorted([new_value, new_value, overflow_value])]
		for node in new_nodes:
			self._children.append(node)
			node.parents.append(self)
		self._expands.append((3, Node.expand_split, (conveyor_speed,)))
		return new_nodes

	def min_level(self, seen):
		r = self.level
		
		if self in seen: return r
		seen.add(self)

		for child in self.children:
			if child in seen: continue
			r = min(r, child.min_level(seen))
		
		return r

	def tag_levels_update(self, threshold, amount, seen):
		
		if self in seen: return
		seen.add(self)

		if self.level >= threshold:
			self.levels_to_add += amount
		for child in self.children:
			if child in seen: continue
			child.tag_levels_update(threshold, amount, seen)

	def apply_levels_update(self, seen):
		
		if self in seen: return
		seen.add(self)

		self.level += self.levels_to_add
		self.levels_to_add = 0
		for child in self.children:
			if child in seen: continue
			child.apply_levels_update(seen)

	@staticmethod
	def expand_extract(node, conveyor_speed):
		# print(f"expand_extract {node}")
		import math
		divided_value = math.gcd(node.value, node.value - conveyor_speed)
		d = node.value // divided_value
		n, m, l, n_splitters = find_n_m_l(d)
		to_loop_value = l * divided_value
		new_node_value = node.value + to_loop_value
		loop_node = Node(to_loop_value) if new_node_value > config.conveyor_speed_limit else node
		is_loop_node_root = len(loop_node.parents) == 0
		n_extract = conveyor_speed // Fraction(node.value, d)
		branches_count = compute_branches_count(n, m)
		looping_branches = compute_looping_branches(n, m, l, branches_count)
		extract_branches = compute_looping_branches(n, m, n_extract, branches_count)
		overflow_branches = compute_looping_branches(n, m, d - n_extract, branches_count)
		# print(looping_branches)
		# print(extract_branches)
		# print(overflow_branches)
		values = [new_node_value]
		for _ in range(n): values.append(Fraction(values[-1], 2))
		for _ in range(m): values.append(Fraction(values[-1], 3))
		# print(f"{n = }\n{m = }\n{l = }\n{n_splitters = }\n{looping_branches = }\n{extract_branches = }\n{overflow_branches = }\n{values = }")
		cur_level = node.level + 1
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
		node.children = []
		cur_nodes = [node]
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
				cur.children.append(loop_node)
				loop_node.parents.append(cur)

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
				cur.children.append(loop_node)
				loop_node.parents.append(cur)

			for j in range(n_extract_branches):
				cur = cur_nodes[(n_looping_branches + j) // 3]
				cur.children.append(extract_node)
				extract_node.parents.append(cur)

			for j in range(n_overflow_branches):
				cur = cur_nodes[(n_looping_branches + n_extract_branches + j) // 3]
				cur.children.append(overflow_node)
				overflow_node.parents.append(cur)

			cur_nodes, new_nodes = new_nodes, []

		if new_node_value > config.conveyor_speed_limit:
			for merge_node in node.children:
				loop_node.children.append(merge_node)
				merge_node.parents.append(loop_node)

		threshold = 2 if is_loop_node_root else 3
		if len(loop_node.parents) > threshold:
			loop_node._expands.append((2, Node.expand_merge, (threshold,)))

		if len(extract_node.parents) > 3:
			extract_node._expands.append((2, Node.expand_merge, tuple()))

		if len(overflow_node.parents) > 3:
			overflow_node._expands.append((2, Node.expand_merge, tuple()))

		return node.level + 1, n + m

	@staticmethod
	def expand_divide(node, d):
		# print(f"expand_divide {node}")
		n, m, l, n_splitters = find_n_m_l(d)
		to_loop_value = l * Fraction(l * node.value, 2**n*3**m)
		new_node_value = node.value + to_loop_value
		loop_node = Node(to_loop_value) if new_node_value > config.conveyor_speed_limit else node
		is_loop_node_root = len(loop_node.parents) == 0
		branches_count = compute_branches_count(n, m)
		looping_branches = compute_looping_branches(n, m, l, branches_count)
		values = [new_node_value]
		for _ in range(n): values.append(Fraction(values[-1], 2))
		for _ in range(m): values.append(Fraction(values[-1], 3))
		# print(f"{n = }\n{m = }\n{l = }\n{n_splitters = }\n{looping_branches = }\n{values = }")
		cur_level = node.level + 1
		original_children = node.children
		node.children = []
		cur_nodes = [node]
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
					cur = cur_nodes[(n_looping_branches + j) // 2]
					cur.children.append(new_node)
					new_node.parents = [cur]

			for j in range(n_looping_branches):
				cur = cur_nodes[j // 2]
				cur.children.append(loop_node)
				loop_node.parents.append(cur)
			
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
					cur = cur_nodes[(n_looping_branches + j) // 3]
					cur.children.append(new_node)
					new_node.parents = [cur]

			# print(f"\n{i = }\n{n_looping_branches = }\n{n_ignore_branches = }\n{total_to_ignore = }\n{cur_nodes = } {len(cur_nodes)}\n{new_nodes = } {len(new_nodes)}")

			for j in range(n_looping_branches):
				cur = cur_nodes[j // 3]
				cur.children.append(node)
				node.parents.append(cur)

			cur_nodes, new_nodes = new_nodes, []

		if new_node_value > config.conveyor_speed_limit:
			for merge_node in node.children:
				loop_node.children.append(merge_node)
				merge_node.parents.append(loop_node)

		threshold = 2 if is_loop_node_root else 3
		if len(loop_node.parents) > threshold:
			loop_node._expands.append((2, Node.expand_merge, (threshold,)))

		return node.level + 1, n + m

	@staticmethod
	def expand_merge(node, threshold=3):
		# print(f"expand_merge {node}")
		nodes_to_merge = node.parents
		while len(nodes_to_merge) > threshold:
			to_merge = [nodes_to_merge.pop(), nodes_to_merge.pop(), nodes_to_merge.pop()]
			merged_node = Node.merge(to_merge)
			for n in to_merge:
				i = 0
				while i < len(n.children):
					if n.children[i] is node: n.children.pop(i)
					else: i += 1
			merged_node.level = node.level
			nodes_to_merge.append(merged_node)
		for n in nodes_to_merge:
			n.children.append(node)
		node.parents = nodes_to_merge
		# all nodes with level >= node.level must have their levels increased by cur_level - node.level
		return 0, 0

	@staticmethod
	def expand_split(node, conveyor_speed):
		# TODO: reuse the expand_divide and add a node on top (will become a splitter in overflow mode or whatever)
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

	def expand(self, seen):

		if self in seen: return []
		seen.add(self)

		levels_updates = []
		while self._expands:
			_, _expand, args = self._expands.pop(0)
			levels_updates.append(_expand(self, *args))

		for child in self.children:
			if child in seen: continue
			levels_updates.extend(child.expand(seen))
		
		return levels_updates

	@staticmethod
	def expand_roots(roots):
		min_level_after_zero = 2**32
		seen = set()
		for root in roots:
			for child in root.children:
				min_level_after_zero = min(min_level_after_zero, child.min_level(seen))

		seen = set()
		levels_updates = [(1, 1 - min_level_after_zero)] # why not
		for root in roots:
			levels_updates.extend(root.expand(seen))
		
		for threshold, amount in levels_updates:
			seen = set()
			for root in roots:
				root.tag_levels_update(threshold, amount, seen)
		
		seen = set()
		for root in roots:
			root.apply_levels_update(seen)

	@staticmethod
	def save(roots, filename, unit_flow_ratio=1):
		import io, pygraphviz as pgv, traceback, os
		try:
			G = pgv.AGraph(strict=False, directed=True)

			# Populate graph
			seen = set()
			for root in roots:
				root.level = 0
				root.populate(G, seen, unit_flow_ratio)

			# Group nodes by level for same rank
			levels = {}
			for node in G.nodes():
				level = int(G.get_node(node).attr.get("level", 0))
				levels.setdefault(level, []).append(node)
			
			for level, nodes in levels.items():
				G.add_subgraph(nodes, rank="same")

			# Invert colors
			G.graph_attr["bgcolor"] = "black"
			for node in G.nodes():
				node.attr["color"] = "white"
				node.attr["fontcolor"] = "white"
				node.attr["style"] = "filled"
				node.attr["fillcolor"] = "black"

			for edge in G.edges():
				edge.attr["color"] = "white"

			# Layout and export
			G.layout(prog="dot")
			img_stream = io.BytesIO()
			G.draw(img_stream, format=config.solutions_filename_extension)
			img_stream.seek(0)
			filepath = f"{filename}.{config.solutions_filename_extension}"
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			with open(filepath, "wb") as f:
				f.write(img_stream.getvalue())

			Node.wrap(roots[0], f"{filename}.data")
		except Exception:
			print(traceback.format_exc(), end="")

	@staticmethod
	def get_node_values(nodes):
		return tuple(map(lambda node: node.value, nodes))

	@staticmethod
	def get_node_ids(nodes):
		return set(map(lambda node: node.node_id, nodes))

	@staticmethod
	def get_short_node_ids(nodes, short=3):
		return set(map(lambda node: str(node.node_id)[-short:], nodes))

	@staticmethod
	def group_values(values):
		# 2 2 2 2 2 5 5 6 6 10 50
		from cost import divide_cost
		from bisect import insort
		levels = []
		total_cost = 0
		levels.append([Node(v) for v in sorted(values)])

		while True:
			current_level = levels[-1]
			next_level = []

			groups = {}
			for node in current_level:
				groups.setdefault(node.value, []).append(node)

			for value, nodes in groups.items():
				count = len(nodes)
				if count > 1:
					parent_node_value = value * count
					parent_node = Node(parent_node_value)
					for node in nodes:
						parent_node.children.append(node)
						node.parents = [parent_node]
					if count > 0:
						total_cost += divide_cost(parent_node_value, count)
						if count > 3:
							parent_node._expands.append((1, Node.expand_divide, (count,)))
					insort(next_level, parent_node, key=lambda node: node.value)
				else:
					insort(next_level, nodes[0], key=lambda node: node.value)

			if len(next_level) == len(current_level):
				break

			levels.append(next_level)

		return levels[-1], total_cost

	# graveyard

	# @staticmethod
	# def pop_node(node, nodes):
	# 	for i, other in enumerate(nodes):
	# 		if other.node_id == node.node_id:
	# 			return nodes.pop(i)
	# 	return None

	# @staticmethod
	# def sort_nodes(nodes):
	# 	return sorted(nodes, key=lambda node: node.value)

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
	# 	new_value = self.value + sum(Node.get_node_values(other))
	# 	new_node = Node(new_value)
	# 	self.parents.append(new_node)
	# 	new_node.children.append(self)
	# 	for node in other:
	# 		node.parents.append(new_node)
	# 		new_node.children.append(node)
	# 	return new_node