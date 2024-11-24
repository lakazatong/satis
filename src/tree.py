from utils.fastlist import FastList
from node import Node

# responsible for updating level of all nodes while providing a quick access to past sources
class Tree:
	def __init__(self, roots):
		self.roots = roots
		self.sources = roots
		self.levels = [roots]
		self.past = FastList()
		self.current_level = 0
		self.source_values = tuple(root.value for root in roots)
		self.n_sources = len(roots)
		self.size = 0
		# self.total_seen = {}
		for root in roots:
			root.level = self.current_level
			# self.total_seen[root.value] = self.total_seen.get(root.value, 0) + 1

	def __str__(self):
		attrs = ",\n\t".join(f"{key}={str(value)}" for key, value in self.__dict__.items())
		return f"{self.__class__.__name__}(\n\t{attrs}\n)"

	def pretty(self):
		dummy_root = Node(0)
		dummy_root.children = self.roots
		return dummy_root.pretty()

	def to_nx_graph(self):
		import networkx as nx
		g = nx.Graph()
		
		def add_edges(node):
			for child in node.children:
				g.add_edge(node, child)
				add_edges(child)
		
		for root in self.roots:
			add_edges(root)
		
		return g

	def __eq__(self, t2):
		from networkx import is_isomorphic
		return is_isomorphic(self.to_nx_graph(), t2.to_nx_graph())

	def deepcopy(self):
		copied_nodes = {}
		copied_roots = [root._deepcopy(copied_nodes) for root in self.roots]
		new_tree = Tree(copied_roots)
		new_tree.sources = [copied_nodes[src] for src in self.sources]
		new_tree.levels = [[copied_nodes[src] for src in level] for level in self.levels]
		# new_tree.levels += [[copied_nodes[src] for src in level] for level in self.levels[1:]] # may be faster
		new_tree.past = FastList()
		new_tree.past.extend(self.past)
		new_tree.current_level = self.current_level
		new_tree.source_values = self.source_values # tuples are deepcopied in python
		new_tree.size = self.size
		return new_tree

	def add(self, nodes, cost):
		from bisect import insort
		self.current_level += 1
		# init new nodes
		# for node in nodes: node.size = 1
		
		parents = nodes[0].parents
		self.sources = [src for src in self.levels[-1] if src not in parents]
		for node in nodes: insort(self.sources, node, key=lambda node: node.value)
		self.n_sources = len(self.sources)
		
		# update levels and size
		for src in self.sources: src.level = self.current_level
		self.levels.append(self.sources)
		self.size += cost

		# update past
		self.past.append(self.source_values)
		self.source_values = Node.get_node_values(self.sources)
		# for value in self.source_values:
		# 	self.total_seen[value] = self.total_seen.get(value, 0) + 1

	def save(self, filename, unit_flow_ratio=1):
		Node.save(self.roots, filename, unit_flow_ratio=unit_flow_ratio)
		Node.expand_roots(self.roots)
		Node.save(self.roots, filename + "_expanded", unit_flow_ratio=unit_flow_ratio)

	def _has_sufficient_sources(self, coeffs, srcs_list, srcs_counts):
		for i, coeff in enumerate(coeffs):
			source = srcs_list[i]
			if srcs_counts[source] < coeff:
				return False
		return True

	def _apply_best_merge(self, sources, targets):
		min_cost = float('inf')
		best_merge_sources = None
		best_merge_target = None
		best_n_sources = None
		for t in targets:
			srcs = [src for src in sources if src < t]
			if not srcs: continue
			srcs_set = list(set(srcs))
			n = len(srcs_set)
			tmp = find_linear_combinations(srcs_set, t)
			if not tmp: continue
			all_coeffs = sorted(map(lambda coeffs: (n-coeffs.count(0), coeffs), tmp), key=lambda x: -x[0])
			srcs_counts = {src: srcs.count(src) for src in srcs}
			for n_sources, coeffs in all_coeffs:
				if not has_sufficient_sources(coeffs, srcs_set, srcs_counts):
					continue
				# we pick the first combination of sources that can make that target
				# without considering which merge keeps as many "interesting" sources for later
				# the merge of a cost is only dependent on the number of sources
				if best_n_sources and n_sources == best_n_sources: break
				cost = merge_cost(sum(coeffs), 1)
				if cost >= min_cost:
					continue
				min_cost = cost
				best_merge_sources = []
				best_merge_target = t
				best_n_sources = n_sources
				for i, coeff in enumerate(coeffs):
					best_merge_sources.extend([srcs_set[i]] * coeff)
		if best_merge_sources:
			for val in best_merge_sources:
				sources.remove(val)
			targets.remove(best_merge_target)
			logs.append(('merge', best_merge_sources, tuple()))
			return True
		return False

	def _apply_best_extract_two_targets(self, sources, targets):
		min_cost = float('inf')
		best_extraction = None
		best_source = None
		min_target = min(targets)
		for source in (src for src in sources if src > min_target):
			for i, t1 in enumerate(targets):
				for t2 in targets[i + 1:]:
					if source != t1 + t2 or t1 == t2:
						continue
					cost = extract_cost(source, t1)
					if cost < min_cost:
						min_cost = cost
						best_extraction = (t1, t2)
						best_source = source

		if best_extraction:
			t1, t2 = best_extraction
			sources.remove(best_source)
			targets.remove(t1)
			targets.remove(t2)
			logs.append(('extract', (best_source,), (t1, t2)))
			return True
		return False

	def _apply_best_extract_one_target(self, sources, targets):
		min_cost = float('inf')
		best_t = None
		best_source = None

		for source in sources:
			for t in targets:
				if source <= t:
					continue
				cost = extract_cost(source, t)
				if cost < min_cost:
					min_cost = cost
					best_t = t
					best_source = source

		if not best_t: return False
		overflow = best_source - best_t
		if best_t == overflow: return False
		sources.remove(best_source)
		sources.append(overflow)
		targets.remove(best_t)
		logs.append(('extract', (best_source,), (best_t, overflow)))
		return True

	def quick_solve(self, targets):
		n_targets = len(targets)

		while True:
		
			sources, targets = remove_pairs(self.source_values, targets)

			if (len(sources) == 0 and len(targets) != 0) or (len(sources) != 0 and len(targets) == 0):
				print('quick_solve: impossible case reached')
				exit(1)

			if len(targets) == 1:
				self.add([Node.merge(sources)], merge_cost(len(sources)))
				break

			if not self._apply_best_merge(sources, targets):
				if not self._apply_best_extract_two_targets(sources, targets):
					self._apply_best_extract_one_target(sources, targets)

			if sources == targets:
				break

	# all_nodes
	# [2(0a1), 2(73a), 2(66f), 2(938), 2(7fc), 5(d91), 5(251), 6(8be), 6(3e9)]
	# [10(5a5, [(1, <function Node.expand_divide at 0x000001FC6BF48F40>, (5,))]), 10(752), 10(2d4)]
	
	# all_ns
	# [5, 2, 1, 2]
	# [3, 1]
	
	# all_costs
	# [4, 1, 0, 1]
	# [1, 0]
	
	# self.source_values
	# [30, 12]
	def attach_leaves(self, leaves):
		# attach
		assert len(leaves) == self.n_sources
		levels = []
		for i in range(self.n_sources):
			src = self.sources[i]
			for child in leaves[i].children:
				src.children.append(child)
				child.parents = [src]
			if src.children:
				levels.append(src.children)
		while levels:
			level = levels.pop(0)
			# print(self.sources)
			# print(level)
			# print()
			self.add(level, 0)
			for node in level:
				if node.children:
					levels.append(node.children)

	# graveyard

	# def simplify(self):
	# 	# doesn't restore this tree's past to reflect the changes
	# 	seen_ids = set()
	# 	queue = [root for root in self.roots]
	# 	while queue:
	# 		node = queue.pop()
	# 		deepest_node = node.simplify_info()
	# 		if not deepest_node: continue
	# 		for child in node.children:
	# 			for grandchild in child.children:
	# 				grandchild.parents.remove(child)
	# 				grandchild.value -= child.value
	# 		deepest_node.parents.append(node)
	# 		self.size -= len(node.children) # outdated
	# 		childrens = set(child for child in node.children)
	# 		for level in self.levels:
	# 			for i in range(len(level)-1, -1, -1):
	# 				if level[i] in childrens:
	# 					level.pop(i)
	# 		node.children = [deepest_node]