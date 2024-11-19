import networkx as nx

from bisect import insort
from config import config
from fastList import FastList
from utils import get_node_values, get_node_ids
from networkx import is_isomorphic
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
		g = nx.Graph()
		
		def add_edges(node):
			for child in node.children:
				g.add_edge(node.node_id, child.node_id)
				add_edges(child)
		
		for root in self.roots:
			add_edges(root)
		
		return g

	def __eq__(self, t2):
		return is_isomorphic(self.to_nx_graph(), t2.to_nx_graph())

	def deepcopy(self):
		copied_nodes = {}
		copied_roots = [root._deepcopy(copied_nodes) for root in self.roots]
		new_tree = Tree(copied_roots)
		new_tree.sources = [copied_nodes[src.node_id] for src in self.sources]
		new_tree.levels = [[copied_nodes[src.node_id] for src in level] for level in self.levels]
		# new_tree.levels += [[copied_nodes[src.node_id] for src in level] for level in self.levels[1:]] # may be faster
		new_tree.past = FastList()
		new_tree.past.extend(self.past)
		new_tree.current_level = self.current_level
		new_tree.source_values = self.source_values # tuples are deepcopied in python
		new_tree.size = self.size
		return new_tree

	def add(self, nodes, cost):
		self.current_level += 1
		# init new nodes
		# for node in nodes: node.size = 1
		
		parent_ids = get_node_ids(nodes[0].parents)
		self.sources = [src for src in self.levels[-1] if src.node_id not in parent_ids]
		for node in nodes: insort(self.sources, node, key=lambda node: node.value)
		self.n_sources = len(self.sources)
		
		# update levels and size
		for src in self.sources: src.level = self.current_level
		self.levels.append(self.sources)
		self.size += cost

		# update past
		self.past.append(self.source_values)
		self.source_values = get_node_values(self.sources)
		# for value in self.source_values:
		# 	self.total_seen[value] = self.total_seen.get(value, 0) + 1

	def save(self, filename, unit_flow_ratio):
		Node.save(self.roots, filename, unit_flow_ratio)

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
	# 		children_ids = set(child.node_id for child in node.children)
	# 		for level in self.levels:
	# 			for i in range(len(level)-1, -1, -1):
	# 				if level[i].node_id in children_ids:
	# 					level.pop(i)
	# 		node.children = [deepest_node]