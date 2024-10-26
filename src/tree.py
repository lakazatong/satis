import traceback, io, networkx as nx

from networkx.drawing.nx_agraph import to_agraph
from bisect import insort
from config import config
from fastList import FastList
from utils import get_node_values, get_node_ids

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
		self.size = len(roots)
		for root in roots: root.level = self.current_level

		# graveyard

		# self.init()
		# add it after the init of the roots so that they have no parents when initializing
		# if len(sources) == 1:
		# 	self.dummy_root = sources[0]
		# else:
		# 	self.dummy_root = Node(0)
		# 	self.dummy_root.children = sources
		# 	for src in sources:
		# 		src.parents.append(self.dummy_root)
		# 	# the dummy root is just there to accumulate the tree size
		# 	self.dummy_root.size = len(sources)
		# 	# self.dummy_root.tree_height = sources[0].tree_height # must init for future updates

	def __repr__(self):
		return "\n".join(str(root) for root in self.roots)

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

	def add(self, nodes):
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
		self.size += len(nodes)

		# update past
		self.past.append(self.source_values)
		self.source_values = get_node_values(self.sources)

	def level_optimal(self, i):
		return len(self.levels[i - 1]) - len(self.levels[i]) == 2

	def deepest_optimal_level(self):
		i = 1
		while i <= self.current_level:
			if not self.level_optimal(i): break
			i += 1
		return i - 1

	def visualize(self, filename):
		try:
			G = nx.DiGraph()
			seen_ids = set()
			for root in self.roots:
				root.level = 0
				root.populate(G, seen_ids)

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