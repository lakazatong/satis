import traceback, io, tempfile
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph
from config import config

from fastList import FastList

from utils import get_node_values
from node import Node

# responsible for updating depth, tree_height, size and level of all nodes
# also provides an easy way of accessing past sources
class Tree:
	def __init__(self, sources):
		self.roots = sources
		self.sources = sources
		self.levels = [sources]
		self.past = FastList()
		self.current_level = 1
		self.source_values = tuple(src.value for src in sources)
		self.nodes = sources
		self.parents = None
		self.init()
		# add it after the init of the roots so that they have no parents when initializing
		if len(sources) == 1:
			self.dummy_root = sources[0]
		else:
			self.dummy_root = Node(0)
			self.dummy_root.children = sources
			for src in sources:
				src.parents.append(self.dummy_root)
			# the dummy root is just there to accumulate the tree size
			self.dummy_root.size = len(sources)
			# self.dummy_root.tree_height = sources[0].tree_height # must init for future updates

	def __repr__(self):
		return "\n".join(str(root) for root in self.roots)

	def size(self):
		return self.dummy_root.size

	# def tree_height(self):
	# 	return self.dummy_root.tree_height

	def init(self):
		new_depth = 1 + (min(p.depth for p in self.parents) if self.parents else 0)
		for node in self.nodes:
			node.depth = new_depth
			# node.tree_height = 1
			node.size = 1
			node.level = self.current_level

	def update(self):
		n = len(self.nodes)
		queue = [p for p in self.parents]
		seen = set()
		while queue:
			p = queue.pop()
			if p.node_id in seen: continue
			seen.add(p.node_id)
			# p.tree_height += 1
			p.size += n
			queue.extend(p.parents)

	def add(self, nodes):
		# just to avoid doing it three times
		self.nodes = nodes
		self.parents = nodes[0].parents
		self.current_level += 1
		self.init()
		self.update()
		parent_ids = set(p.node_id for p in self.parents)
		self.sources = [src for src in self.levels[-1] if src.node_id not in parent_ids] + nodes
		self.levels.append(self.sources)
		self.past.append(self.source_values)
		self.source_values = get_node_values(self.sources)

	def visualize(self, filename):
		try:
			G = nx.DiGraph()
			for root in self.roots:
				root.populate(G)

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