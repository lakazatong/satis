from fastList import FastList

from utils import get_node_values

# responsible for updating depth, tree_height, size and level of all nodes
# also provides an easy way of accessing past sources
class Tree:
	def __init__(self, root):
		self.root = root
		self.sources = [root]
		self.levels = [self.sources]
		self.past = FastList()
		self.current_level = 1
		self.source_values = (root.value,)
		self.nodes = self.sources
		self.parents = None
		self.init()

	def init(self):
		new_depth = 1 + (min(p.depth for p in self.parents) if self.parents else 0)
		for node in self.nodes:
			node.depth = new_depth
			node.tree_height = 1
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
			p.tree_height += 1
			p.size += n

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