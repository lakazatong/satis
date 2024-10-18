import json, os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from config import config
from functools import partial

def gcd_incompatible(gcd, value):
	return value < gcd

def get_gcd_incompatible(gcd):
	return partial(gcd_incompatible, gcd)

def remove_pairs(list_a, list_b):
	count_a = Counter(list_a)
	count_b = Counter(list_b)
	for item in count_a.keys():
		if item in count_b:
			pairs_to_remove = min(count_a[item], count_b[item])
			count_a[item] -= pairs_to_remove
			count_b[item] -= pairs_to_remove
	remaining_a = []
	remaining_b = []
	for item, count in count_a.items(): remaining_a.extend([item] * count)
	for item, count in count_b.items(): remaining_b.extend([item] * count)
	return remaining_a, remaining_b

def sort_nodes(nodes):
	return sorted(nodes, key=lambda node: node.value)

def get_node_values(nodes):
	return tuple(map(lambda node: node.value, nodes))

def get_node_ids(nodes):
	return set(map(lambda node: node.node_id, nodes))

def get_short_node_ids(nodes, short=3):
	return set(map(lambda node: node.node_id[-short:], nodes))

def pop_node(node, nodes):
	for i, other in enumerate(nodes):
		if other.node_id == node.node_id:
			return nodes.pop(i)
	return None

def compute_cant_use(target_counts, sources):
	source_counts = {}
	for src in sources:
		if src.value in source_counts:
			source_counts[src.value] += 1
		else:
			source_counts[src.value] = 1
	cant_use = set()
	for src in sources:
		value = src.value
		src_count = source_counts.get(value, None)
		target_count = target_counts.get(value, None)
		if src_count and target_count and max(0, src_count - target_count) == 0:
			cant_use.add(value)
	return cant_use

def get_compute_cant_use(target_counts):
	return partial(compute_cant_use, target_counts)

def insert_into_sorted(sorted_list, item, key=lambda x: x):
	low, high = 0, len(sorted_list)
	while low < high:
		mid = low + (high - low) // 2
		if key(item) > key(sorted_list[mid]):
			low = mid + 1
		else:
			high = mid
	sorted_list.insert(low, item)

def get_sim_without(value, sources):
	sim = [src.value for src in sources]
	sim.remove(value)
	return sim

def clear_solution_files():
	for filename in os.listdir('.'):
		if config.solution_regex.match(filename):
			os.remove(filename)

def parse_user_input(user_input):
	separator = 'to'
	if len(user_input.split(" ")) < 3 or separator not in user_input:
		print(f"Usage: <source_args> {separator} <target_args>")
		return [], []

	source_part, target_part = user_input.split(separator)
	source_args = source_part.strip().split()
	target_args = target_part.strip().split()

	if not source_args:
		print("Error: At least one source value must be provided.")
		return None, None

	if not target_args:
		print("Error: At least one target value must be provided.")
		return None, None

	source_values = []
	i = 0
	while i < len(source_args):
		src = source_args[i]
		if not src.endswith('x'):
			source_value = int(src)
			source_values.append(source_value)
			i += 1
			continue
		if len(src) < 2 or not src[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return None, None
		multiplier = int(src[:-1])
		source_value = int(source_args[source_args.index(src) + 1])
		for _ in range(multiplier):
			source_values.append(source_value)
		i += 2

	target_values = []
	i = 0
	while i < len(target_args):
		target = target_args[i]
		if not target.endswith('x'):
			target_value = int(target)
			target_values.append(target_value)
			i += 1
			continue
		if len(target) < 2 or not target[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return None, None
		multiplier = int(target[:-1])
		if i + 1 == len(target_args):
			print("Error: You must provide a target value after Nx.")
			return None, None
		target_value = int(target_args[i + 1])
		for _ in range(multiplier):
			target_values.append(target_value)
		i += 2

	return source_values, target_values

# graveyard

# class Binary:
# 	def __init__(self, n):
# 		self.n = n
# 		self._arr = [0] * n
# 		self.bit_count = 0

# 	def increment(self):
# 		# returns if it's 0 after the increment
# 		for i in range(self.n):
# 			self._arr[i] = not self._arr[i]
# 			if self._arr[i]:
# 				self.bit_count += 1
# 				return True
# 			self.bit_count -= 1
# 		return False

# 	def __iadd__(self, other):
# 		for _ in range(other - 1): self.increment()
# 		return self.increment()

# 	def __getitem__(self, index):
# 		return self._arr[index]

# 	def __setitem__(self, index, value):
# 		old_bit = self._arr[index]
# 		self._arr[index] = value
# 		self.bit_count += (value - old_bit) 

# 	def __iter__(self):
# 		return iter(self._arr)

# 	def __str__(self):
# 		return str(self._arr)

# def show(G):
# 	edge_labels = nx.get_edge_attributes(G, 'label')
# 	pos = nx.spring_layout(G)
# 	plt.figure(figsize=(12, 8))
# 	nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=10, font_weight='bold')
# 	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
# 	plt.show()

# def load_graph_from_json(file_path):
# 	with open(file_path, 'r') as f:
# 		data = json.load(f)
# 	G = nx.node_link_graph(data)
# 	return G

# def find_shortest_path_with_operations(G, src, dst):
# 	if src not in G or dst not in G or dst >= src: 
# 		return None, None
	
# 	filtered_graph = nx.DiGraph()
# 	filtered_graph.add_nodes_from(G.nodes(data=True))
# 	empty = True
	
# 	for u, v in G.edges():
# 		for op in [2, 3]:
# 			label = str(op)
# 			if G.edges[u, v]['label'] == label and v == int(u / op):
# 				filtered_graph.add_edge(u, v, label=label)
# 				empty = False
# 			if G.edges[v, u]['label'] == label and u == int(v / op):
# 				filtered_graph.add_edge(v, u, label=label)
# 				empty = False

# 	if empty: 
# 		return None, None
	
# 	filtered_graph.remove_nodes_from([node for node in filtered_graph.nodes() if filtered_graph.degree(node) == 0])

# 	try:
# 		# Find the shortest path in the filtered graph
# 		shortest_path = nx.shortest_path(filtered_graph, source=src, target=dst)
# 		operations = []
		
# 		for i in range(len(shortest_path) - 1):
# 			u, v = shortest_path[i], shortest_path[i + 1]
# 			# Get the operation label from the filtered graph
# 			label = filtered_graph.edges[u, v]['label']
# 			operations.append(f"/{label}")  # Format as /2 or /3

# 		return shortest_path, operations
# 	except nx.NetworkXNoPath:
# 		return None, None

# def get_all_pairs_operations(G):
# 	results = []
	
# 	for src in G.nodes():
# 		for dst in G.nodes():
# 			if src > dst:
# 				path, operations = find_shortest_path_with_operations(G, src, dst)
# 				if path is not None:
# 					results.append([src, dst, operations])
	
# 	return results

# if __name__ == "__main__":
# 	graph_file = "graph_data.json"
# 	G = load_graph_from_json(graph_file)
	
# 	all_operations = get_all_pairs_operations(G)
	
# 	for res in all_operations:
# 		print(res)