# graveyard

# def divides(a, b):
# 	if a == 0: raise ValueError("a == 0")
# 	q, remainder = divmod(b, a)
# 	return q if remainder == 0 and q != 1 else None

# def all_sums(numbers):
# 	sums = {Fraction(0): 0}
# 	for num in numbers:
# 		new_sums = {s + num: count + 1 for s, count in sums.items()}
# 		sums.update(new_sums)
# 	sums.pop(Fraction(0))
# 	return sums

# def compute_minimum_possible_fraction(values):
# 	min_fraction = None

# 	for value in values:
# 		if value.denominator == 1:
# 			fraction = Fraction(1, 1)  # Treat integers as Fraction(1, 1)
# 		else:
# 			fraction = value - Fraction(value.numerator // value.denominator)  # Get the fractional part

# 		if min_fraction is None or fraction < min_fraction:
# 			min_fraction = fraction

# 	return min_fraction

# def compute_gcd(*fractions):
# 	numerators = [f.numerator for f in fractions]
# 	denominators = [f.denominator for f in fractions]
# 	gcd_numerator = functools.reduce(math.gcd, numerators)
# 	lcm_denominator = functools.reduce(lambda x, y: x * y // math.gcd(x, y), denominators)
# 	return Fraction(gcd_numerator, lcm_denominator)

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