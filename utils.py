import networkx as nx
import json
import matplotlib.pyplot as plt

def show(G):
	edge_labels = nx.get_edge_attributes(G, 'label')
	pos = nx.spring_layout(G)
	plt.figure(figsize=(12, 8))
	nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=10, font_weight='bold')
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
	plt.show()

def load_graph_from_json(file_path):
	with open(file_path, 'r') as f:
		data = json.load(f)
	G = nx.node_link_graph(data)
	return G

def find_shortest_path_with_operations(G, src, dst):
	if src not in G or dst not in G or dst >= src: 
		return None, None
	
	filtered_graph = nx.DiGraph()
	filtered_graph.add_nodes_from(G.nodes(data=True))
	empty = True
	
	for u, v in G.edges():
		for op in [2, 3]:
			label = str(op)
			if G.edges[u, v]['label'] == label and v == int(u / op):
				filtered_graph.add_edge(u, v, label=label)
				empty = False
			if G.edges[v, u]['label'] == label and u == int(v / op):
				filtered_graph.add_edge(v, u, label=label)
				empty = False

	if empty: 
		return None, None
	
	filtered_graph.remove_nodes_from([node for node in filtered_graph.nodes() if filtered_graph.degree(node) == 0])

	try:
		# Find the shortest path in the filtered graph
		shortest_path = nx.shortest_path(filtered_graph, source=src, target=dst)
		operations = []
		
		for i in range(len(shortest_path) - 1):
			u, v = shortest_path[i], shortest_path[i + 1]
			# Get the operation label from the filtered graph
			label = filtered_graph.edges[u, v]['label']
			operations.append(f"/{label}")  # Format as /2 or /3

		return shortest_path, operations
	except nx.NetworkXNoPath:
		return None, None

def get_all_pairs_operations(G):
	results = []
	
	for src in G.nodes():
		for dst in G.nodes():
			if src > dst:
				path, operations = find_shortest_path_with_operations(G, src, dst)
				if path is not None:
					results.append([src, dst, operations])
	
	return results

if __name__ == "__main__":
	graph_file = "graph_data.json"
	G = load_graph_from_json(graph_file)
	
	all_operations = get_all_pairs_operations(G)
	
	for res in all_operations:
		print(res)
