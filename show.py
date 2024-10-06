import networkx as nx
import matplotlib.pyplot as plt
import json

def load_graph_from_json(file_path):
	with open(file_path, 'r') as f:
		data = json.load(f)
	G = nx.node_link_graph(data)
	return G

if __name__ == "__main__":
	graph_filename = "graph_data.json"
	G = load_graph_from_json(graph_filename)

	edge_labels = nx.get_edge_attributes(G, 'label')
	pos = nx.spring_layout(G, seed=34)
	plt.figure(figsize=(12, 8))
	nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=10, font_weight='bold')
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
	plt.show()