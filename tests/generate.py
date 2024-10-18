import networkx as nx
import matplotlib.pyplot as plt
import json, os

G = nx.Graph()
nodes = [60, 120, 270, 480, 780, 1200]
last = 0
G.add_nodes_from(nodes)

operation_labels = ["2", "3"]

while True:
	old_last = last
	last = len(nodes)
	done = True
	for i in range(old_last, len(nodes)):
		done = False
		node = nodes[i]
		operations = [node / 2, node / 3, node * 2, node * 3]
		for i in range(len(operations)):
			result = operations[i]
			if result != int(result): continue
			result = int(result)
			if result > 2000: continue
			if result not in nodes:
				nodes.append(result)
				G.add_node(result)
				print(result, 'added')
			label = operation_labels[i % 2]
			if not G.has_edge(node, result): G.add_edge(node, result, label=label)
	if done: break

print('done')

old_content = None
graph_filename = "graph_data.json"
if os.path.exists(graph_filename):
	with open(graph_filename, "r") as f:
		old_content = f.read()

content = nx.node_link_data(G)
content['nodes'] = sorted(content['nodes'], key=lambda x: x['id'])
content['links'] = sorted(content['links'], key=lambda x: (x['source'], x['target']))

with open(graph_filename, "w") as f:
	json.dump(content, f)

old_content_json = json.loads(old_content) if old_content else {}
print('different', content != old_content_json)
