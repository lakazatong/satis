import os, sys, re, math, time, uuid, signal, pathlib, tempfile, matplotlib
import networkx as nx, matplotlib.pyplot as plt, pygraphviz, pydot
from contextlib import redirect_stdout
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph

if sys.platform == 'win32':
	path = pathlib.Path(r'C:\Program Files\Graphviz\bin')
	if path.is_dir() and str(path) not in os.environ['PATH']:
		os.environ['PATH'] += f';{path}'

log_filename = "logs.txt"
logging = False
allowed_divisors = [2, 3]
allowed_divisors_r = allowed_divisors[::-1]
min_sum_count, max_sum_count = allowed_divisors[0], allowed_divisors_r[0]
conveyor_speeds = [60, 120, 270, 480, 780, 1200]
conveyor_speeds_r = conveyor_speeds[::-1]
conveyor_speed_limit = conveyor_speeds_r[0]
short_repr = False
include_depth_informations = False
solution = None

def printt(*args, **kwargs):
	if logging:
		with open(log_filename, "a") as f:
			with redirect_stdout(f):
				print(*args, **kwargs)

if logging:
	open(log_filename, "w").close()
elif os.path.exists(log_filename):
	os.remove(log_filename)

timings = {
	"total": 0,
	"copy_sources": 0,
	"try_divide": 0,
	"try_extract": 0,
	"try_merge": 0,
	"get_sim_without": 0,
	"enqueue": 0,
}

def print_timings():
	total_time = timings["total"]
	for key, val in timings.items():
		if key == "total": continue
		if total_time == 0: continue
		print(f"{key}: {val / total_time:.4f}")

def conclude():
	if solution:
		print(f"\n\n\tSmallest solution found (size = {solution.size}):\n")
		print(solution)
		solution.visualize()
	else:
		print(f"\n\n\tNo solution found? bruh\n")

def handler(signum, frame):
	print("Stopping and printing timing averages...")
	print_timings()
	conclude()
	sys.exit(0)

signal.signal(signal.SIGINT, handler)

def time_block(key, fn, *args, **kwargs):
	start_time = time.time()
	result = fn(*args, **kwargs)
	timings[key] += time.time() - start_time
	return result

def set_time(key, start_time):
	timings[key] += time.time() - start_time

# def safe_add_parent(parent, node):
# 	if node is parent or node.has_parent(parent):
# 		print("self parent")
# 		print(node)
# 		sys.exit(1)
# 	node.parents.append(parent)

def visualize(src, nodes):
	src.children = nodes
	for node in nodes:
		node.parents.append(src)
	src.visualize()

def sort_nodes(nodes):
	return sorted(nodes, key=lambda node: node.value)

def get_node_values(nodes):
	return list(map(lambda node: node.value, nodes))

def get_node_ids(nodes):
	return list(map(lambda node: node.node_id, nodes))

def get_short_node_ids(nodes, short=3):
	return list(map(lambda node: node.node_id[-short:], nodes))

def pop(node, nodes):
	for i, other in enumerate(nodes):
		if other.node_id == node.node_id:
			return nodes.pop(i)
	return None

def can_split(value, divisor):
	global allowed_divisors
	if not divisor in allowed_divisors: return False
	return value % divisor == 0

def increment(binary_array):
	for i in range(len(binary_array)):
		binary_array[i] = not binary_array[i]
		if binary_array[i]:
			return True
	return False

def insert_into_sorted(sorted_list, item, key=lambda x: x):
	low, high = 0, len(sorted_list)
	while low < high:
		mid = low + (high - low) // 2
		if key(item) > key(sorted_list[mid]):
			low = mid + 1
		else:
			high = mid
	sorted_list.insert(low, item)

class Node:
	def __init__(self, value, node_id=None):
		if value < 0: raise ValueError("negative value")
		self.value = value
		self.node_id = node_id if node_id is not None else str(uuid.uuid4())
		self.depth = 1
		self.tree_height = 1
		self.level = None
		self.size = None
		self.parents = []
		self.children = []

	def __repr__(self):
		if short_repr:
			return f"{"\t" * (self.depth - 1)}{self.value}({self.node_id[-3:]})"
		r = f"{"\t" * (self.depth - 1)}Node(value={self.value}, short_node_id={self.node_id[-3:]}"
		if include_depth_informations:
			r += f", depth={self.depth}, tree_height={self.tree_height}, level={self.level}, size={self.size}"
		r += f", parents={get_short_node_ids(self.parents)}, children=["
		if self.children:
			r += "\n"
			for child in self.children:
				r += str(child) + "\n"
			r += "\t" * (self.depth - 1)
		r += "])"
		return r

	def get_leaves(self):
		if not self.children:
			return [self]
		leaves = []
		for child in self.children:
			leaves.extend(child.get_leaves())
		return leaves

	def get_root(self):
		cur = self
		while cur.parents:
			cur = cur.parents[0]
		return cur

	def _compute_size(self, visited):
		# the returned boolean is there to indicate if the returned size is unseen for parents
		if self.node_id in visited:
			return self.size, False
		visited.add(self.node_id)
		self.size = 1
		unseen_size = self.size
		for child in self.children:
			child_size, child_unvisited = child._compute_size(visited)
			if child_unvisited:
				unseen_size += child_size
			self.size += child_size
		return unseen_size, True

	def compute_size(self):
		self.get_root()._compute_size(set())
		return self.size

	def _deepcopy(self, copied_nodes):
		if self.node_id in copied_nodes:
			return copied_nodes[self.node_id], []

		new_node = Node(self.value, node_id=self.node_id)
		new_node.depth = self.depth
		new_node.tree_height = self.tree_height
		new_node.level = self.level
		new_node.size = self.size

		copied_nodes[self.node_id] = new_node

		leaves = []
		leave_ids = []

		if self.children:
			for child in self.children:
				new_child, child_leaves = child._deepcopy(copied_nodes)
				if new_child in new_node.children:
					print("wtf")
					sys.exit(1)

				new_node.children.append(new_child)

				if new_node in new_child.parents:
					print("nooooo")
					sys.exit(1)

				new_child.parents.append(new_node)

				for child_leave in child_leaves:
					if child_leave.node_id not in leave_ids:
						leaves.append(child_leave)
						leave_ids.append(child_leave.node_id)
		else:
			leaves.append(new_node)
			leave_ids.append(new_node.node_id)

		return new_node, leaves

	def deepcopy(self):
		return self.get_root()._deepcopy({})

	def find(self, node_id):
		if self.node_id == node_id: return self
		for child in self.children:
			result = child.find(node_id)
			if result: return result
		return None

	def has_parent(self, parent):
		for p in self.parents:
			if p is parent or p.has_parent(parent): return True
		return False

	def _compute_depth_and_tree_height(self):
		self.depth = 1 + (max(parent.depth for parent in self.parents) if self.parents else 0)
		max_child_tree_height = 0
		for child in self.children:
			_, child_tree_height = child._compute_depth_and_tree_height()
			if child_tree_height > max_child_tree_height:
				max_child_tree_height = child_tree_height
		# Include the current node level in the height calculation
		self.tree_height = max_child_tree_height + 1
		return self.depth, self.tree_height

	# def set_max_tree_height(self, max_tree_height):
	# 	self.max_tree_height = max_tree_height
	# 	for child in self.children:
	# 		child.set_max_tree_height(max_tree_height)

	def compute_level(self, max_tree_height):
		self.level = max_tree_height - \
		(max(map(lambda node: node.tree_height, self.children)) if self.children else 0)
		for child in self.children:
			child.compute_level(max_tree_height)

	def _compute_depth_informations(self):
		self._compute_depth_and_tree_height()
		# self.set_max_tree_height(self.tree_height)
		self.compute_level(self.tree_height)

	def compute_depth_informations(self):
		self._compute_depth_informations(self.get_root())

	def populate(self, G):
		G.add_node(self.node_id, label=str(self.value), level=self.level)
		for child in self.children:
			G.add_edge(self.node_id, child.node_id)
			child.populate(G)

	def visualize(self):
		try:
			G = nx.DiGraph()
			self.populate(G)

			A = to_agraph(G)

			for node in G.nodes:
				level = G.nodes[node]['level']
				A.get_node(node).attr['rank'] = f'{level}'

			for level in set(nx.get_node_attributes(G, 'level').values()):
				subgraph = A.add_subgraph(
					[n for n, attr in G.nodes(data=True) if attr['level'] == level],
					rank='same'
				)

			A.graph_attr['rankdir'] = 'TB'

			# Invert colors
			A.graph_attr['bgcolor'] = 'black'
			for node in A.nodes():
				node.attr['color'] = 'white'
				node.attr['fontcolor'] = 'white'
				node.attr['style'] = 'filled'
				node.attr['fillcolor'] = 'black'

			for edge in A.edges():
				edge.attr['color'] = 'white'

			with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
				A.layout(prog='dot')
				A.draw(tmpfile.name, format='png')
				plt.figure(figsize=(10, 7))
				img = plt.imread(tmpfile.name)
				plt.imshow(img)
				plt.axis('off')

				# Remove margins
				plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

				if hasattr(sys, '_MEIPASS'):
					# matplotlib.use('Agg')
					plt.savefig("graph_visualization.png", bbox_inches='tight', pad_inches=0)
					print("Graph saved as 'graph_visualization.png'")
				else:
					plt.show()
		except:
			print("call compute_depth_informations before trying to visualize")
			sys.exit(1)


	def merge_down(self, other):
		new_value = self.value + sum(get_node_values(other))
		new_node = Node(new_value)
		self.children.append(new_node)
		new_node.parents.append(self)
		for node in other:
			node.children.append(new_node)
			new_node.parents.append(node)
		return new_node

	def merge_up(self, other):
		new_value = self.value + sum(get_node_values(other))
		new_node = Node(new_value)
		self.parents.append(new_node)
		new_node.children.append(self)
		for node in other:
			node.parents.append(new_node)
			new_node.children.append(node)
		return new_node

	def can_split(self, divisor):
		return can_split(self.value, divisor)

	def split_down(self, divisor):
		new_value = int(self.value / divisor)
		new_nodes = [Node(new_value) for _ in range(divisor)]
		for node in new_nodes:
			self.children.append(node)
			node.parents.append(self)
		return new_nodes

	def split_up(self, divisor):
		new_value = int(self.value / divisor)
		new_nodes = [Node(new_value) for _ in range(divisor)]
		for node in new_nodes:
			self.parents.append(node)
			node.children.append(self)
		return new_nodes

	def extract_down(self, value):
		# print("caca2")
		# print(self)
		extracted_node = Node(value)
		overflow_value = self.value - value
		overflow_node = Node(overflow_value)
		self.children.append(extracted_node)
		self.children.append(overflow_node)
		extracted_node.parents.append(self)
		overflow_node.parents.append(self)
		return [extracted_node, overflow_node]

	def extract_up(self, value):
		extracted_node = Node(value)
		overflow_value = self.value - value
		overflow_node = Node(overflow_value)
		self.parents.append(extracted_node)
		self.parents.append(overflow_node)
		extracted_node.children.append(self)
		overflow_node.children.append(self)
		return [extracted_node, overflow_node]

	def __add__(self, other):
		if isinstance(other, list) and all(isinstance(node, Node) for node in other):
			return self.merge_down(other)
		elif isinstance(other, Node):
			return self.merge_down([other])		
		raise ValueError("Operand must be a Node[] or Node")

	def __truediv__(self, divisor):
		if isinstance(divisor, (int, float)):
			return self.split_down(divisor)
		raise ValueError(f"Divisor must be an integer (one of these: {"/".join(allowed_divisors)})")

	def __sub__(self, amount):
		if isinstance(amount, (int, float)):
			return self.extract_down(amount)
		raise ValueError("Amount must be a number")

def load_tree(tree_representation):
	node_dict = {}

	def parse_node(node_str):
		match = re.search(r'Node\(value=(\d+), short_node_id=([-0-9a-f]+)', node_str)
		if match:
			value, short_id = int(match.group(1)), match.group(2)
			if short_id not in node_dict:
				node_dict[short_id] = Node(value)
			return node_dict[short_id]
		return None

	# def parse_children(children_str):
	# 	children = []
	# 	for child_str in re.findall(r'Node\(value=\d+, short_node_id=[-0-9a-f]+.*?\)', children_str):
	# 		children.append(parse_node(child_str))
	# 	return children

	stack = []
	root = None

	for line in tree_representation.splitlines():
		indent_level = len(line) - len(line.lstrip())
		if 'Node' in line:
			node = parse_node(line)
			if not root:
				root = node
			if stack and indent_level > stack[-1][0]:
				parent = stack[-1][1]
				parent.children.append(node)
				node.parents.append(parent)
			else:
				while stack and indent_level <= stack[-1][0]:
					stack.pop()
				if stack:
					parent = stack[-1][1]
					parent.children.append(node)
					node.parents.append(parent)
			stack.append((indent_level, node))

	return root

def _simplify_merge(nodes):
	global allowed_divisors_r
	# Step 1: Merge nodes with the same value until all are different
	has_merged = False
	while True:
		merged_nodes = []
		done = True
		i = 0

		while i < len(nodes):
			current_node = nodes[i]
			current_value = current_node.value
			same_value_nodes = []

			i += 1
			while i < len(nodes) and nodes[i].value == current_value:
				if len(same_value_nodes) == allowed_divisors_r[0] - 1:
					break
				same_value_nodes.append(nodes[i])
				i += 1

			if len(same_value_nodes) > 0:
				merged_node = current_node.merge_up(same_value_nodes)
				merged_nodes.append(merged_node)
				done = False
				has_merged = True
			else:
				merged_nodes.append(current_node)

		if done: break

		merged_nodes = sort_nodes(merged_nodes)
		nodes = [node for node in merged_nodes]
	return nodes, has_merged

def _simplify_extract(nodes):
	global conveyor_speeds_r
	# Step 2: Extract maximum conveyor speed that fits (ignore nodes with value already equal to a conveyor speed)
	extracted_nodes = []
	for node in nodes:
		extracted_flag = False
		for speed in conveyor_speeds_r:
			if node.value == speed: break
			if node.value > speed:
				extracted_node, overflow_node = node.extract_up(speed)
				extracted_nodes.append(extracted_node)
				extracted_nodes.append(overflow_node)
				extracted_flag = True
				break
		if not extracted_flag:
			extracted_nodes.append(node)

	nodes = sort_nodes(extracted_nodes)
	return nodes

def simplify(nodes):
	nodes, has_merged = _simplify_merge(nodes)
	nodes = _simplify_extract(nodes)
	while has_merged:
		nodes, has_merged = _simplify_merge(nodes)
		if not has_merged: break
		nodes = _simplify_extract(nodes)
	return nodes

def _get_sim_without(sources, value):
	sim = []
	found = False
	for src in sources:
		if src.value == value and not found:
			found = True
		else:
			sim.append(src.value)
	return sim

def get_sim_without(sources, value):
	return time_block("get_sim_without", _get_sim_without, sources, value)

def solution_found(new_solution_root):
	global solution
	new_solution_root._compute_size(set())
	if solution is None or new_solution_root.size < solution.size:
		solution = new_solution_root
	else:
		print("impossible case reached, should have been checked already")
		sys.exit(1)
	print(f"\n\n\tSolution of size {solution.size} found\n")
	print(solution)
	# solution.visualize()

def _solve(source_values, target_values, starting_node_sources=None):
	print(f"\nsolving: {sorted(source_values)} to {sorted(target_values)}\n")
	# steps = -1

	# enqueued_sims = set()
	target_values = sorted(target_values)
	target_counts = {
		value: target_values.count(value) for value in set(target_values)
	}
	gcd = math.gcd(*target_values)
	# node_targets = list(map(lambda value: Node(value), target_values))
	# print('\n'.join([str(src) for src in copy]))

	def gcd_incompatible(value):
		nonlocal gcd
		return value < gcd or value % gcd != 0

	filtered_conveyor_speeds = [speed for speed in conveyor_speeds if not gcd_incompatible(speed)]
	filtered_conveyor_speeds_r = filtered_conveyor_speeds[::-1]
	print(f"gcd = {gcd}, filtered_conveyor_speeds = {filtered_conveyor_speeds}")

	node_sources = None
	if starting_node_sources:
		node_sources = starting_node_sources
	else:
		node_sources = list(map(lambda value: Node(value), source_values))
		if len(node_sources) > 1:
			root = Node(sum(source_values))
			root.children = node_sources
			for child in root.children:
				child.parents.append(root)

	queue = []

	def get_extract_sim(sources, i):
		global solution
		src = sources[i]
		simulations = []
		tmp_sim = None
		sources_root = None
		parent_values = get_node_values(src.parents)
		
		for speed in filtered_conveyor_speeds:
			if src.value <= speed: break
			
			# if so then it would have been better to leave it as is
			# and merge all the other values to get the overflow value
			# we would get by exctracting speed amount
			if speed in parent_values: continue

			if solution:
				if not sources_root:
					sources_root = sources[0].get_root()
					sources_root._compute_size(set())
				if sources_root.size + 2 >= solution.size: continue
			
			overflow_value = src.value - speed
			if gcd_incompatible(overflow_value): continue

			tmp_sim = tmp_sim if tmp_sim else get_sim_without(sources, src.value)
			sim = tuple(tmp_sim + [speed, overflow_value])
			# if sim in enqueued_sims: continue
			simulations.append((sim, (i, speed)))
		
		return simulations

	def get_extract_sims(sources, cant_use):
		simulations = []
		seen_values = set()
		n = len(sources)

		for i in range(n):
			src = sources[i]
			if cant_use[src.value] or src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(get_extract_sim(sources, i))

		return simulations

	def get_divide_sim(sources, i):
		global solution, allowed_divisors_r
		src = sources[i]
		simulations = []
		tmp_sim = None
		sources_root = None

		for divisor in allowed_divisors_r:
			if not src.can_split(divisor): continue

			if sum(get_node_values(src.parents)) == src.value and len(src.parents) == divisor: continue

			if solution:
				if not sources_root:
					sources_root = sources[0].get_root()
					sources_root._compute_size(set())
				if sources_root.size + divisor >= solution.size: continue
			
			divided_value = int(src.value / divisor)
			if gcd_incompatible(divided_value): continue

			tmp_sim = tmp_sim if tmp_sim else get_sim_without(sources, src.value)
			sim = tuple(tmp_sim + [divided_value] * divisor)
			# if sim in enqueued_sims: continue
			simulations.append((sim, (i, divisor)))

		return simulations

	def get_divide_sims(sources, cant_use):
		simulations = []
		seen_values = set()
		n = len(sources)
		
		for i in range(n):
			src = sources[i]
			if cant_use[src.value] or src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(get_divide_sim(sources, i))
		
		return simulations

	def get_merge_sim(sources, flags, to_sum_count, seen_sums, cant_use, log=False):
		to_not_sum_indices = []
		i = 0
		n = len(sources)
		
		while not flags[i]:
			to_not_sum_indices.append(i)
			i += 1
		
		src = sources[i]
		
		if cant_use[src.value]: return None
		
		to_sum_indices = [i]
		parent = src.parents[0]
		same_parent = len(src.parents) == 1
		
		while i < n - 1:
			i += 1
			
			try:
				if not flags[i]:
					to_not_sum_indices.append(i)
					continue
			except:
				print("binary?")
				sys.exit(1)

			src = sources[i]
			
			if cant_use[src.value]: return None
			
			if len(src.parents) != 1 or not src.parents[0] is parent:
				same_parent = False
			
			to_sum_indices.append(i)

		if same_parent and to_sum_count == len(src.parents[0].children):
			# can happen that the parent was the artificial root created to unify all sources
			# in this case only we don't skip
			if parent.parents or len(source_values) == 1: return None

		to_sum_values = sorted([sources[i].value for i in to_sum_indices])
		summed_value = sum(to_sum_values)
		if gcd_incompatible(summed_value) or summed_value > conveyor_speed_limit: return None
		
		to_sum_values = tuple(sorted(to_sum_values))
		if to_sum_values in seen_sums: return None
		seen_sums.add(to_sum_values)

		sim = tuple([sources[i].value for i in to_not_sum_indices] + [summed_value])
		# if sim in enqueued_sims: return None
		
		return sim, to_sum_indices

	def get_merge_sims(sources, cant_use, log=False):
		global min_sum_count, max_sum_count, solution
		simulations, n = [], len(sources)
		
		if n < 2: return simulations
		
		seen_sums = set()
		binary = [False] * n
		binary[1] = True
		sources_root = None
		
		while increment(binary):
			to_sum_count = sum(binary)
			
			if to_sum_count < min_sum_count or to_sum_count > max_sum_count: continue
			if solution:
				# print("coucou")
				if not sources_root:
					sources_root = sources[0].get_root()
					sources_root._compute_size(set())
				if sources_root.size + 1 >= solution.size: continue
			
			# if log:
			# 	print("coucou start")
			# 	print("sources", sources)
			# 	print("binary", binary)
			# 	print("to_sum_count", to_sum_count)
			# 	print("seen_sums", seen_sums)
			# 	print("cant_use", cant_use)
			r = get_merge_sim(sources, binary, to_sum_count, seen_sums, cant_use, log)
			# if log:
			# 	print("r", r)
			# 	print("coucou end")
			if r: simulations.append(r)
		
		return simulations

	def compute_cant_use(sources):
		nonlocal target_counts
		source_counts = {}
		for src in sources:
			if src.value in source_counts:
				source_counts[src.value] += 1
			else:
				source_counts[src.value] = 1
		cant_use = {}
		for src in sources:
			value = src.value
			src_count = source_counts.get(value, None)
			target_count = target_counts.get(value, None)
			cant_use[value] = max(0, src_count - target_count) == 0 if src_count and target_count else False
		return cant_use

	def compute_distance(sim):
		nonlocal target_values, filtered_conveyor_speeds_r
		global allowed_divisors, allowed_divisors_r
		sim = list(sim)
		targets = target_values[:]
		distance = 0
		# debug = sim == [40, 70, 70, 70]
		# if debug:
		# 	print("\n\nSIM\n", sim)
		# 	print("TARGETS\n", targets)
		# 	print("DISTANCE\n", distance)
		
		while True:
			done = True
			for value in sim[:]:
				if value in targets:
					sim.remove(value)
					targets.remove(value)
					done = False
			
			if not sim and not targets: return distance
			# if debug:
			# 	print("\n\nSIM\n", sim)
			# 	print("TARGETS\n", targets)
			# 	print("DISTANCE\n", distance)

			# remove all perfect extractions
			for speed in filtered_conveyor_speeds_r:
				if not speed in targets: continue

				for value in sim:
					if value <= speed: continue
					overflow = value - speed
					if gcd_incompatible(overflow) or not overflow in targets: continue
					sim.remove(value)
					targets.remove(speed)
					targets.remove(overflow)
					distance += 1
					done = False
			
			if not sim and not targets: return distance
			# if debug:
			# 	print("\n\nSIM\n", sim)
			# 	print("TARGETS\n", targets)
			# 	print("DISTANCE\n", distance)
			
			# remove all non perfect extractions
			for speed in filtered_conveyor_speeds:
				for value in sim:
					if value <= speed: continue
					overflow = value - speed
					if gcd_incompatible(overflow): continue
					if speed in targets:
						if overflow in targets:
							print("impossible case reached, all perfect extractions were removed already")
							sys.exit(1)
						sim.remove(value)
						targets.remove(speed)
						sim.append(overflow)
					elif overflow in targets:
						if speed in targets:
							print("impossible case reached, all perfect extractions were removed already")
							sys.exit(1)
						sim.remove(value)
						targets.remove(overflow)
						sim.append(speed)
					else:
						continue
					distance += 2
					done = False
			
			# if not sim and not targets: return distance
			# if debug:
			# 	print("\n\nSIM\n", sim)
			# 	print("TARGETS\n", targets)
			# 	print("DISTANCE\n", distance)

			def try_divide(value, divisor):
				nonlocal targets
				if value % divisor != 0: return None, 0
				
				divided_value = value // divisor
				
				if gcd_incompatible(divided_value): return None, 0

				matches, remaining_targets = 0, targets[:]

				for _ in range(divisor):
					if divided_value in remaining_targets:
						matches += 1
						remaining_targets.remove(divided_value)

				return divided_value, matches

			# remove all perfect divisions
			for divisor in allowed_divisors_r:
				for value in sim[:]:
					divided_value, matches = try_divide(value, divisor)
					if not matches or matches != divisor: continue
					sim.remove(value)
					for _ in range(matches): targets.remove(divided_value)
					distance += 1
					done = False
			
			if not sim and not targets: return distance
			# if debug:
			# 	print("\n\nSIM\n", sim)
			# 	print("TARGETS\n", targets)
			# 	print("DISTANCE\n", distance)
			
			# remove all divisions that have at least one match
			for divisor in allowed_divisors:
				for value in sim[:]:
					divided_value, matches = try_divide(value, divisor)
					if not matches: continue
					if matches == divisor:
						print("impossible case reached, all perfect divisions were removed already")
						sys.exit(1)
					sim.remove(value)
					extras = divisor - matches
					for _ in range(matches): targets.remove(divided_value)
					for _ in range(extras): sim.append(divided_value)
					distance += 1 + extras
					done = False
			
			if not sim and not targets: return distance
			# if debug:
			# 	print("\n\nSIM\n", sim)
			# 	print("TARGETS\n", targets)
			# 	print("DISTANCE\n", distance)

			# remove all sums that yield a target
			for target in targets[:]:
				if len(sim) < 2: break
				binary = [False] * len(sim)
				binary[1] = True

				while increment(binary):
					to_sum_count = sum(binary)
					if to_sum_count < min_sum_count or to_sum_count > max_sum_count: continue

					to_sum_values = [sim[i] for i, b in enumerate(binary) if b]

					if sum(to_sum_values) != target: continue

					targets.remove(target)
					for val in to_sum_values: sim.remove(val)

					distance += 1
					done = False
					break
			
			# if debug:
			# 	print("\n\nSIM\n", sim)
			# 	print("TARGETS\n", targets)
			# 	print("DISTANCE\n", distance)
			if done: break
		
		return distance + len(sim) + len(targets)

	def is_solution(sources):
		nonlocal target_values
		n = len(sources)
		if n != len(target_values): return False
		for i in range(n):
			if sources[i].value != target_values[i]:
				return False
		return True
		# # Link the simplified targets' trees with the current one
		# for target in node_targets:
		# 	for child in target.children:
		# 		child.parents = []
		# for i in range(n):
		# 	src = sources[i]
		# 	src.children = node_targets[i].children
		# 	for child in src.children:
		# 		src.parents.append(child)

	# computes how close the sources are from the target_values
	# the lower the better
	def compute_sources_score(sources):
		if is_solution(sources): return 0
		n = len(sources)
		simulations = []
		cant_use = compute_cant_use(sources)
		simulations.extend(get_extract_sims(sources, cant_use))
		simulations.extend(get_divide_sims(sources, cant_use))
		simulations.extend(get_merge_sims(sources, cant_use))
		score = -1
		# even if one simulation matches the targets and has a score of 0
		# it required at least one operation to get there, hence the 1 +
		if simulations: score = 1 + min(compute_distance(sim) for sim, _ in simulations)
		return score

	def _enqueue(nodes):
		nonlocal queue
		nodes = sort_nodes(nodes)
		score = compute_sources_score(nodes)
		# print("score", score, nodes)
		if score < 0: return
		# print("queue = ")
		# for e in queue:
		# 	print("e = ", e)
		insert_into_sorted(queue, (nodes, score), key=lambda x: x[1])
		# print("queue = ")
		# for e in queue:
		# 	print("e = ", e)

	def enqueue(nodes):
		time_block("enqueue", _enqueue, nodes)

	# will be popped just after, no need to compute the score here
	queue.append((node_sources, 1 << 16))
	lowest_score = 1000

	while queue:
		start_total = time.time()
		tmp, score = queue.pop(0)
		sources = sort_nodes(tmp)
		sources_root = sources[0].get_root()
		sources_root._compute_depth_informations()
		sources_root._compute_size(set())
		if score == 0:
			solution_found(sources_root)
			continue
		elif score < lowest_score:
			print(f"\n\n\tlowest score = {score}, tree =\n")
			print(sources_root)
			lowest_score = score

		n = len(sources)
		cant_use = compute_cant_use(sources)

		# steps -= 1
		# if steps + 1 == 0:
		# 	print("stopping")
		# 	print_timings()
		# 	sys.exit(0)
		# if (-steps) % 1000 == 0:
		# 	print(f"step {abs(steps)}")

		def _copy_sources():
			nonlocal sources, sources_root
			_, leaves = sources_root._deepcopy({})
			return sort_nodes([leaf for leaf in leaves if leaf.node_id in get_node_ids(sources)])

		def copy_sources():
			return time_block("copy_sources", _copy_sources)

		def _try_extract():
			nonlocal sources, cant_use, sources_root
			simulations = get_extract_sims(sources, cant_use)
			# print('extract', simulations)
			for sim, (i, speed) in simulations:
				copy = copy_sources()
				src_copy = copy[i]
				
				# if sources[i].value == 130 and speed == 60:
				# 	print("caca")
				# 	print(sources[0].get_root())
				# 	print(copy)
				# 	print("\n")
				# 	print(sources[i])
				# 	print(copy[i])
				# 	print("\n")
				
				pop(src_copy, copy)

				printt("\n\nFROM")
				printt(sources_root)
				printt("DID")
				printt(f"{sources[i]} - {speed}")

				sources_to_enqueue = copy + (src_copy - speed)
				enqueue(sources_to_enqueue)

				# if sources[i].value == 130 and speed == 60:
				# 	sources_to_enqueue_root = sources_to_enqueue[0].get_root()
				# 	sources_to_enqueue_root._compute_depth_informations()
				# 	print("sources_to_enqueue")
				# 	print(sources_to_enqueue_root)
				# 	print("sources_to_enqueue ids")
				# 	print(get_node_ids(sources_to_enqueue))
				# 	print(src_copy)
				# 	sys.exit(0)
				
				# enqueued_sims.add(sim)

		def try_extract():
			time_block("try_extract", _try_extract)
		
		def _try_divide():
			nonlocal sources, cant_use, sources_root
			simulations = get_divide_sims(sources, cant_use)
			# print('divide', simulations)
			for sim, (i, divisor) in simulations:
				copy = copy_sources()
				src_copy = copy[i]
				pop(src_copy, copy)

				printt("\n\nFROM")
				printt(sources_root)
				printt("DID")
				printt(f"{sources[i]} / {divisor}")

				enqueue(copy + (src_copy / divisor))
				# enqueued_sims.add(sim)

		def try_divide():
			time_block("try_divide", _try_divide)

		def _try_merge():
			nonlocal sources, cant_use, sources_root
			simulations = get_merge_sims(sources, cant_use)
			# print('merge', simulations, cant_use)
			for sim, to_sum_indices in simulations:
				copy = copy_sources()
				to_sum = [copy[i] for i in to_sum_indices]
				to_sum_values = get_node_values(to_sum)
				list(map(lambda src: pop(src, copy), to_sum))
				summed_node = to_sum[0] + to_sum[1:]
				copy.append(summed_node)
				# print(get_node_values(copy))

				# print("allo?")
				printt("\n\nFROM")
				printt(sources_root)
				printt("DID")
				printt("+".join(str(ts) for ts in to_sum))

				enqueue(copy)
				# enqueued_sims.add(sim)

		def try_merge():
			time_block("try_merge", _try_merge)
		
		# print("aluile", get_node_values(sources))

		try_divide()
		try_extract()
		try_merge()

		# if get_node_values(sources) == [30, 40, 40, 70, 70] and sources_root.tree_height == 5 and sources_root.size == 11:
		# 	print("cacaAAAAAAAA")
		# 	sys.exit(0)

		timings["total"] += time.time() - start_total

def solve(source_values, target_values):
	sources_total = sum(source_values)
	targets_total = sum(target_values)
	if sources_total > targets_total:
		target_values.append(sources_total - targets_total)
	elif sources_total < targets_total:
		source_values.append(targets_total - sources_total)
	_solve(source_values, target_values)

def main():
	global solution
	separator = 'to'
	if len(sys.argv) < 3 or separator not in ' '.join(sys.argv[1:]):
		print(f"Usage: python solve.py <source_args> {separator} <target_args>")
		sys.exit(0)

	source_part, target_part = ' '.join(sys.argv[1:]).split(separator)
	source_args = source_part.strip().split()
	target_args = target_part.strip().split()

	if not source_args:
		print("Error: At least one source value must be provided.")
		sys.exit(1)

	if not target_args:
		print("Error: At least one target value must be provided.")
		sys.exit(1)

	sources = []
	i = 0
	while i < len(source_args):
		src = source_args[i]
		if not src.endswith('x'):
			source_value = int(src)
			if source_value % 5 != 0:
				print("Error: all values must be multiples of 5")
				sys.exit(1)
			sources.append(source_value)
			i += 1
			continue
		if len(src) < 2 or not src[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			sys.exit(1)
		multiplier = int(src[:-1])
		source_value = int(source_args[source_args.index(src) + 1])
		if source_value % 5 != 0:
			print("Error: all values must be multiples of 5")
			sys.exit(1)
		for _ in range(multiplier):
			sources.append(source_value)
		i += 2

	targets = []
	i = 0
	while i < len(target_args):
		target = target_args[i]
		if not target.endswith('x'):
			target_value = int(target)
			if target_value % 5 != 0:
				print("Error: all values must be multiples of 5")
				sys.exit(1)
			targets.append(target_value)
			i += 1
			continue
		if len(target) < 2 or not target[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			sys.exit(1)
		multiplier = int(target[:-1])
		if i + 1 == len(target_args):
			print("Error: You must provide a target value after Nx.")
			sys.exit(1)
		target_value = int(target_args[i + 1])
		if target_value % 5 != 0:
			print("Error: all values must be multiples of 5")
			sys.exit(1)
		for _ in range(multiplier):
			targets.append(target_value)
		i += 2

	solve(sources, targets)
	conclude()

def test():
	root = load_tree("""Node(value=250, short_node_id=93d, parents=[], children=[
	Node(value=120, short_node_id=575, parents=['93d'], children=[])
	Node(value=130, short_node_id=c62, parents=['93d'], children=[])
])""")
	root._compute_depth_informations()
	print(root)
	leaves = root.get_leaves()
	_solve(get_node_values(leaves), [70, 70, 70, 40], leaves)
	conclude()

if __name__ == '__main__':
	# test()
	main()