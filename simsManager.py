import os, itertools, json, pickle

from utils import remove_pairs, get_sim_without, get_node_ids, insert_into_sorted, get_gcd_incompatible
from config import config
from node import Node

class SimsManager:
	def __init__(self, solver):
		self.solver = solver
		# independant from any problem but the config's conveyor_speeds and allowed_divisors
		# which should not change since it's meant to solve a satisfactory problem lol
		self.extract_sims_cache = {}
		self.divide_sims_cache = {}
		self.merge_sims_cache = {}

	def load_cache(self):
		if not os.path.isfile(config.cache_filepath): return
		
		cache = None
		with open(config.cache_filepath, "rb") as f: cache = pickle.load(f)
		if not cache: return

		cached_conveyor_speeds = sorted(cache["conveyor_speeds"])
		cached_allowed_divisors = sorted(cache["allowed_divisors"])

		if cached_conveyor_speeds == sorted(list(config.conveyor_speeds)):
			self.extract_sims_cache = cache["extract_sims_cache"]
		else:
			print("Invalidated extract_sims_cache cache because config.conveyor_speeds has changed")
		
		if cached_allowed_divisors == sorted(list(config.allowed_divisors)):
			self.divide_sims_cache = cache["divide_sims_cache"]
		else:
			print("Invalidated divide_sims_cache cache because config.allowed_divisors has changed")
		
		if max(cached_conveyor_speeds) == config.conveyor_speed_limit:
			self.merge_sims_cache = cache["merge_sims_cache"]
		else:
			print("Invalidated merge_sims_cache cache because config.conveyor_speed_limit has changed")

	def save_cache(self):
		cache = {
			"conveyor_speeds": list(config.conveyor_speeds),
			"allowed_divisors": list(config.allowed_divisors),
			"extract_sims_cache": self.extract_sims_cache,
			"divide_sims_cache": self.divide_sims_cache,
			"merge_sims_cache": self.merge_sims_cache
		}
		with open(config.cache_filepath, "wb") as f: pickle.dump(cache, f)

	def get_extract_sim(self, sources, i):
		src = sources[i]
		simulations = []
		tmp_sim = None
		for speed in config.conveyor_speeds:
			if src.value <= speed: break
			overflow = src.value - speed
			if not tmp_sim: tmp_sim = get_sim_without(src.value, sources)
			sim = tuple(sorted(tmp_sim + [speed, overflow]))
			simulations.append((sim, (i, speed)))
		return simulations

	def get_extract_sims(self, tree):
		cached_simulations = self.extract_sims_cache.get(tree.source_values)
		if cached_simulations: return cached_simulations
		simulations = []
		sources = tree.sources
		n = len(sources)
		seen_values = set()
		for i in range(n):
			src = sources[i]
			if src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(self.get_extract_sim(sources, i))
		self.extract_sims_cache[tree.source_values] = simulations
		return simulations

	def get_divide_sim(self, sources, i):
		src = sources[i]
		simulations = []
		tmp_sim = None
		sources_root = None
		for divisor in config.allowed_divisors:
			if not src.can_split(divisor): continue
			divided_value = src.value // divisor
			if not tmp_sim: tmp_sim = get_sim_without(src.value, sources)
			sim = tuple(sorted(tmp_sim + [divided_value] * divisor))
			simulations.append((sim, (i, divisor)))
		return simulations

	def get_divide_sims(self, tree):
		cached_simulations = self.divide_sims_cache.get(tree.source_values)
		if cached_simulations: return cached_simulations
		simulations = []
		sources = tree.sources
		n = len(sources)
		seen_values = set()
		for i in range(n):
			src = sources[i]
			if src.value in seen_values: continue
			seen_values.add(src.value)
			simulations.extend(self.get_divide_sim(sources, i))
		self.divide_sims_cache[tree.source_values] = simulations
		return simulations

	def get_merge_sims(self, tree):
		cached_simulations = self.merge_sims_cache.get(tree.source_values)
		if cached_simulations: return cached_simulations
		simulations = []
		sources = tree.sources
		n = len(sources)
		seen_sims = set()
		indices = range(n)
		for to_sum_count in range(config.min_sum_count, config.max_sum_count + 1):
			for to_sum_indices in itertools.combinations(indices, to_sum_count):
				to_sum_indices = list(to_sum_indices)
				to_sum_nodes = [sources[i] for i in to_sum_indices]
				node_ids = get_node_ids(to_sum_nodes)
				sim = sorted(node.value for node in sources if node.node_id not in node_ids)
				summed_value = sum(node.value for node in to_sum_nodes)
				if summed_value > config.conveyor_speed_limit: continue
				insert_into_sorted(sim, summed_value)
				sim = tuple(sim)
				if sim in seen_sims: continue
				seen_sims.add(sim)
				simulations.append((sim, (to_sum_indices, to_sum_count)))
		self.merge_sims_cache[tree.source_values] = simulations
		return simulations

	# doesn't have to be generic
	def compute_distance(self, sim, target_values):
		if sim in self.solver.compute_distances_cache: return self.solver.compute_distances_cache[sim]
		original_sim = sim
		sim = list(sim)
		targets = target_values[:]
		distance = 0
		
		# remove common elements
		sim, targets = remove_pairs(sim, targets)

		sim_set = set(sim)
		possible_extractions = [
			(value, speed, overflow)
			for speed in self.solver.filtered_conveyor_speeds_r
			for value in sim_set
			if (overflow := value - speed) \
				and value > speed \
				and not self.solver.gcd_incompatible(overflow) \
				and (speed in targets or overflow in targets)
		]

		# remove perfect extractions
		for i in range(len(possible_extractions)-1, -1, -1):
			value, speed, overflow = possible_extractions[i]
			if value not in sim: continue
			if speed == overflow:
				if len([v for v in targets if v == speed]) < 2: continue
			else:
				if speed not in targets or overflow not in targets: continue
			sim.remove(value)
			targets.remove(speed)
			targets.remove(overflow)
			distance += 1
			possible_extractions.pop(i)

		# remove unperfect extractions
		for value, speed, overflow in possible_extractions:
			if value not in sim: continue
			if speed in targets:
				sim.remove(value)
				targets.remove(speed)
				sim.append(overflow)
				distance += 2
			elif overflow in targets:
				sim.remove(value)
				targets.remove(overflow)
				sim.append(speed)
				distance += 2

		sim_set = set(sim)
		possible_divisions = sorted([
			(value, divisor, divided_value, min(divisor, sum(1 for v in targets if v == divided_value)))
			for divisor in self.solver.allowed_divisors_r
			for value in sim_set
			if (divided_value := value // divisor) \
				and value % divisor == 0 \
				and not self.solver.gcd_incompatible(divided_value) \
				and divided_value in targets
		], key=lambda x: x[3]-x[1])

		# remove perfect divisions
		for i in range(len(possible_divisions)-1, -1, -1):
			value, divisor, divided_value, divided_values_count = possible_divisions[i]
			if divided_values_count != divisor: break
			if value not in sim or len([v for v in targets if v == divided_value]) < divisor: continue
			sim.remove(value)
			for _ in range(divided_values_count): targets.remove(divided_value)
			possible_divisions.pop(i)
			distance += 1
		
		# remove unperfect divisions
		for i in range(len(possible_divisions)-1, -1, -1):
			value, divisor, divided_value, divided_values_count = possible_divisions[i]
			if value not in sim or len([v for v in targets if v == divided_value]) < divided_values_count: continue
			sim.remove(value)
			for _ in range(divided_values_count): targets.remove(divided_value)
			for _ in range(divisor - divided_values_count): sim.append(divided_value)
			distance += 2

		# remove all possible merges that yield a target, prioritizing the ones that merge the most amount of values in sim
		for target in reversed(sorted(targets[:])):
			possible_merges = sorted([
				comb
				for r in range(config.min_sum_count, config.max_sum_count + 1)
				for comb in itertools.combinations(sim, r)
				if sum(comb) == target
			], key=lambda x: len(x))
			if possible_merges:
				comb = possible_merges[-1]
				for v in comb: sim.remove(v)
				targets.remove(target)
				distance += 1

		r = distance + len(sim) + len(targets)
		self.solver.compute_distances_cache[original_sim] = r
		return r