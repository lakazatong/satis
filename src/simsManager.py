import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools, json, pickle

from utils import remove_pairs, get_sim_without, get_node_ids, insert_into_sorted, get_gcd_incompatible, can_split
from config import config
from node import Node

class SimsManager:
	def __init__(self, solver=None):
		self.solver = solver
		# independant from any problem but the config's conveyor_speeds and allowed_divisors
		# which should not change since it's meant to solve a satisfactory problem lol
		self.extract_sims_cache = {}
		self.divide_sims_cache = {}
		self.merge_sims_cache = {}

	def load_cache(self):
		if not os.path.isfile(config.cache_filepath): return
		
		print("Loading cache...")

		cache = None
		with open(config.cache_filepath, "rb") as f: cache = pickle.load(f)
		if not cache: return

		cached_conveyor_speeds = cache["conveyor_speeds"]
		cached_allowed_divisors = cache["allowed_divisors"]

		if cached_conveyor_speeds == config.conveyor_speeds:
			self.extract_sims_cache = cache["extract_sims_cache"]
		else:
			print("Invalidated extract_sims_cache cache because config.conveyor_speeds has changed")
		
		if cached_allowed_divisors == config.allowed_divisors:
			self.divide_sims_cache = cache["divide_sims_cache"]
		else:
			print("Invalidated divide_sims_cache cache because config.allowed_divisors has changed")
		
		if max(cached_conveyor_speeds) == config.conveyor_speed_limit:
			self.merge_sims_cache = cache["merge_sims_cache"]
		else:
			print("Invalidated merge_sims_cache cache because config.conveyor_speed_limit has changed")

	def save_cache(self):
		print("Saving cache... please do not exit")
		cache = {
			"conveyor_speeds": config.conveyor_speeds,
			"allowed_divisors": config.allowed_divisors,
			"extract_sims_cache": self.extract_sims_cache,
			"divide_sims_cache": self.divide_sims_cache,
			"merge_sims_cache": self.merge_sims_cache
		}
		with open(config.cache_filepath, "wb") as f: pickle.dump(cache, f)

	def get_extract_sim(self, values, i):
		value = values[i]
		simulations = []
		tmp_sim = None
		for speed in config.conveyor_speeds:
			if value <= speed: break
			overflow = value - speed
			if not tmp_sim: tmp_sim = get_sim_without(value, values)
			sim = tuple(sorted(tmp_sim + [speed, overflow]))
			simulations.append((sim, (i, speed)))
		return simulations

	def get_extract_sims(self, values):
		cached_simulations = self.extract_sims_cache.get(values)
		if cached_simulations: return cached_simulations
		simulations = []
		n = len(values)
		seen_values = set()
		for i in range(n):
			value = values[i]
			if value in seen_values: continue
			seen_values.add(value)
			simulations.extend(self.get_extract_sim(values, i))
		self.extract_sims_cache[values] = simulations
		return simulations

	def get_divide_sim(self, values, i):
		value = values[i]
		simulations = []
		tmp_sim = None
		for divisor in config.allowed_divisors:
			if not can_split(value, divisor): continue
			divided_value = value // divisor
			if not tmp_sim: tmp_sim = get_sim_without(value, values)
			sim = tuple(sorted(tmp_sim + [divided_value] * divisor))
			simulations.append((sim, (i, divisor)))
		return simulations

	def get_divide_sims(self, values):
		cached_simulations = self.divide_sims_cache.get(values)
		if cached_simulations: return cached_simulations
		simulations = []
		n = len(values)
		seen_values = set()
		for i in range(n):
			value = values[i]
			if value in seen_values: continue
			seen_values.add(value)
			simulations.extend(self.get_divide_sim(values, i))
		self.divide_sims_cache[values] = simulations
		return simulations

	def get_merge_sims(self, values):
		cached_simulations = self.merge_sims_cache.get(values)
		if cached_simulations: return cached_simulations
		simulations = []
		n = len(values)
		seen_sums = set()
		indices = range(n)
		for to_sum_count in range(config.min_sum_count, config.max_sum_count + 1):
			for to_sum_indices in itertools.combinations(indices, to_sum_count):
				to_sum_indices = list(to_sum_indices)
				to_sum_values = tuple(values[i] for i in to_sum_indices)
				if to_sum_values in seen_sums: continue
				seen_sums.add(to_sum_values)
				summed_value = sum(to_sum_values)
				if summed_value > config.conveyor_speed_limit: continue
				sim = [value for value in values]
				for value in to_sum_values: sim.remove(value)
				insert_into_sorted(sim, summed_value)
				simulations.append((tuple(sim), (to_sum_indices, to_sum_count)))
		self.merge_sims_cache[values] = simulations
		return simulations

	# doesn't have to be generic
	def compute_distance(self, sim, target_values):
		if not self.solver: return -1
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