import itertools

from utils import remove_pairs, insert_into_sorted
from config import config

def distance_division(sources, targets, divisor, threshold):
	original_n_targets = len(targets)
	for i in range(len(sources) - 1, -1, -1):
		s = sources[i]
		if s % divisor != 0: continue
		divided_value = s // divisor
		n_matches = targets.count(divided_value)
		if n_matches == threshold:
			sources.pop(i)
			for _ in range(n_matches): targets.remove(divided_value)
			for _ in range(divisor - n_matches): insert_into_sorted(sources, divided_value)
	return len(targets) != original_n_targets

def distance_loop_division(sources, targets, threshold):
	original_n_targets = len(targets)
	for i in range(len(sources) - 1, -1, -1):
		s = sources[i]
		c = next((c for c in config.conveyor_speeds if c > s), None)
		if not c: continue
		c_over_3 = c // 3
		c_over_3_times_2 = c_over_3 << 1
		if s <= c_over_3_times_2: continue
		overflow = s - c_over_3_times_2
		n_c_over_3 = targets.count(c_over_3)
		n_overflow = 1 if overflow in targets else 0
		n_matches = n_c_over_3 + n_overflow
		if n_matches == threshold:
			sources.pop(i)
			for _ in range(n_c_over_3): targets.remove(c_over_3)
			for _ in range(2 - n_c_over_3): insert_into_sorted(sources, c_over_3)
			if n_overflow == 1:
				targets.remove(overflow)
			else:
				insert_into_sorted(sources, overflow)
	return len(targets) != original_n_targets

def distance_extraction(sources, targets, threshold):
	original_n_targets = len(targets)
	for i in range(len(sources) - 1, -1, -1):
		s = sources[i]
		for c in config.conveyor_speeds:
			if s <= c: continue
			overflow = s - c
			if c == overflow: continue # would be the same as splitting in two
			c_in_targets = c in targets
			overflow_in_targets = overflow in targets
			n_matches = 0
			if c_in_targets: n_matches += 1
			if overflow_in_targets: n_matches += 1
			if n_matches != threshold: continue
			sources.pop(i)
			if c_in_targets:
				targets.remove(c)
			else:
				insert_into_sorted(sources, c)
			if overflow_in_targets:
				targets.remove(overflow)
			else:
				insert_into_sorted(sources, overflow)
			break
	return len(targets) != original_n_targets

def find_best_merges(all_merges, targets):
	best_sources_left, best_targets_left = None, None
	best_n_targets_left = None
	n_targets = len(targets)

	def _find_best_merges(cur_target_idx, sources_left, targets_left):
		nonlocal best_sources_left, best_targets_left, best_n_targets_left
		if cur_target_idx >= n_targets:
			n_targets_left = len(targets_left)
			if not best_n_targets_left or n_targets_left < best_n_targets_left:
				best_n_targets_left = n_targets_left
				best_sources_left = sources_left[:]
				best_targets_left = targets_left[:]
			return
		
		target = targets[cur_target_idx]
		merges = all_merges[cur_target_idx]
		
		if not merges or target not in targets_left:
			_find_best_merges(cur_target_idx + 1, sources_left, targets_left)
			return
		
		for i, merge in enumerate(merges):
			if any(sources_left.count(v) < merge.count(v) for v in merge):
				_find_best_merges(cur_target_idx + 1, sources_left, targets_left)
				continue
			
			new_sources_left = sources_left[:]
			for v in merge: new_sources_left.remove(v)
			new_targets_left = targets_left[:]
			new_targets_left.remove(target)
			_find_best_merges(cur_target_idx + 1, new_sources_left, new_targets_left)

	return best_sources_left, best_targets_left, best_n_targets_left

def distance_merge(sources, targets, to_sum_count):
	all_combinations = list(itertools.combinations(sources, to_sum_count))
	combinations_count = len(all_combinations)
	all_combinations_sum = [sum(combination) for combination in all_combinations]
	all_merges = [[
		all_combinations[i]
		for i in range(combinations_count)
		if all_combinations_sum[i] == target
	] for target in targets]
	sources_left, targets_left, n_targets_left = find_best_merges(all_merges, targets)
	if sources_left: # and targets_left and n_targets_left
		sources = sources_left
		targets = targets_left
		return True
	return False

def distance(sources, targets):
	while True:
		# removes ?, add 0
		sources, targets = remove_pairs(sources, targets)
		if not sources or not targets: break
		
		go_on = False
		
		# removes 4, adds 0
		go_on |= distance_division(sources, targets, 3, 3)
		go_on |= distance_loop_division(sources, targets, 3)
		go_on |= distance_merge(sources, targets, 3)
		
		# removes 3, adds 0
		go_on |= distance_division(sources, targets, 2, 2)
		go_on |= distance_extraction(sources, targets, 2)
		go_on |= distance_merge(sources, targets, 2)
		
		# removes 3, adds 1
		go_on |= distance_division(sources, targets, 3, 2)
		go_on |= distance_loop_division(sources, targets, 2)
		
		# removes 2, adds 1
		go_on |= distance_division(sources, targets, 2, 1)
		go_on |= distance_extraction(sources, targets, 1)
		
		# removes 2, adds 2
		go_on |= distance_loop_division(sources, targets, 1)
		go_on |= distance_division(sources, targets, 3, 1)
		
		if not go_on: break
	
	return len(targets)