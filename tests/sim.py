from cost import extract_cost, merge_cost
from utils import remove_pairs
from itertools import combinations

def group_values(L):
	grouped = False
	i = 0
	while i < len(L) - 1:
		ref = L[i]
		while i < len(L) - 1 and L[i + 1] == ref:
			L[i] += L.pop(i + 1)
			grouped = True
		i += 1
	return grouped

def apply_best_merge(sources, targets):
	min_cost = float('inf')
	best_merge = None

	for r in range(2, len(sources) + 1):
		for comb in combinations(sources, r):
			merge_result = sum(comb)
			if merge_result in targets:
				cost = merge_cost(len(comb), 1) # merge len(comb) values into 1
				if cost < min_cost:
					min_cost = cost
					best_merge = comb

	if best_merge:
		for val in best_merge:
			sources.remove(val)
		return True
	return False

def apply_best_extract_two_targets(sources, targets):
    min_cost = float('inf')
    best_extraction = None
    best_source = None

    for source in sources:
        for i, t1 in enumerate(targets):
            for t2 in targets[i + 1:]:
                if source != t1 + t2:
                    continue
                cost = extract_cost(source, t1)  # Extract t1 from source
                if cost < min_cost:
                    min_cost = cost
                    best_extraction = (t1, t2)
                    best_source = source

    if best_extraction:
        t1, t2 = best_extraction
        sources.remove(best_source)
        targets.remove(t1)
        targets.remove(t2)
        return True
    return False

def apply_best_extract_one_target(sources, targets):
    min_cost = float('inf')
    best_extraction = None
    best_source = None

    for source in sources:
        for t in targets:
            if source < t:
                continue
            cost = extract_cost(source, t)  # Extract t from source
            if cost < min_cost:
                min_cost = cost
                best_extraction = t
                best_source = source

    t = best_extraction
    sources.remove(best_source)
    sources.append(best_source - t)
    targets.remove(t)
    return True

def solve(sources, targets):
	sources = sorted(sources)
	targets = sorted(targets)

	sources_sum = sum(sources)
	targets_sum = sum(targets)
	if sources_sum < targets_sum:
		sources.append(targets_sum - sources_sum)
	if sources_sum > targets_sum:
		targets.append(sources_sum - targets_sum)

	while True:
	
		sources, targets = remove_pairs(sources, targets)

		if (len(sources) == 0 and len(targets) != 0) or (len(sources) != 0 and len(targets) == 0):
			print('oopsie')
			exit(1)

		if len(targets) == 1:
			return

		while group_values(targets):
			continue

		if not apply_best_merge(sources, targets):
			if not apply_best_extract_two_targets(sources, targets):
				apply_best_extract_one_target(sources, targets)

		if sources == targets:
			break