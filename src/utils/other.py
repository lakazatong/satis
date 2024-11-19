def remove_pairs(list_a, list_b):
	from collections import Counter
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