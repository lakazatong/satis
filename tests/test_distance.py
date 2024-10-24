import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cProfile
from utils import generate_test_cases
from distance import distance

def test_distance():
	n_cases = 1000
	test_cases = generate_test_cases(n_cases, 100)
	for i, (sources, targets) in enumerate(test_cases):
		d = distance(sources[:], targets[:])
		# d2 = distance2(sources[:], targets[:])
		print("\r" + " " * 100 + f"\r{i/n_cases*100:.0f}%", end="")
		# to_sum_count = 2
		# d = distance_merge(sources[:], targets[:], to_sum_count)
		# d2 = distance_merge2(sources[:], targets[:], to_sum_count)
		# if d != d2:
		# 	print("\nnope", d, d2, sources, targets)
		# 	exit(0)

cProfile.run("test_distance()")
# test_distance()