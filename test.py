import sys, os, pathlib
dirpath = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(dirpath)
sys.path.append(os.path.join(dirpath, 'src'))
if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

import cProfile, random, time, math
from bisect import insort
from fastList import FastList
from config import config

def compute(x, n, m, l):
	d = 2**n*3**m
	b = x / d
	t = b + (l * b) / (d - l)
	return t

# print(compute(780, 1, 1, 1))
# print(compute(28, 0, 2, 2))

def count_splits(n, m):
	result = 0
	nodes = 1
	for _ in range(n):
		nodes *= 2
		result += nodes // 2
	for _ in range(m):
		nodes *= 3
		result += nodes // 3
	return result

def find_n_m_l(X):
	max_n = 0
	max_m = 0
	while 2 ** max_n <= X: max_n += 1
	while 3 ** max_m <= X: max_m += 1
	min_splits = float('inf')
	best_n = best_m = 0
	best_l = 0
	splits = None
	for n in range(max_n + 1):
		for m in range(max_m + 1):
			product = 2 ** n * 3 ** m
			if product > X:
				l = product - X
				splits = count_splits(n, m)
				if splits < min_splits:
					min_splits = splits
					best_n = n
					best_m = m
					best_l = l
	return best_n, best_m, best_l, min_splits

def count_merges(n):
	if n < 0: raise ValueError("cannot merge negative amount of nodes")
	if n <= 1: return n
	r = 0
	while n > 1:
		r += 1
		n -= 2
	return r

def test(x, t):
	t_count = int(x/t)
	n, m, l, splits = find_n_m_l(t_count)
	print("\n" + "\n".join([
		f"splitting {x} {n} times in 2, then {m} times in 3",
		f"results in {t_count}x {t}",
		f"loops back {l} branches to {x}",
		f"uses {splits} splitters + {count_merges(l)} mergers"
	]))

test(780, 156)
test(28, 4)
test(1200, 1)
test(5, 0.5)

# from tests.test_distance import test_distance

# cProfile.run("test_distance()")
# test_distance()