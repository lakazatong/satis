import sys, os, pathlib
dirpath = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(dirpath)
sys.path.append(os.path.join(dirpath, 'src'))
if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

import itertools, time
from bisect import insort
from fastList import FastList
from config import config
from utils import get_divisors, decimal_representation_info, divide_cost, extract_cost, merge_cost, find_n_m_l, all_sums
from fractions import Fraction
from score import ScoreCalculator
import matplotlib.pyplot as plt

print(merge_cost(4, 1))
print(divide_cost(28, 7))

exit(0)

# print(decimal_representation_info(Fraction(14_000_107, 7_812_500))) # terminate, 9 digits
# print(decimal_representation_info(Fraction(144_573, 96_040))) # non terminating repeating after m digits
# print(decimal_representation_info(Fraction(144_573, 2_401))) # non terminating and never repeating

x = 60
values = [extract_cost(x, i) for i in range(1, x+1)]
plt.plot(range(1, x+1), values, 'o')
plt.xlabel('i')
plt.ylabel('Extract Cost')
plt.title('Extract Cost vs. i (Dot Plot)')
plt.grid()
plt.show()

exit(0)

# targets = [4] * 7

# scoreCalculator = ScoreCalculator(targets)
# print_score = lambda sources: print(f"{sources} -> {scoreCalculator.compute(sources)}")
# # print_score([7] * 4)
# # print_score([7, 1, 6, 7, 7])
# # print_score([7, 1, 6, 14])
# print_score([7, 1, 20])
# # print_score([28])

# exit(0)

# x = 12
# for d in range(1, x+1):
# 	print(d)
# 	print(find_n_m_l(d))
# 	print(divide_cost(x, d))
# 	print()

#      0  1  2  3  4  5  6  7  8  9 10 11 12
# res = [7, 8, 8, 7, 7, 8, 5, 5, 6, 4, 5, 5, 0]
# for i in range(x+1):
# 	print()
# 	cost = divide_cost(x, x, force_l = i)
# 	print(f"{cost = }")
# 	if cost != res[i]:
# 		print(f"failed for {i}")
# 		break

#      1  2  3  4  5  6  7  8  9 10 11 12
# res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# for i in range(1, x+1):
# 	print()
# 	cost = extract_cost(x, i)
# 	print(f"{i = }")
# 	print(f"{cost = }")
# 	# if cost != res[i]:
# 	# 	print(f"failed for {i}")
# 	# 	break

# exit(0)

# # for t in targets:
# # 	print(f"minimum divisor for {t}: {min(get_divisors(t))}")

# h = lambda value: f"{value}: {maximum_value(value, targets)}"
# print(h(5))
# print(h(2))
# print(h(11))
# print(h(20))
# print(h(50))
# print(h(100))
# print(h(25))
# print(h(55))

# exit(0)

# print(divides(Fraction(11, 4), 11))
# print(divides(25, 457))
# print(divides(Fraction(1, 2), Fraction(11, 2)))

# exit(0)

def compute(x, n, m, l):
	d = 2**n*3**m
	b = x / d
	t = b + (l * b) / (d - l)
	return t

# print(compute(780, 1, 1, 1))
# print(compute(28, 0, 2, 2))


def count_merges(n, m, l):
	if l < 0: raise ValueError("cannot merge negative amount of nodes")
	if l <= 1: return l
	r = 0
	while l > 1:
		r += 1
		l -= 2
	return r

def test(x, t):
	if not divides(t, x):
		print("no?")
		return
	t_count = int(x/t)
	n, m, l, splits = find_n_m_l(t_count)
	print("\n" + "\n".join([
		f"splitting {x} {n} times in 2, then {m} times in 3",
		f"results in {t_count}x {t}",
		f"loops back {l} branches to {x}",
		f"uses {splits} splitters + {count_merges(n, m, l)} mergers"
	]))

test(7, 1)
# test(780, 156)
# test(28, 4)
# test(1200, 1)
# test(5, 0.5)
# test(5.5, 0.5)
# test(60, 1)
# test(2285, 125)
# test(457, 25)
# test(115, 5)
# test(11, Fraction(11, 4))
# test(45, 5)
# print(count_merges(60))
