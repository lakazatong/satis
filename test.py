queue = list(i for i in range(1, 101))
import random, math

count = 0
L = [1, 2, 3]

def getlen(arr):
	global count
	count += 1
	return len(arr)

for i in range(getlen(L)):
	print(i)

print(count)

exit(0)

def dequeue(queue):
	n = len(queue)
	i = 1
	while True:
		tmp = 1 << (i - 1)
		prob = 1 / (tmp << 1)
		idx = round((1 - 1 / tmp) * n)
		print(round(prob * 100, 1), idx)
		i += 1
		if i > n or idx >= n: return queue.pop(-1)
		# if random.random() < prob: return queue.pop(idx)

print(dequeue(list(i for i in range(1, 101))))