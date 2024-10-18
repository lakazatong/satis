import time
from collections import Counter
import random

# Your original method
def remove_pairs_original(sim, targets):
	for value in sim[:]:  # Iterate over a copy of sim
		if value in targets[:]:  # Iterate over a copy of targets
			sim.remove(value)
			targets.remove(value)
	return sim, targets

# My optimized method
def remove_pairs(list_a, list_b):
	count_a = Counter(list_a)
	count_b = Counter(list_b)
	
	for item in count_a.keys():
		if item in count_b:
			pairs_to_remove = min(count_a[item], count_b[item])
			count_a[item] -= pairs_to_remove
			count_b[item] -= pairs_to_remove
			
	remaining_a = []
	remaining_b = []
	
	for item, count in count_a.items():
		remaining_a.extend([item] * count)
		
	for item, count in count_b.items():
		remaining_b.extend([item] * count)
		
	return remaining_a, remaining_b

# Benchmarking function
def benchmark(method1, method2, test_cases):
	total_time1 = 0
	total_time2 = 0
	consistent = True
	
	for sim, targets in test_cases:
		# Time method 1
		start_time = time.time()
		result1 = method1(sim.copy(), targets.copy())
		duration1 = time.time() - start_time
		total_time1 += duration1

		# Time method 2
		start_time = time.time()
		result2 = method2(sim.copy(), targets.copy())
		duration2 = time.time() - start_time
		total_time2 += duration2
		
		# Check if results are the same
		a1, b1 = result1
		a2, b2 = result2
		a1 = sorted(a1)
		a2 = sorted(a2)
		b1 = sorted(b1)
		b2 = sorted(b2)
		if (a1, b1) != (a2, b2):
			consistent = False
			print(sim)
			print(targets)
			print(result1)
			print(result2)
			break

	avg_time1 = total_time1 / len(test_cases)
	avg_time2 = total_time2 / len(test_cases)
	
	return avg_time1, avg_time2, consistent

# Create test cases
def create_test_cases(num_cases, size):
	test_cases = []
	for _ in range(num_cases):
		# Create two lists with random integers
		sim = [random.randint(1, 10) for _ in range(size)]
		targets = [random.randint(1, 10) for _ in range(size)]
		test_cases.append((sim, targets))
	return test_cases

# Number of test cases and size of each list
num_cases = 1000  # Number of test cases
size = 1000     # Size of each list

# Generate test cases
test_cases = create_test_cases(num_cases, size)

# Run the benchmark
avg_time1, avg_time2, results_match = benchmark(remove_pairs_original, remove_pairs_optimized, test_cases)

# Print results
if results_match:
	print("Both methods returned the same results.")
else:
	print("Methods returned different results.")
	
print(f"Average Duration - Original Method: {avg_time1:.6f}s, Optimized Method: {avg_time2:.6f}s")
