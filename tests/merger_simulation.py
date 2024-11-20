import math, os, pickle
from fractions import Fraction
from math import gcd
from functools import reduce

def lcm(*values):
	return reduce(lambda a, b: a * b // gcd(a, b), values)

class Input:
	def __init__(self, speed):
		self.speed = speed
		self.stock = 0
		self.rate = Fraction(speed, 60)
		# self.time_step = Fraction(60, speed)
		self.last_used = -1
		self.history = []

	def set_speed(self, speed):
		self.speed = speed
		self.rate = Fraction(speed, 60)

	def __str__(self):
		return f"Input(speed={self.speed}, stock={self.stock:.2f}, rate={self.rate:.2f}, last_used={self.last_used})"

	def reset(self):
		self.stock = 0
		self.last_used = -1
		self.history = []

class Merger:
	def __init__(self, inputs):
		self.speed = None
		self.inputs = sorted(inputs, key=lambda inp: inp.speed)
		self.current_step = 0
		self.history = []
		self.simulations = 0
		self.min_input_speed = min(inp.speed for inp in self.inputs)

	def set_speed(self, speed):
		self.speed = speed
		self.time_step = Fraction(60, speed) # seconds/item

	def __str__(self):
		return f"Merger(speed={self.speed}, current_step={self.current_step}, simulations={self.simulations}, time_step={self.time_step})"

	def reset(self):
		self.current_step = 0
		self.history = []
		for inp in self.inputs:
			inp.reset()
		self.simulations = 0

	def simulate(self, total_steps):
		if not self.speed:
			raise Exception("cannot simulate without setting the speed first")
		
		for _ in range(total_steps):
			for inp in self.inputs:
				x = inp.stock + inp.rate * self.time_step
				inp.stock = x if x <= 1 else math.floor(x)
			
			self.current_step += 1

			ready_inputs = [inp for inp in self.inputs if inp.stock == 1]
			if not ready_inputs:
				self.history.append(None)
				continue
			
			chosen_input = min(ready_inputs, key=lambda inp: inp.last_used)
			chosen_input.stock -= 1
			chosen_input.last_used = self.current_step
			chosen_input.history.append(self.current_step)
			self.history.append(chosen_input)
			
		self.simulations += 1

	def stabilize_effective_rates(self):
		self.simulate(math.ceil(self.speed/self.min_input_speed))
		# while not self.history:
		# 	self.simulate(1)
		# cur = self.get_current_effective_rates()
		# seen = set(cur)
		# while (cur := self.get_current_effective_rates()) not in seen:
		# 	self.simulate(1)

	def get_current_effective_rates(self):
		return tuple(Fraction(len(inp.history), self.current_step * self.time_step) * 60 for inp in self.inputs)

	def summarize(self):
		return f"{self.speed} " + " ".join(str(rate) for rate in self.get_current_effective_rates())

def generate_simulations():
	result_list = []
	limit = 1200
	
	for x in range(1, limit):
		inputs = [Input(x), Input(limit)]
		merger = Merger(inputs)
		
		for y in range(x + 1, min(x + limit, limit + 1)):
			assert y < x + limit
			merger.reset()
			merger.set_speed(y)
			merger.stabilize_effective_rates()
			effective_rates = merger.get_current_effective_rates()
			result_list.append((x, limit, y, *effective_rates))
		
		for inp in inputs:
			del inp
		del merger
	
	return result_list

def generate_simulation(input_values, output):
	inputs = [Input(i) for i in input_values]
	merger = Merger(inputs)
	merger.set_speed(output)
	merger.stabilize_effective_rates()
	effective_rates = merger.get_current_effective_rates()
	for inp in inputs:
		print(inp.history)
	print(merger)
	for inp in inputs:
		del inp
	del merger
	return (input_values, output, effective_rates)

# n MiB is n * 1024 * 1024
def save_formatted_simulations(filename, format_function, chunk_size, simulations=None):
	simulations = simulations if simulations else generate_simulations()
	if chunk_size < 0:
		with open(filename + ".txt", 'w') as file:
			for sim in simulations:
				file.write(format_function(*sim))
		return
	
	chunk_count = 1
	current_file = filename + f"_chunk_{chunk_count}.txt"
	file = open(current_file, 'w')

	total_size = 0
	
	for sim in simulations:
		file.write(format_function(*sim))
		file.flush()
		
		total_size = os.path.getsize(current_file)
		if total_size >= chunk_size:
			file.close()
			chunk_count += 1
			current_file = filename + f"_chunk_{chunk_count}.txt"
			file = open(current_file, 'w')
			total_size = 0

	file.close()

def save_simulations(filename):
	simulations = generate_simulations()
	with open(filename, 'wb') as file:
		pickle.dump(simulations, file)
	return simulations

def load_simulations(filename):
	with open(filename, 'rb') as file:
		return pickle.load(file)

class Predicate:
	def __init__(self, lambda_func, lambda_code):
		self.lambda_func = lambda_func
		self.lambda_code = lambda_code

	def __call__(self, *args, **kwargs):
		return self.lambda_func(*args, **kwargs)

	def __repr__(self):
		return self.lambda_code

	def __str__(self):
		return self.lambda_code


def run_predicates(simulations):
	predicates = [
		(lambda x, limit, y, r0, r1: x >= y/2, Predicate(lambda x, limit, y, r0, r1: r0 == r1, "r0 == r1"))
	]

	for sim in simulations:
		for test, p in predicates:
			if not test(*sim): continue
			if not p(*sim):
				print(f'"{p}" failed on {sim}')

def learn_relation(simulations, degree=2):
	from sympy import symbols, simplify
	from sympy.utilities.lambdify import lambdify
	from sklearn.linear_model import LinearRegression
	from sklearn.preprocessing import PolynomialFeatures
	import numpy as np
	X = np.array([[x, limit, y] for x, limit, y, _, _ in simulations], dtype=float)
	r0 = np.array([float(r0) for _, _, _, r0, _ in simulations], dtype=float)
	r1 = np.array([float(r1) for _, _, _, _, r1 in simulations], dtype=float)

	poly = PolynomialFeatures(degree=degree, include_bias=False)
	X_poly = poly.fit_transform(X)

	model_r0 = LinearRegression().fit(X_poly, r0)
	r0_coefficients = model_r0.coef_
	r0_intercept = model_r0.intercept_

	model_r1 = LinearRegression().fit(X_poly, r1)
	r1_coefficients = model_r1.coef_
	r1_intercept = model_r1.intercept_

	x, limit, y = symbols('x limit y')
	feature_names = poly.get_feature_names_out(['x', 'limit', 'y'])
	
	# Convert feature names into valid Python expressions for SymPy
	variables = [
		var.replace(" ", "*").replace("^", "**") for var in feature_names
	]

	symbolic_eq_r0 = simplify(
		sum(coef * eval(var, {'x': x, 'limit': limit, 'y': y}) for coef, var in zip(r0_coefficients, variables))
		+ r0_intercept
	)
	symbolic_eq_r1 = simplify(
		sum(coef * eval(var, {'x': x, 'limit': limit, 'y': y}) for coef, var in zip(r1_coefficients, variables))
		+ r1_intercept
	)

	return {
		"r0": {
			"equation": symbolic_eq_r0,
			"function": lambdify((x, limit, y), symbolic_eq_r0, 'numpy')
		},
		"r1": {
			"equation": symbolic_eq_r1,
			"function": lambdify((x, limit, y), symbolic_eq_r1, 'numpy')
		}
	}

def main():
	print(generate_simulation((120, 270, 480), 780))

	return

	# simulations = save_simulations('merger_simulations.pkl')
	simulations = load_simulations('merger_simulations.pkl')
	# save_formatted_simulations("merger_simulations", lambda x, limit, y, r0, r1: f"{x} {limit} to {y} -> {r0} {r1}\n", simulations=simulations)
	# save_formatted_simulations("merger_simulations", lambda x, limit, y, r0, r1: f"{x} {limit} {y} {r0} {r1}\n", 8 * 1024 * 1024, simulations=simulations)

	relations = learn_relation(simulations)

	print("r0 approximation:")
	print(relations['r0']['equation'])

	print("\nr1 approximation:")
	print(relations['r1']['equation'])

	test_x, test_limit, test_y = 100, 1200, 150
	r0_pred = relations['r0']['function'](test_x, test_limit, test_y)
	r1_pred = relations['r1']['function'](test_x, test_limit, test_y)
	print(f"\nTest inputs: x={test_x}, limit={test_limit}, y={test_y}")
	print(f"Predicted r0: {r0_pred}")
	print(f"Predicted r1: {r1_pred}")

if __name__ == '__main__':
	main()