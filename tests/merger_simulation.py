import math, os, pickle
from fractions import Fraction
from math import gcd
from functools import reduce

def lcm(*values):
	return reduce(lambda a, b: a * b // gcd(a, b), values)

def divides(a, b):
	if a == 0: raise ValueError("a == 0")
	q, remainder = divmod(b, a)
	return q if remainder == 0 and q != 1 else None

class Input:
	def __init__(self, speed, symbol=None):
		self.speed = speed
		self.stock = Fraction(0, 1)
		self.rate = Fraction(speed, 60)
		# self.time_step = Fraction(60, speed)
		self.last_used = -1
		self.history = []
		self.symbol = symbol if symbol else chr(speed)

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
		self.simulations = 0
		self.n_items = 0
		self.min_input_speed = min(inp.speed for inp in self.inputs)
		self.n_inputs = len(inputs)
		self.buffers = [0] * self.n_inputs

	def set_speed(self, speed):
		self.speed = speed
		# self.substeps = lcm(speed, *(inp.speed for inp in self.inputs))
		# self.time_step = Fraction(60, self.substeps)
		self.time_step = Fraction(60, speed) # seconds/item

	def __str__(self):
		return f"Merger(speed={self.speed}, current_step={self.current_step}, simulations={self.simulations}, time_step={self.time_step})"

	def reset(self):
		self.current_step = 0
		for inp in self.inputs:
			inp.reset()
		self.simulations = 0

	def simulate(self, total_steps):
		if not self.speed:
			raise Exception("cannot simulate without setting the speed first")
		
		for _ in range(total_steps):
			for i, inp in enumerate(self.inputs):
				inp.stock += inp.rate * self.time_step
				if inp.stock < 1: continue
				if self.buffers[i] < 3:
					self.buffers[i] += 1
				inp.stock -= 1

			move_to_last = []
			for i in range(self.n_inputs):
				if self.buffers[i] == 0: continue
				self.current_step += 1
				self.inputs[i].history.append(self.current_step)
				self.buffers[i] -= 1
				move_to_last.append(i)
				break

			offset = 0
			for i in move_to_last:
				self.inputs.append(self.inputs.pop(i - offset))
				self.buffers[i], self.buffers[-1] = self.buffers[-1], self.buffers[i]
				offset += 1

		self.simulations += 1

	def stabilize_effective_rates(self):
		# while self.n_items == 0:
		# 	self.simulate(1)
		# rates = self.get_current_effective_rates()
		# while any(rate == 0 for rate in rates) or sum(rates) != self.speed:
		# 	self.simulate(1)
		# 	rates = self.get_current_effective_rates()
		while self.current_step < self.speed:
			self.simulate(1)
		# self.simulate(1)

	def get_current_effective_rates(self):
		return tuple(len(inp.history) for inp in sorted(self.inputs, key=lambda inp: inp.speed))

	def summarize(self):
		return f"{self.speed} " + " ".join(str(rate) for rate in self.get_current_effective_rates())

	def generate_sequence(self):
		r = ''
		indices = [0] * len(self.inputs)
		for i in range(1, self.current_step + 1):
			for j, inp in enumerate(self.inputs):
				k = indices[j]
				if k < len(inp.history) and inp.history[k] == i:
					r += inp.symbol
					indices[j] = k + 1
					break
		return r

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
	return merger, (input_values, output, effective_rates)

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
	# 60, 120, 270, 480, 780, 1200
	# print(reduce(lcm, [60, 120, 270, 480, 780, 1200]))
	# mks = [Fraction(60, 60), Fraction(60, 120), Fraction(60, 270), Fraction(60, 480), Fraction(60, 780), Fraction(60, 1200)]
	# mk6 = Fraction(60, 280800)
	# for mk in mks:
	# 	print(mk, divides(mk6, mk))
	# print(generate_simulation((60, 1200), 270))
	merger, r = generate_simulation((120, 270, 480), 780)
	for inp in merger.inputs:
		print(inp.history)
	print(r, sum(r[2]))
	merger.inputs[0].symbol = 'g'
	merger.inputs[1].symbol = 'r'
	merger.inputs[2].symbol = 'b'
	print(merger.generate_sequence())

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