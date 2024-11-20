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
		self.inputs = inputs
		self.current_step = 0
		self.history = []
		self.simulations = 0
		self.min_input_speed = min(inp.speed for inp in self.inputs)

	def set_speed(self, speed):
		self.speed = speed
		self.time_step = Fraction(60, speed) # seconds/item

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
				inp.stock = min(1, inp.stock + inp.rate * self.time_step)
			
			ready_inputs = [inp for inp in self.inputs if inp.stock == 1]
			if not ready_inputs:
				self.history.append(None)
				continue
			
			chosen_input = min(ready_inputs, key=lambda inp: inp.last_used)
			chosen_input.stock -= 1
			chosen_input.last_used = self.current_step
			chosen_input.history.append(self.current_step)
			self.history.append(chosen_input)
			
			self.current_step += 1

		self.simulations += 1

	def stabilize_effective_rates(self):
		self.simulate(math.ceil(self.speed/self.min_input_speed))

	def get_current_effective_rates(self):
		return [Fraction(len(inp.history), self.current_step * self.time_step) * 60 for inp in self.inputs]

	def summarize(self):
		return f"{self.speed} " + " ".join(str(rate) for rate in self.get_current_effective_rates())

def generate_simulations():
	result_list = []
	other_input_speed = 1200
	
	for x in range(1, other_input_speed):
		inputs = [Input(x), Input(other_input_speed)]
		merger = Merger(inputs)
		
		for y in range(x + 1, min(x + other_input_speed, other_input_speed + 1)):
			assert y < x + other_input_speed
			merger.reset()
			merger.set_speed(y)
			merger.stabilize_effective_rates()
			effective_rates = merger.get_current_effective_rates()
			result_list.append((x, other_input_speed, y, effective_rates[0], effective_rates[1]))
		
		del merger
		for inp in inputs:
			del inp
	
	return result_list

def save_formatted_simulations(filename, simulations=None, chunk_size=5 * 1024 * 1024):
	simulations = simulations if simulations else generate_simulations()
	
	chunk_count = 1
	current_file = filename + f"_chunk_{chunk_count}.txt"
	file = open(current_file, 'w')

	total_size = 0
	
	for sim in simulations:
		x, other_input_speed, y, rate0, rate1 = sim
		file.write(f"{x} {other_input_speed} to {y} -> {effective_rates[0]} {effective_rates[1]}\n")
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

def load_simulations(filename):
	with open(filename, 'rb') as file:
		return pickle.load(file)

# save_simulations('merger_simulations.pkl')

simulations = load_simulations('merger_simulations.pkl')

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

predicates = [
	(lambda x, limit, y, r0, r1: x >= y/2, Predicate(lambda x, limit, y, r0, r1: r0 == r1, "x >= y/2"))
]

for sim in simulations:
	for test, p in predicates:
		if not test(*sim): continue
		if not p(*sim):
			print(f'"{p}" failed on {sim}')

# save_formatted_simulations("merger_simulations", simulations=simulations)