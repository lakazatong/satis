import re
from re import Pattern
from collections.abc import Callable

class SatisSolverConfig:
	def __init__(self,
			allowed_divisors: list[int],
			conveyor_speeds: list[int],
			logging: bool,
			log_filename: str,
			short_repr: bool,
			include_depth_informations: bool,
			solutions_filename: Callable[[int], str],
			solution_regex: Pattern
		):
		self.allowed_divisors = allowed_divisors
		self.conveyor_speeds = conveyor_speeds
		self.logging = logging
		self.log_filename = log_filename
		self.short_repr = short_repr
		self.include_depth_informations = include_depth_informations
		self.solutions_filename = solutions_filename
		self.solution_regex = solution_regex
		
		self.allowed_divisors_r = self.allowed_divisors[::-1]
		self.min_sum_count = self.allowed_divisors[0]
		self.max_sum_count = self.allowed_divisors_r[0]

		self.conveyor_speeds_r = self.conveyor_speeds[::-1]
		self.conveyor_speed_limit = self.conveyor_speeds_r[0]

config = SatisSolverConfig(
	allowed_divisors = [2, 3], # must be sorted
	conveyor_speeds = [60, 120, 270, 480, 780, 1200], # must be sorted
	logging = False,
	log_filename = "logs.txt",
	short_repr = False,
	include_depth_informations = False,
	solutions_filename = lambda i: f"solution{i}",
	solution_regex = re.compile(r'solution\d+\.png') # extension is always png
)