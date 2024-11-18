import re
from re import Pattern
from collections.abc import Callable

class SatisSolverConfig:
	def __init__(self,
			conveyor_speeds: list[int],
			logging: bool,
			log_filepath: str,
			short_repr: bool,
			include_level_in_logs: bool,
			solutions_filename: Callable[[int], str],
			solutions_filename_extension: str,
			solution_regex: Pattern
		):
		self.conveyor_speeds = sorted(conveyor_speeds)
		self.logging = logging
		self.log_filepath = log_filepath
		self.short_repr = short_repr
		self.include_level_in_logs = include_level_in_logs
		self.solutions_filename = solutions_filename
		self.solutions_filename_extension = solutions_filename_extension
		self.solution_regex = solution_regex
		
		self.conveyor_speed_limit = self.conveyor_speeds[-1]
		self.allowed_divisors = [d for d in range(2, self.conveyor_speed_limit + 1)]
		# self.allowed_extractors = [Fraction(c) for c in range(1, self.conveyor_speed_limit + 1)]
		self.allowed_extractors = [c for c in range(1, self.conveyor_speed_limit + 1)]
		self.conveyor_speeds_r = reversed(self.conveyor_speeds)
		self.allowed_divisors_r = reversed(self.allowed_divisors)
		self.allowed_extractors_r = reversed(self.allowed_extractors)
		self.min_sum_count = 2
		self.max_sum_count = 3

config = SatisSolverConfig(
	conveyor_speeds = [60, 120, 270, 480, 780, 1200],
	logging = False,
	log_filepath = "logs.txt",
	# cache_filepath = "sims_caches",
	short_repr = True,
	include_level_in_logs = False,
	solutions_filename = lambda i: f"solution{i}",
	solutions_filename_extension = "png",
	solution_regex = re.compile(r'solution\d+\.png')
)