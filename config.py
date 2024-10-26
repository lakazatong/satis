import re
from re import Pattern
from collections.abc import Callable

class SatisSolverConfig:
	def __init__(self,
			allowed_divisors: list[int],
			conveyor_speeds: list[int],
			logging: bool,
			log_filepath: str,
			# cache_filepath: str,
			short_repr: bool,
			include_level_in_logs: bool,
			solutions_filename: Callable[[int], str],
			solutions_filename_extension: str,
			solution_regex: Pattern
		):
		self.allowed_divisors = sorted(allowed_divisors)
		self.conveyor_speeds = sorted(conveyor_speeds)
		self.logging = logging
		self.log_filepath = log_filepath
		# self.cache_filepath = cache_filepath
		self.short_repr = short_repr
		self.include_level_in_logs = include_level_in_logs
		self.solutions_filename = solutions_filename
		self.solutions_filename_extension = solutions_filename_extension
		self.solution_regex = solution_regex
		
		self.min_sum_count = self.allowed_divisors[0]
		self.max_sum_count = self.allowed_divisors[-1]
		self.conveyor_speed_limit = self.conveyor_speeds[-1]

config = SatisSolverConfig(
	allowed_divisors = [2, 3],
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