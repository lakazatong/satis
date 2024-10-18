import re
from re import Pattern
from collections.abc import Callable

class SatisSolverConfig:
	def __init__(self,
			allowed_divisors: list[int],
			conveyor_speeds: list[int],
			logging: bool,
			log_filepath: str,
			cache_filepath: str,
			short_repr: bool,
			include_depth_informations: bool,
			solutions_filename: Callable[[int], str],
			solutions_filename_extension: str,
			solution_regex: Pattern
		):
		self.allowed_divisors = set(sorted(allowed_divisors))
		self.conveyor_speeds = set(sorted(conveyor_speeds))
		self.logging = logging
		self.log_filepath = log_filepath
		self.cache_filepath = cache_filepath
		self.short_repr = short_repr
		self.include_depth_informations = include_depth_informations
		self.solutions_filename = solutions_filename
		self.solutions_filename_extension = solutions_filename_extension
		self.solution_regex = solution_regex
		
		self.min_sum_count = min(self.allowed_divisors)
		self.max_sum_count = max(self.allowed_divisors)
		self.conveyor_speed_limit = max(self.conveyor_speeds)

config = SatisSolverConfig(
	allowed_divisors = [2, 3],
	conveyor_speeds = [60, 120, 270, 480, 780, 1200],
	logging = True,
	log_filepath = "logs.txt",
	cache_filepath = "sims_caches.json",
	short_repr = False,
	include_depth_informations = False,
	solutions_filename = lambda i: f"solution{i}",
	solutions_filename_extension = "png",
	solution_regex = re.compile(r'solution\d+\.png')
)