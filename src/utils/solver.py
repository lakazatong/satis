def gcd_incompatible(gcd, value):
	return value < gcd

def get_gcd_incompatible(gcd):
	from functools import partial
	return partial(gcd_incompatible, gcd)

def debug_parsed_values(source_values, target_values):
	if source_values is not None and target_values is not None:
		print("Source values:")
		for val in source_values:
			print(f"{val} (as fraction: {val.numerator}/{val.denominator})")
		print("\nTarget values:")
		for val in target_values:
			print(f"{val} (as fraction: {val.numerator}/{val.denominator})")

def parse_user_input(user_input):
	from .fractions import parse_fraction
	separator = 'to'
	if len(user_input.split(" ")) < 3 or separator not in user_input:
		print(f"Usage: <source_args> {separator} <target_args>")
		return [], []

	source_part, target_part = user_input.split(separator)
	source_args = source_part.strip().split()
	target_args = target_part.strip().split()

	if not source_args:
		print("Error: At least one source value must be provided.")
		return None, None
	if not target_args:
		print("Error: At least one target value must be provided.")
		return None, None

	source_values = []
	i = 0
	while i < len(source_args):
		src = source_args[i]
		if not src.endswith('x'):
			source_value = parse_fraction(src)
			source_values.append(source_value)
			i += 1
			continue
		if len(src) < 2 or not src[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return None, None
		multiplier = int(src[:-1])
		source_value = parse_fraction(source_args[i + 1])
		source_values.extend([source_value] * multiplier)
		i += 2

	target_values = []
	i = 0
	while i < len(target_args):
		target = target_args[i]
		if not target.endswith('x'):
			target_value = parse_fraction(target)
			target_values.append(target_value)
			i += 1
			continue
		if len(target) < 2 or not target[:-1].isdigit():
			print("Error: Invalid Nx format. N must be a number followed by 'x'.")
			return None, None
		multiplier = int(target[:-1])
		if i + 1 == len(target_args):
			print("Error: You must provide a target value after Nx.")
			return None, None
		target_value = parse_fraction(target_args[i + 1])
		target_values.extend([target_value] * multiplier)
		i += 2

	# debug_parsed_values(source_values, target_values)
	return source_values, target_values

def compute_cant_use(target_counts, sources):
	source_counts = {}
	for src in sources:
		if src.value in source_counts:
			source_counts[src.value] += 1
		else:
			source_counts[src.value] = 1
	cant_use = set()
	for src in sources:
		value = src.value
		src_count = source_counts.get(value, None)
		target_count = target_counts.get(value, None)
		if src_count and target_count and max(0, src_count - target_count) == 0:
			cant_use.add(value)
	return cant_use

def get_compute_cant_use(target_counts):
	from functools import partial
	return partial(compute_cant_use, target_counts)

def get_sim_without(value, values):
	sim = [v for v in values]
	sim.remove(value)
	return sim