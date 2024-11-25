def fractions_to_integers(fractions):
	import math
	denominators = [frac.denominator for frac in fractions]
	common_denominator = math.lcm(*denominators)
	integers = [frac.numerator * (common_denominator // frac.denominator) for frac in fractions]
	return integers, common_denominator

def decimal_representation_info(fraction):
	denominator = fraction.denominator
	power_of_2 = 0
	power_of_5 = 0
	while denominator % 2 == 0:
		denominator //= 2
		power_of_2 += 1
	while denominator % 5 == 0:
		denominator //= 5
		power_of_5 += 1
	m = max(power_of_2, power_of_5)
	if denominator == 1: return True, m # terminating and has m digits after the decimal point
	if power_of_2 == 0 and power_of_5 == 0: return False, None # non-terminating and non-repeating
	return False, m # non-terminating and repeating at some point after m digits

def format_fractions(fractions):
	output = []
	counts = {}

	for frac in fractions:
		if frac in counts:
			counts[frac] += 1
		else:
			counts[frac] = 1

	def with_count(count, frac_str):
		return f"{count}x {frac_str}" if count > 1 else frac_str

	for frac, count in counts.items():
		terminating, m = decimal_representation_info(frac)

		if not m:
			output.append(with_count(count, str(frac).replace('/', '⧸')))
			continue

		if terminating:
			# Construct the terminating decimal string
			integer_part = frac.numerator // frac.denominator
			decimal_part = abs(frac.numerator) % frac.denominator

			decimal_str = ''
			for _ in range(m):
				decimal_part *= 10
				decimal_digit = decimal_part // frac.denominator
				decimal_str += str(decimal_digit)
				decimal_part %= frac.denominator

			# Combine integer and decimal parts
			output.append(with_count(count, f"{integer_part}.{decimal_str}"))
			continue

		output.append(with_count(count, str(frac).replace('/', '⧸')))

		# graveyard

		# decimal_part = []
		# integer_part = frac.numerator // frac.denominator
		# remainder = frac.numerator % frac.denominator
		
		# seen_remainders = {}
		# index = 0

		# while remainder != 0:
		# 	if remainder in seen_remainders:
		# 		repeat_start = seen_remainders[remainder]
		# 		non_repeating = ''.join(decimal_part[:repeat_start])
		# 		repeating = ''.join(decimal_part[repeat_start:])
		# 		output.append(with_count(count, f"{integer_part}.{non_repeating}({repeating})"))
		# 		break

		# 	seen_remainders[remainder] = index
		# 	remainder *= 10
		# 	decimal_digit = remainder // frac.denominator
		# 	decimal_part.append(str(decimal_digit))
		# 	remainder %= frac.denominator
		# 	index += 1
		# else:
		# 	output.append(with_count(count, str(frac)))

	return ' '.join(output)

def parse_fraction(fraction_str):
	from fractions import Fraction
	slash_symbols = ['/', '⧸']
	for slash_symbol in slash_symbols:
		if slash_symbol in fraction_str:
			numerator, denominator = fraction_str.split(slash_symbol)
			return Fraction(int(numerator), int(denominator))
	
	import re
	match = re.match(r'(\d*)\.(\d*)\((\d+)\)', fraction_str)
	if match:
		# supports notation like 2.(3) for 2.3333333333333...
		# the () notation is widely accepted and is non ambiguous, for example
		# 4.787878(457) means 4.787878457457457457457457... 457 repeating even tho 787878 was before
		whole_part = match.group(1)
		non_repeating = match.group(2)
		repeating = match.group(3)
		
		if not whole_part:
			whole_part = '0'
		
		non_repeating_len = len(non_repeating)
		repeating_len = len(repeating)

		numerator = int(whole_part + non_repeating + repeating) - int(whole_part + non_repeating)
		denominator = (10 ** (non_repeating_len + repeating_len) - 10 ** non_repeating_len)

		return Fraction(numerator, denominator)

	# whereas, 8.123(456)7 wouldnt match, so it would fall down to float(8.1234567) basically
	if '.' in fraction_str:
		return Fraction(fraction_str.replace('(', '').replace(')', '')).limit_denominator()
	
	# if there is no '.' it requires the string to match \d+ basically and will be interpreted as an integer
	return Fraction(int(fraction_str), 1)