import re

def parse_profile_data(data):
	lines = data.strip().splitlines()
	total_time = float(re.search(r"in (\d+\.\d+) seconds", lines[0]).group(1))
	func_times = []
	
	for line in lines[5:]:
		match = re.search(r"(\d+) +[\d.]+ +[\d.]+ +([\d.]+) +[\d.]+ +(.+):\d+\((.+)\)", line)
		if match:
			cumtime = float(match.group(2))
			func_times.append((match.group(4), cumtime, match.group(3)))
	
	func_times.sort(key=lambda x: x[1], reverse=True)
	return [(round(cumtime / total_time * 100, 2), name, filename) for name, cumtime, filename in func_times]

with open("profile.out", "r", encoding="utf-8") as f:
	for percentage, name, filename in parse_profile_data(f.read()):
		print(f"{int(percentage)}%: {name} ({filename})")
