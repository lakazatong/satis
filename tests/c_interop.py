# class combinations_result(ctypes.Structure):
# 	_fields_ = [
# 		("all_combinations", ctypes.POINTER(ctypes.c_int)),
# 		("all_combinations_sum", ctypes.POINTER(ctypes.c_int)),
# 		("combinations_count", ctypes.c_int)
# 	]

# os.add_dll_directory(os.getcwd())
# clib = ctypes.CDLL("combinations.dll", winmode=0)
# clib.old_combinations.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
# clib.old_combinations.restype = ctypes.c_int
# clib.combinations.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
# clib.combinations.restype = ctypes.c_int

# clib.test_depth.argtypes = (ctypes.c_int,)
# clib.test_depth.restype = ctypes.c_int
# depth = 1 << 16 - 1
# print(depth)
# result = clib.test_depth(depth)
# print(result)
# exit(0)

# class test_result(ctypes.Structure):
# 	_fields_ = [
# 		("arr", ctypes.POINTER(ctypes.c_int))
# 	]

# clib.test.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
# clib.test.restype = test_result

# n = 100
# arr = [0] * n
# array = (ctypes.c_int * n)(*arr)
# clib.test(array, n)
# print(list(array))
# exit(0)

# def clib_combinations(sources, to_sum_count):
# 	n = len(sources)
# 	sources_array = (ctypes.c_int * n)(*sources)
# 	combinations_count = math.comb(n, to_sum_count)
# 	all_combinations_length = combinations_count * to_sum_count
# 	all_combinations = [0] * all_combinations_length
# 	all_combinations_sum = [0] * combinations_count
# 	all_combinations_array = (ctypes.c_int * all_combinations_length)(*all_combinations)
# 	all_combinations_sum_array = (ctypes.c_int * combinations_count)(*all_combinations_sum)
# 	combinations_count = clib.old_combinations(sources_array, n, to_sum_count, all_combinations_array, all_combinations_sum_array)
# 	return combinations_count, list(all_combinations_array), list(all_combinations_sum_array)
