#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_N 1 << 16
#define MAX_R 3

int C[MAX_R + 1];

int binomial_coefficient(int r, int n) {
	for (int i = 0; i <= r; i++) C[i] = 0;
	C[0] = 1;
	for (int i = 1; i <= n; i++) {
		for (int j = (i < r ? i : r); j > 0; j--) {
			C[j] += C[j - 1];
		}
	}
	return C[r];
}

typedef struct combinations_const_args {
	int* values;
	int n;
	int r;
	int* cur_combination;
} combinations_const_args;

typedef struct combinations_var_args {
	int start;
	int index;
} combinations_var_args;

typedef struct combinations_result {
	int* all_combinations;
	int* all_combinations_sum;
	int cur;
} combinations_result;

void sort_combination(int* combination, int r) {
	for (int i = 0; i < r - 1; i++) {
		for (int j = i + 1; j < r; j++) {
			if (combination[i] > combination[j]) {
				int temp = combination[i];
				combination[i] = combination[j];
				combination[j] = temp;
			}
		}
	}
}

bool is_duplicate(int* all_combinations, int combinations_count, int r, int* cur_combination) {
	for (int i = 0; i < combinations_count; i++) {
		bool duplicate = true;
		for (int j = 0; j < r; j++) {
			if (all_combinations[i * r + j] != cur_combination[j]) {
				duplicate = false;
				break;
			}
		}
		if (duplicate) return true;
	}
	return false;
}

void old_generate_combinations(combinations_const_args* const_args, combinations_var_args* var_args, combinations_result* result) {
	int* values = const_args->values;
	int n = const_args->n;
	int r = const_args->r;
	int* cur_combination = const_args->cur_combination;

	int start = var_args->start;
	int index = var_args->index;
	
	if (index == r) {
		int cur = result->cur;
		int* all_combinations = result->all_combinations;
		int* all_combinations_sum = result->all_combinations_sum;
		
		sort_combination(cur_combination, r);
		if (is_duplicate(all_combinations, cur, r, cur_combination)) return;

		all_combinations_sum[cur] = 0;
		for (int i = 0; i < r; i++) {
			int tmp = cur_combination[i];
			all_combinations[cur * r + i] = tmp;
			all_combinations_sum[cur] += tmp;
		}
		
		result->cur++;
		return;
	}

	for (int i = start; i < n; i++) {
		cur_combination[index] = values[i];
		combinations_var_args new_var_args = (combinations_var_args){ start + 1, index + 1 };
		old_generate_combinations(const_args, &new_var_args, result);
	}
}

int old_combinations(int* values, int n, int r, int* all_combinations, int* all_combinations_sum) {	
	int cur_combination[r];

	combinations_const_args const_args;
	const_args.values = values;
	const_args.n = n;
	const_args.r = r;
	const_args.cur_combination = cur_combination;

	combinations_var_args var_args;
	var_args.start = 0;
	var_args.index = 0;

	combinations_result result;
	result.all_combinations = all_combinations;
	result.all_combinations_sum = all_combinations_sum;
	result.cur = 0;

	old_generate_combinations(&const_args, &var_args, &result);
		
	return result.cur;
}

// int max_stack_top = 0;

int combinations(int* values, int n, int r, int* all_combinations, int* all_combinations_sum) {
	int cur = 0;

	struct Stack {
		int start;
		int index;
		int cur_combination[MAX_R];
	};

	struct Stack stack[MAX_N];
	int stack_top = -1;
	stack[++stack_top] = (struct Stack){0, 0, {0}};

	while (stack_top >= 0) {
		// if (stack_top > max_stack_top) {
		// 	max_stack_top = stack_top;
		// }

		struct Stack current = stack[stack_top--];

		if (current.index == r) {
			sort_combination(current.cur_combination, r);
			if (is_duplicate(all_combinations, cur, r, current.cur_combination)) {
				continue;
			}
			all_combinations_sum[cur] = 0;
			for (int i = 0; i < r; i++) {
				int tmp = current.cur_combination[i];
				all_combinations[cur * r + i] = tmp;
				all_combinations_sum[cur] += tmp;
			}
			cur++;
			continue;
		}


		for (int i = current.start; i < n; i++) {
			current.cur_combination[current.index] = values[i];
			stack[++stack_top] = (struct Stack){i + 1, current.index + 1, {0}};
			memcpy(stack[stack_top].cur_combination, current.cur_combination, sizeof(current.cur_combination));
		}
	}

	return cur;
}

// max call stack is 2^16-1 = 65535
// int test_depth(int depth) {
// 	if (depth <= 0) return 12;
// 	return test_depth(depth - 1);
// }

// typedef struct test_result {
// 	int* arr;
// } test_result;

// test_result test(int* arr, int n) {
// 	test_result result;
// 	if (arr == NULL) {
// 		result.arr = NULL; // or handle the error as needed
// 		return result;
// 	}
// 	for (int i = 0; i < n; i++) arr[i] = i;
// 	result.arr = arr;
// 	return result;
// }

// int main() {
// 	test_result result = test();
// 	for (int i = 0; i < 100; i++) {
// 		printf("%d ", result.arr[i]);
// 	}
// 	free(result.arr);
// 	return 0;
// }

// void old_test_combinations() {
// 	int values[] = {25, 25, 75, 25};
// 	int n = sizeof(values) / sizeof(values[0]);
// 	int r = 2;
// 	int all_combinations_count = binomial_coefficient(r, n);
// 	int all_combinations[all_combinations_count * MAX_R];
// 	int all_combinations_sum[all_combinations_count];
// 	int combinations_count = old_combinations(values, n, r, (int*)all_combinations, (int*)all_combinations_sum);

// 	for (int i = 0; i < combinations_count; i++) {
// 		printf("sum(%d", all_combinations[i * r + 0]);
// 		for (int j = 1; j < r; j++) printf(", %d", all_combinations[i * r + j]);
// 		printf(") = %d\n", all_combinations_sum[i]);
// 	}
// }

// void test_combinations() {
// 	int values[] = {25, 25, 75, 25};
// 	int n = sizeof(values) / sizeof(values[0]);
// 	int r = 2;
// 	int all_combinations_count = binomial_coefficient(r, n);
// 	int all_combinations[all_combinations_count * MAX_R];
// 	int all_combinations_sum[all_combinations_count];
// 	int combinations_count = combinations(values, n, r, (int*)all_combinations, (int*)all_combinations_sum);

// 	for (int i = 0; i < combinations_count; i++) {
// 		printf("sum(%d", all_combinations[i * r + 0]);
// 		for (int j = 1; j < r; j++) printf(", %d", all_combinations[i * r + j]);
// 		printf(") = %d\n", all_combinations_sum[i]);
// 	}
// 	// printf("max_stack_top = %d\n", max_stack_top);
// }

// int main() {
// 	old_test_combinations();
// 	test_combinations();
// 	return 0;
// }