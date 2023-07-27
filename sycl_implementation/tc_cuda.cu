#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

using namespace std;
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct Entity {
    int key;
    int value;
};

struct Output {
    int block_size;
    int grid_size;
    long int input_rows;
    long int hashtable_rows;
    double load_factor;
    double initialization_time;
    double memory_clear_time;
    double read_time;
    double reverse_time;
    double hashtable_build_time;
    long int hashtable_build_rate;
    double join_time;
    double projection_time;
    double deduplication_time;
    double union_time;
    double total_time;
    const char *dataset_name;
} output;

struct KernelTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    KernelTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~KernelTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start_timer() {
        cudaEventRecord(start, 0);
    }

    void stop_timer() {
        cudaEventRecord(stop, 0);
    }

    float get_spent_time() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0;
        return elapsed;
    }
};

struct is_equal {
    __host__ __device__
    bool operator()(const Entity &lhs, const Entity &rhs) {
        if ((lhs.key == rhs.key) && (lhs.value == rhs.value))
            return true;
        return false;
    }
};


struct cmp {
    __host__ __device__
    bool operator()(const Entity &lhs, const Entity &rhs) {
        if (lhs.key < rhs.key)
            return true;
        else if (lhs.key > rhs.key)
            return false;
        else {
            if (lhs.value < rhs.value)
                return true;
            else if (lhs.value > rhs.value)
                return false;
            return true;
        }
    }
};

__device__ int get_position(int key, int hash_table_row_size) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (hash_table_row_size - 1);
}

void show_time_spent(string message,
                     chrono::high_resolution_clock::time_point time_point_begin,
                     chrono::high_resolution_clock::time_point time_point_end) {
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    cout << message << ": " << time_span.count() << " seconds" << endl;
}

double get_time_spent(string message,
                      chrono::high_resolution_clock::time_point time_point_begin,
                      chrono::high_resolution_clock::time_point time_point_end) {
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    if (message != "")
        cout << message << ": " << time_span.count() << " seconds" << endl;
    return time_span.count();
}

void show_relation(int *data, int total_rows,
                   int total_columns, const char *relation_name,
                   int visible_rows, int skip_zero) {
    int count = 0;
    cout << "Relation name: " << relation_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < total_rows; i++) {
        int skip = 0;
        for (int j = 0; j < total_columns; j++) {
            if ((skip_zero == 1) && (data[(i * total_columns) + j] == 0)) {
                skip = 1;
                continue;
            }
            cout << data[(i * total_columns) + j] << " ";
        }
        if (skip == 1)
            continue;
        cout << endl;
        count++;
        if (count == visible_rows) {
            cout << "Result cropped at row " << count << "\n" << endl;
            return;
        }

    }
    cout << "Result counts " << count << "\n" << endl;
    cout << "" << endl;
}

int *get_relation_from_file(const char *file_path, int total_rows, int total_columns, char separator) {
    int *data = (int *) malloc(total_rows * total_columns * sizeof(int));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                fscanf(data_file, "%d%c", &data[(i * total_columns) + j], &separator);
            } else {
                fscanf(data_file, "%d", &data[(i * total_columns) + j]);
            }
        }
    }
    return data;
}

void get_relation_from_file_gpu(int *data, const char *file_path, int total_rows, int total_columns, char separator) {
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                fscanf(data_file, "%d%c", &data[(i * total_columns) + j], &separator);
            } else {
                fscanf(data_file, "%d", &data[(i * total_columns) + j]);
            }
        }
    }
}


void get_random_relation(int *data, int total_rows, int total_columns) {
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            data[(i * total_columns) + j] = (rand() % (32767 - 0 + 1)) + 0;
        }
    }
}

void get_string_relation(int *data, int total_rows, int total_columns) {
    int x = 1, y = 2;
    for (int i = 0; i < total_rows; i++) {
        data[(i * total_columns) + 0] = x++;
        data[(i * total_columns) + 1] = y++;
    }
}

void get_reverse_relation_gpu(int *reverse_data, int *data, int total_rows, int total_columns) {
    for (int i = 0; i < total_rows; i++) {
        int pos = total_columns - 1;
        for (int j = 0; j < total_columns; j++) {
            reverse_data[(i * total_columns) + j] = data[(i * total_columns) + pos];
            pos--;
        }
    }
}


void show_hash_table(Entity *hash_table, long int hash_table_row_size, const char *hash_table_name) {
    int count = 0;
    cout << "Hashtable name: " << hash_table_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < hash_table_row_size; i++) {
        if (hash_table[i].key != -1) {
            cout << hash_table[i].key << " " << hash_table[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

void show_entity_array(Entity *data, int data_rows, const char *array_name) {
    long int count = 0;
    cout << "Entity name: " << array_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < data_rows; i++) {
        if (data[i].key != -1) {
            cout << data[i].key << " " << data[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

long int get_row_size(const char *data_path) {
    long int row_size = 0;
    int base = 1;
    for (int i = strlen(data_path) - 1; i >= 0; i--) {
        if (isdigit(data_path[i])) {
            int digit = (int) data_path[i] - '0';
            row_size += base * digit;
            base *= 10;
        }
    }
    return row_size;
}

/*
 * Method that returns position in the hashtable for a key using Murmur3 hash
 * */


__global__
void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = relation[(i * relation_columns) + 0];
        int value = relation[(i * relation_columns) + 1];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            int existing_key = atomicCAS(&hash_table[position].key, -1, key);
            if (existing_key == -1) {
                hash_table[position].value = value;
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}

__global__
void initialize_result_t_delta(Entity *result, Entity *t_delta,
                               int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        t_delta[i].key = result[i].key = relation[(i * relation_columns) + 0];
        t_delta[i].value = result[i].value = relation[(i * relation_columns) + 1];
    }
}

__global__
void copy_struct(Entity *source, long int source_rows, Entity *destination) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < source_rows; i += stride) {
        destination[i].key = source[i].key;
        destination[i].value = source[i].value;
    }
}

__global__
void negative_fill_struct(Entity *source, long int source_rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < source_rows; i += stride) {
        source[i].key = -1;
        source[i].value = -1;
    }
}

__global__
void get_reverse_relation(int *relation, long int relation_rows, int relation_columns, Entity *t_delta) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < relation_rows; i += stride) {
        t_delta[i].key = relation[(i * relation_columns) + 0];
        t_delta[i].value = relation[(i * relation_columns) + 1];
    }
}


__global__
void get_join_result_size(Entity *hash_table, long int hash_table_row_size,
                          Entity *t_delta, long int relation_rows,
                          int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[i].value;
        int current_size = 0;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

__global__
void get_join_result(Entity *hash_table, int hash_table_row_size,
                     Entity *t_delta, int relation_rows, int *offset, Entity *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[i].value;
        int value = t_delta[i].key;
        int start_index = offset[i];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].key = value;
                join_result[start_index].value = hash_table[position].value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}

__global__
void get_join_result_size_ar(Entity *hash_table, long int hash_table_row_size,
                             int *t_delta, long int relation_rows,
                             int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[(i * 2) + 1];
        int current_size = 0;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

__global__
void get_join_result_ar(Entity *hash_table, int hash_table_row_size,
                        int *t_delta, int relation_rows, int *offset, Entity *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[(i * 2) + 1];
        int value = t_delta[i * 2];
        int start_index = offset[i];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].key = value;
                join_result[start_index].value = hash_table[position].value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


void gpu_tc(const char *data_path, char separator,
            long int relation_rows, double load_factor,
            int preferred_grid_size, int preferred_block_size, const char *dataset_name, int number_of_sm) {
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point temp_time_begin;
    std::chrono::high_resolution_clock::time_point temp_time_end;
    KernelTimer timer;
    time_point_begin = chrono::high_resolution_clock::now();
    double spent_time;
    output.initialization_time = 0;
    output.join_time = 0;
    output.projection_time = 0;
    output.deduplication_time = 0;
    output.memory_clear_time = 0;
    output.union_time = 0;
    output.total_time = 0;
    double sort_time = 0.0;
    double unique_time = 0.0;
    double merge_time = 0.0;
    double temp_spent_time = 0.0;

    int block_size, grid_size;
    int *relation;
    int *relation_host;
    Entity *hash_table, *result, *t_delta;
    Entity *result_host;
    long int join_result_rows;
    long int t_delta_rows = relation_rows;
    long int result_rows = relation_rows;
    long int iterations = 0;
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));

    checkCuda(cudaMallocHost((void **) &relation_host, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMalloc((void **) &relation, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMalloc((void **) &result, result_rows * sizeof(Entity)));
    checkCuda(cudaMalloc((void **) &t_delta, relation_rows * sizeof(Entity)));
    checkCuda(cudaMalloc((void **) &hash_table, hash_table_rows * sizeof(Entity)));

    // Block size is 512 if preferred_block_size is 0
    block_size = 512;
    // Grid size is 32 times of the number of streaming multiprocessors if preferred_grid_size is 0
    grid_size = 32 * number_of_sm;
    if (preferred_grid_size != 0) {
        grid_size = preferred_grid_size;
    }
    if (preferred_block_size != 0) {
        block_size = preferred_block_size;
    }
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    get_relation_from_file_gpu(relation_host, data_path,
                               relation_rows, relation_columns, separator);
    cudaMemcpy(relation, relation_host, relation_rows * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.read_time = spent_time;

    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;
    time_point_begin = chrono::high_resolution_clock::now();
    thrust::fill(thrust::device, hash_table, hash_table + hash_table_rows, negative_entity);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    timer.start_timer();
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    spent_time = timer.get_spent_time();
    output.hashtable_build_time = spent_time;
    output.hashtable_build_rate = (double) relation_rows / spent_time;
    output.join_time += spent_time;

    timer.start_timer();
    // initial result and t delta both are same as the input relation
    initialize_result_t_delta<<<grid_size, block_size>>>(result, t_delta, relation, relation_rows, relation_columns);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    spent_time = timer.get_spent_time();
    output.union_time += spent_time;
    temp_time_begin = chrono::high_resolution_clock::now();
    thrust::stable_sort(thrust::device, result, result + relation_rows, cmp());
    temp_time_end = chrono::high_resolution_clock::now();
    temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
    sort_time += temp_spent_time;
    output.deduplication_time += temp_spent_time;

    time_point_begin = chrono::high_resolution_clock::now();
    cudaFree(relation);
    cudaFreeHost(relation_host);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;

    // Run the fixed point iterations for transitive closure computation
    while (true) {
        double temp_join = 0.0, temp_union = 0.0, temp_deduplication = 0.0, temp_memory_clear = 0.0;
        double temp_merge = 0.0, temp_sort = 0.0, temp_unique = 0.0;
        time_point_begin = chrono::high_resolution_clock::now();
        int *offset;
        Entity *join_result;
        checkCuda(cudaMalloc((void **) &offset, t_delta_rows * sizeof(int)));
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;
        timer.start_timer();
        // First pass to get the join result size for each row of t_delta
        get_join_result_size<<<grid_size, block_size>>>(hash_table, hash_table_rows, t_delta, t_delta_rows,
                                                        offset);
        checkCuda(cudaDeviceSynchronize());
        timer.stop_timer();
        spent_time = timer.get_spent_time();
        temp_join += spent_time;
        output.join_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        join_result_rows = thrust::reduce(thrust::device, offset, offset + t_delta_rows, 0);
        thrust::exclusive_scan(thrust::device, offset, offset + t_delta_rows, offset);
        checkCuda(cudaMalloc((void **) &join_result, join_result_rows * sizeof(Entity)));
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;
        timer.start_timer();
        // Second pass to generate the join result of t_delta and the hash_table
        get_join_result<<<grid_size, block_size>>>(hash_table, hash_table_rows,
                                                   t_delta, t_delta_rows, offset, join_result);
        checkCuda(cudaDeviceSynchronize());
        timer.stop_timer();
        spent_time = timer.get_spent_time();
        temp_join += spent_time;
        output.join_time += spent_time;
        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements
        temp_time_begin = chrono::high_resolution_clock::now();
        thrust::stable_sort(thrust::device, join_result, join_result + join_result_rows, cmp());
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_sort += temp_spent_time;
        temp_deduplication += temp_spent_time;
        sort_time += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        temp_time_begin = chrono::high_resolution_clock::now();
        long int projection_rows = (thrust::unique(thrust::device,
                                                   join_result, join_result + join_result_rows,
                                                   is_equal())) - join_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_unique += temp_spent_time;
        temp_deduplication += temp_spent_time;
        unique_time += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(t_delta);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(cudaMalloc((void **) &t_delta, projection_rows * sizeof(Entity)));
        thrust::copy(thrust::device, join_result, join_result + projection_rows, t_delta);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;

        time_point_begin = chrono::high_resolution_clock::now();
        Entity *concatenated_result;
        long int concatenated_rows = projection_rows + result_rows;
        checkCuda(cudaMalloc((void **) &concatenated_result, concatenated_rows * sizeof(Entity)));
        temp_time_begin = chrono::high_resolution_clock::now();
        // merge two sorted array: previous result and join result
        thrust::merge(thrust::device,
                      result, result + result_rows,
                      join_result, join_result + projection_rows,
                      concatenated_result, cmp());
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_merge += temp_spent_time;
        merge_time += temp_spent_time;
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_union += spent_time;
        output.union_time += spent_time;
        long int deduplicated_result_rows;
        temp_time_begin = chrono::high_resolution_clock::now();
        deduplicated_result_rows = (thrust::unique(thrust::device,
                                                   concatenated_result,
                                                   concatenated_result + concatenated_rows,
                                                   is_equal())) - concatenated_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_unique += temp_spent_time;
        unique_time += temp_spent_time;
        temp_deduplication += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(cudaMalloc((void **) &result, deduplicated_result_rows * sizeof(Entity)));
        // Copy the deduplicated concatenated result to result
        thrust::copy(thrust::device, concatenated_result,
                     concatenated_result + deduplicated_result_rows, result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_union += spent_time;
        output.union_time += spent_time; // changed this time from deduplication to union
        t_delta_rows = projection_rows;
        time_point_begin = chrono::high_resolution_clock::now();
        // Clear intermediate memory
        cudaFree(join_result);
        cudaFree(offset);
        cudaFree(concatenated_result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;

        if (result_rows == deduplicated_result_rows) {
            iterations++;
            break;
        }
        result_rows = deduplicated_result_rows;
        iterations++;
    }
    time_point_begin = chrono::high_resolution_clock::now();
    checkCuda(cudaMallocHost((void **) &result_host, result_rows * sizeof(Entity)));
    cudaMemcpy(result_host, result, result_rows * sizeof(Entity),
               cudaMemcpyDeviceToHost);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.union_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    // Clear memory
    cudaFree(t_delta);
    cudaFree(result);
    cudaFree(hash_table);
    cudaFreeHost(result_host);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;
    double calculated_time = output.initialization_time +
                             output.read_time + output.reverse_time + output.hashtable_build_time + output.join_time +
                             output.projection_time +
                             output.union_time + output.deduplication_time + output.memory_clear_time;
    cout << endl;
    cout << "| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |" << endl;
    cout << "| --- | --- | --- | --- | --- | --- |" << endl;
    cout << "| " << dataset_name << " | " << relation_rows << " | " << result_rows;
    cout << fixed << " | " << iterations << " | ";
    cout << fixed << grid_size << " x " << block_size << " | " << calculated_time << " |\n" << endl;
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.input_rows = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_rows;
    output.dataset_name = dataset_name;
    output.total_time = calculated_time;

    cout << endl;
    cout << "Initialization: " << output.initialization_time;
    cout << ", Read: " << output.read_time << endl;
    cout << "Hashtable rate: " << output.hashtable_build_rate << " keys/s, time: ";
    cout << output.hashtable_build_time << endl;
    cout << "Join: " << output.join_time << endl;
    cout << "Deduplication: " << output.deduplication_time;
    cout << " (sort: " << sort_time << ", unique: " << unique_time << ")" << endl;
    cout << "Memory clear: " << output.memory_clear_time << endl;
    cout << "Union: " << output.union_time << " (merge: " << merge_time << ")" << endl;
    cout << "Total: " << output.total_time << endl;
}

void run_benchmark(int grid_size, int block_size, double load_factor) {
    // Variables to store device information
    int device_id;
    int number_of_sm;

    // Get the current CUDA device
    cudaGetDevice(&device_id);
    // Get the number of streaming multiprocessors (SM) on the device
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);

    // Set locale for printing numbers with commas as thousands separator
    std::locale loc("");
    std::cout.imbue(loc);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);

    // Separator character for dataset names and paths
    char separator = '\t';

    // Array of dataset names and paths, filename pattern: data_<number_of_rows>.txt
    string datasets[] = {
            "OL.cedge_initial", "data_7035.txt"
    };

    // Iterate over the datasets array
    // Each iteration processes a dataset
    for (int i = 0; i < sizeof(datasets) / sizeof(datasets[0]); i += 2) {
        const char *data_path, *dataset_name;
        // Extract the dataset name and path from the array
        dataset_name = datasets[i].c_str();
        data_path = datasets[i + 1].c_str();

        // Get the row size of the dataset
        long int row_size = get_row_size(data_path);

        // Print benchmark information for the current dataset
        cout << "Benchmark for " << dataset_name << endl;
        cout << "----------------------------------------------------------" << endl;

        // Run the GPU graph processing function with the dataset parameters
        gpu_tc(data_path, separator,
               row_size, load_factor,
               grid_size, block_size, dataset_name, number_of_sm);

        cout << endl;
    }
}


int main() {
    run_benchmark(0, 0, 0.4);
    return 0;
}

/*
Run instructions:
make run
*/
