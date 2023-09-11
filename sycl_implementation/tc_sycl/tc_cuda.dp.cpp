#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <dpct/dpl_utils.hpp>

#include <functional>

#include <chrono>

using namespace std;
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(dpct::err0 code, const char *file, int line,
                      bool abort = true) {
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
    dpct::event_ptr start;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    dpct::event_ptr stop;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

    KernelTimer() {
        start = new sycl::event();
        stop = new sycl::event();
    }

    ~KernelTimer() {
        dpct::destroy_event(start);
        dpct::destroy_event(stop);
    }

    void start_timer() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        /*
        DPCT1012:5: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        start_ct1 = std::chrono::steady_clock::now();
        *start = q_ct1.ext_oneapi_submit_barrier();
    }

    void stop_timer() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        /*
        DPCT1012:6: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        *stop = q_ct1.ext_oneapi_submit_barrier();
    }

    float get_spent_time() {
        float elapsed;
        stop->wait_and_throw();
        elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                      .count();
        elapsed /= 1000.0;
        return elapsed;
    }
};

struct is_equal {
    
    bool operator()(const Entity &lhs, const Entity &rhs) const {
        if ((lhs.key == rhs.key) && (lhs.value == rhs.value))
            return true;
        return false;
    }
};


struct cmp {
    
    bool operator()(const Entity &lhs, const Entity &rhs) const {
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

int get_position(int key, int hash_table_row_size) {
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



void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      int *relation, long int relation_rows, int relation_columns,
                      const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= relation_rows) return;

    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    for (int i = index; i < relation_rows; i += stride) {
        int key = relation[(i * relation_columns) + 0];
        int value = relation[(i * relation_columns) + 1];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            int existing_key = dpct::atomic_compare_exchange_strong<
                sycl::access::address_space::generic_space>(
                &hash_table[position].key, -1, key);
            if (existing_key == -1) {
                hash_table[position].value = value;
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


void initialize_result_t_delta(Entity *result, Entity *t_delta,
                               int *relation, long int relation_rows, int relation_columns,
                               const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= relation_rows) return;

    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    for (int i = index; i < relation_rows; i += stride) {
        t_delta[i].key = result[i].key = relation[(i * relation_columns) + 0];
        t_delta[i].value = result[i].value = relation[(i * relation_columns) + 1];
    }
}


void copy_struct(Entity *source, long int source_rows, Entity *destination,
                 const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= source_rows) return;

    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    for (int i = index; i < source_rows; i += stride) {
        destination[i].key = source[i].key;
        destination[i].value = source[i].value;
    }
}


void negative_fill_struct(Entity *source, long int source_rows,
                          const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= source_rows) return;

    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    for (int i = index; i < source_rows; i += stride) {
        source[i].key = -1;
        source[i].value = -1;
    }
}


void get_reverse_relation(int *relation, long int relation_rows, int relation_columns, Entity *t_delta,
                          const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= relation_rows) return;

    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    for (long int i = index; i < relation_rows; i += stride) {
        t_delta[i].key = relation[(i * relation_columns) + 0];
        t_delta[i].value = relation[(i * relation_columns) + 1];
    }
}



void get_join_result_size(Entity *hash_table, long int hash_table_row_size,
                          Entity *t_delta, long int relation_rows,
                          int *join_result_size,
                          const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= relation_rows) return;

    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

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


void get_join_result(Entity *hash_table, int hash_table_row_size,
                     Entity *t_delta, int relation_rows, int *offset, Entity *join_result,
                     const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= relation_rows) return;
    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
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


void get_join_result_size_ar(Entity *hash_table, long int hash_table_row_size,
                             int *t_delta, long int relation_rows,
                             int *join_result_size,
                             const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= relation_rows) return;

    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

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


void get_join_result_ar(Entity *hash_table, int hash_table_row_size,
                        int *t_delta, int relation_rows, int *offset, Entity *join_result,
                        const sycl::nd_item<3> &item_ct1) {
    int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
                item_ct1.get_local_id(2);
    if (index >= relation_rows) return;
    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
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

void gpu_tc(const char *data_path, char separator, long int relation_rows,
            double load_factor, int preferred_grid_size,
            int preferred_block_size, const char *dataset_name,
            int number_of_sm) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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
    hash_table_rows = sycl::pow<double>(2, ceil(log(hash_table_rows) / log(2)));

    checkCuda(DPCT_CHECK_ERROR(relation_host = sycl::malloc_host<int>(
                                   relation_rows * relation_columns, q_ct1)));
    checkCuda(DPCT_CHECK_ERROR(relation = sycl::malloc_device<int>(
                                   relation_rows * relation_columns, q_ct1)));
    checkCuda(DPCT_CHECK_ERROR(
        result = sycl::malloc_device<Entity>(result_rows, q_ct1)));
    checkCuda(DPCT_CHECK_ERROR(
        t_delta = sycl::malloc_device<Entity>(relation_rows, q_ct1)));
    checkCuda(DPCT_CHECK_ERROR(
        hash_table = sycl::malloc_device<Entity>(hash_table_rows, q_ct1)));

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
    q_ct1
        .memcpy(relation, relation_host,
                relation_rows * relation_columns * sizeof(int))
        .wait();
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.read_time = spent_time;

    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;
    time_point_begin = chrono::high_resolution_clock::now();
    std::fill(oneapi::dpl::execution::make_device_policy(q_ct1), hash_table,
              hash_table + hash_table_rows, negative_entity);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    timer.start_timer();
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                           build_hash_table(hash_table, hash_table_rows,
                                            relation, relation_rows,
                                            relation_columns, item_ct1);
                       });
    checkCuda(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
    timer.stop_timer();
    spent_time = timer.get_spent_time();
    output.hashtable_build_time = spent_time;
    output.hashtable_build_rate = (double) relation_rows / spent_time;
    output.join_time += spent_time;

    timer.start_timer();
    // initial result and t delta both are same as the input relation
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                           initialize_result_t_delta(
                               result, t_delta, relation, relation_rows,
                               relation_columns, item_ct1);
                       });
    checkCuda(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
    timer.stop_timer();
    spent_time = timer.get_spent_time();
    output.union_time += spent_time;
    temp_time_begin = chrono::high_resolution_clock::now();
    oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1),
                             result, result + relation_rows, cmp());
    temp_time_end = chrono::high_resolution_clock::now();
    temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
    sort_time += temp_spent_time;
    output.deduplication_time += temp_spent_time;

    time_point_begin = chrono::high_resolution_clock::now();
    sycl::free(relation, q_ct1);
    sycl::free(relation_host, q_ct1);
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
        checkCuda(DPCT_CHECK_ERROR(
            offset = sycl::malloc_device<int>(t_delta_rows, q_ct1)));
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;
        timer.start_timer();
        // First pass to get the join result size for each row of t_delta
        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                get_join_result_size(hash_table, hash_table_rows, t_delta,
                                     t_delta_rows, offset, item_ct1);
            });
        checkCuda(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        timer.stop_timer();
        spent_time = timer.get_spent_time();
        temp_join += spent_time;
        output.join_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        join_result_rows =
            std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1),
                        offset, offset + t_delta_rows, 0);
        std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1),
                            offset, offset + t_delta_rows, offset,
                            0);
        checkCuda(DPCT_CHECK_ERROR(join_result = sycl::malloc_device<Entity>(
                                       join_result_rows, q_ct1)));
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;
        timer.start_timer();
        // Second pass to generate the join result of t_delta and the hash_table
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                get_join_result(hash_table, hash_table_rows, t_delta,
                                t_delta_rows, offset, join_result, item_ct1);
            });
        checkCuda(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        timer.stop_timer();
        spent_time = timer.get_spent_time();
        temp_join += spent_time;
        output.join_time += spent_time;
        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements
        temp_time_begin = chrono::high_resolution_clock::now();
        oneapi::dpl::stable_sort(
            oneapi::dpl::execution::make_device_policy(q_ct1), join_result,
            join_result + join_result_rows, cmp());
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_sort += temp_spent_time;
        temp_deduplication += temp_spent_time;
        sort_time += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        temp_time_begin = chrono::high_resolution_clock::now();
        long int projection_rows =
            (std::unique(oneapi::dpl::execution::make_device_policy(q_ct1),
                         join_result, join_result + join_result_rows,
                         is_equal())) -
            join_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_unique += temp_spent_time;
        temp_deduplication += temp_spent_time;
        unique_time += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        sycl::free(t_delta, q_ct1);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(DPCT_CHECK_ERROR(
            t_delta = sycl::malloc_device<Entity>(projection_rows, q_ct1)));
        std::copy(oneapi::dpl::execution::make_device_policy(q_ct1),
                  join_result, join_result + projection_rows, t_delta);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;

        time_point_begin = chrono::high_resolution_clock::now();
        Entity *concatenated_result;
        long int concatenated_rows = projection_rows + result_rows;
        checkCuda(DPCT_CHECK_ERROR(
            concatenated_result =
                sycl::malloc_device<Entity>(concatenated_rows, q_ct1)));
        temp_time_begin = chrono::high_resolution_clock::now();
        // merge two sorted array: previous result and join result
        std::merge(oneapi::dpl::execution::make_device_policy(q_ct1), result,
                   result + result_rows, join_result,
                   join_result + projection_rows, concatenated_result, cmp());
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
        deduplicated_result_rows =
            (std::unique(oneapi::dpl::execution::make_device_policy(q_ct1),
                         concatenated_result,
                         concatenated_result + concatenated_rows, is_equal())) -
            concatenated_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_unique += temp_spent_time;
        unique_time += temp_spent_time;
        temp_deduplication += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        sycl::free(result, q_ct1);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(DPCT_CHECK_ERROR(result = sycl::malloc_device<Entity>(
                                       deduplicated_result_rows, q_ct1)));
        // Copy the deduplicated concatenated result to result
        std::copy(oneapi::dpl::execution::make_device_policy(q_ct1),
                  concatenated_result,
                  concatenated_result + deduplicated_result_rows, result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_union += spent_time;
        output.union_time += spent_time; // changed this time from deduplication to union
        t_delta_rows = projection_rows;
        time_point_begin = chrono::high_resolution_clock::now();
        // Clear intermediate memory
        sycl::free(join_result, q_ct1);
        sycl::free(offset, q_ct1);
        sycl::free(concatenated_result, q_ct1);
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
    checkCuda(DPCT_CHECK_ERROR(
        result_host = sycl::malloc_host<Entity>(result_rows, q_ct1)));
    q_ct1.memcpy(result_host, result, result_rows * sizeof(Entity)).wait();
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.union_time += spent_time;
    if(relation_rows < 10) {
        show_entity_array(result_host, result_rows, "Result");
    }

    time_point_begin = chrono::high_resolution_clock::now();
    // Clear memory
    sycl::free(t_delta, q_ct1);
    sycl::free(result, q_ct1);
    sycl::free(hash_table, q_ct1);
    sycl::free(result_host, q_ct1);
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
    device_id = dpct::dev_mgr::instance().current_device_id();
    // Get the number of streaming multiprocessors (SM) on the device
    number_of_sm =
        dpct::dev_mgr::instance().get_device(device_id).get_max_compute_units();

    // Set locale for printing numbers with commas as thousands separator
    std::locale loc("");
    std::cout.imbue(loc);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);

    // Separator character for dataset names and paths
    char separator = '\t';

    // Array of dataset names and paths, filename pattern: data_<number_of_rows>.txt
    string datasets[] = {
            "OL.cedge_initial", "data_7035.txt",
            "HIPC", "data_5.txt",
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
