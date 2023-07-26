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

#include "error_handler.dp.cpp"
#include "utils.dp.cpp"
#include "kernels.dp.cpp"

using namespace std;

void gpu_tc(const char *data_path, char separator, long int relation_rows,
            double load_factor, int preferred_grid_size,
            int preferred_block_size, const char *dataset_name,
            int number_of_sm) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
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
            std::reducet(oneapi::dpl::execution::make_device_policy(q_ct1),
                         offset, offset + t_delta_rows, 0);
        std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1),
                            offset, offset + t_delta_rows, offset,
                            (decltype(offset)::value_type)0);
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
