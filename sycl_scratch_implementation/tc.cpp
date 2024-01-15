#include <sycl/sycl.hpp>
#include <math.h>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <functional>
#include <chrono>
#include <iomanip>

using namespace sycl;

struct Output {
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

struct Entity {
    int key;
    int value;
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

struct is_equal {
    bool operator()(const Entity &lhs, const Entity &rhs) const {
        if ((lhs.key == rhs.key) && (lhs.value == rhs.value))
            return true;
        return false;
    }
};

double get_time_spent(std::string message,
                      std::chrono::high_resolution_clock::time_point time_point_begin,
                      std::chrono::high_resolution_clock::time_point time_point_end) {
    std::chrono::duration<double> time_span = time_point_end - time_point_begin;
    if (message != "")
        std::cout << message << ": " << time_span.count() << " seconds" << std::endl;
    return time_span.count();
}


long int get_row_size(const char *data_path) {
    std::ifstream data_file;
    char c;
    long int row_size = 0;
    data_file.open(data_path);
    while (data_file.get(c)) {
        if (c == '\n') {
            row_size++;
        }
    }
    data_file.close();
    return row_size;
}

void get_relation_from_file(Entity *data, const char *file_path, long int total_rows, char separator) {
    FILE *data_file = fopen(file_path, "r");
    for (long int i = 0; i < total_rows; i++) {
        fscanf(data_file, "%d%c", &data[i].key, &separator);
        fscanf(data_file, "%d", &data[i].value);
    }
}

void initialize_t_delta_result_array(queue &q, int data_row_size, Entity *data,
                                     Entity *t_delta, Entity *result) {
    auto task_reverse = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(range<1>(data_row_size), [=](id<1> idx) {
            result[idx].value = t_delta[idx].value = data[idx].value;
            result[idx].key = t_delta[idx].key = data[idx].key;
        });
    });
    task_reverse.wait();
}


void show_entity_array(Entity *data, long int data_rows, const char *array_name, long int max_count) {
    long int count = 0;
    std::cout << "Entity name: " << array_name << std::endl;
    std::cout << "===================================" << std::endl;
    for (long int i = 0; i < data_rows; i++) {
        // Hide Entity with key = -1 for Hash table
        if (data[i].key != -1) {
            std::cout << data[i].key << " " << data[i].value << std::endl;
            count++;
        }
        if (max_count > 0 && max_count == count) {
            break;
        }
    }
    std::cout << "Row showed " << count << "\n" << std::endl;
    std::cout << "" << std::endl;
}

long int get_position(int key, long int hash_table_row_size) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (hash_table_row_size - 1);
}


void build_hash_table(queue &q, Entity *hash_table, long int hash_table_row_size, Entity *relation,
                      long int relation_row_size) {
    auto task_build_hash_table = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(range<1>(relation_row_size), [=](id<1> idx) {
            long int position = get_position(relation[idx].key, hash_table_row_size);
            while (true) {
                auto existing_key = atomic_ref<int, memory_order::acq_rel,
                        memory_scope::device,
                        access::address_space::global_space>(
                        hash_table[position].key);
                int expected_key = -1;
                if (existing_key.load() == expected_key &&
                    existing_key.compare_exchange_strong(expected_key, relation[idx].key)) {
                    hash_table[position].value = relation[idx].value;
                    break;
                }
                position = (position + 1) & (hash_table_row_size - 1);
            }
        });
    });
    task_build_hash_table.wait();
}

void get_join_result_offset(queue &q,
                            Entity *hash_table, long int hash_table_row_size,
                            Entity *t_delta, long int t_delta_row_size,
                            int *join_result_offset) {
    auto task_join_result_size = q.submit([&](sycl::handler &cgh) {
//        stream sout(1024, 256, cgh);
        cgh.parallel_for(range<1>(t_delta_row_size), [=](id<1> idx) {
            // t_delta is reverse
            int key = t_delta[idx].value;
            int match_count = 0;
            long int position = get_position(key, hash_table_row_size);
            while (true) {
                if (hash_table[position].key == key) {
                    match_count++;
                } else if (hash_table[position].key == -1) {
                    break;
                }
                position = (position + 1) & (hash_table_row_size - 1);
            }
//            sout << idx << " " << t_delta[idx].value << " " << match_count << endl;
            join_result_offset[idx] = match_count;
        });
    });
    task_join_result_size.wait();
}

void get_join_result(queue &q,
                     Entity *hash_table, long int hash_table_row_size,
                     Entity *t_delta, long int t_delta_row_size,
                     int *join_result_offset, Entity *join_result) {
    auto task_join_result = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(range<1>(t_delta_row_size), [=](id<1> idx) {
            // t_delta is reverse
            int key = t_delta[idx].value;
            int value = t_delta[idx].key;
            int start_index = join_result_offset[idx];
            long int position = get_position(key, hash_table_row_size);
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
        });
    });
    task_join_result.wait();
}

void compute_tc(queue &q, const char *data_path, char separator,
                long int relation_row_size, const char *dataset_name) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    time_point_begin = std::chrono::high_resolution_clock::now();
    double load_factor = 0.4;
    long int hash_table_row_size = (long int) relation_row_size / load_factor;
    hash_table_row_size = pow(2, ceil(log(hash_table_row_size) / log(2)));
    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;

    Entity *relation = malloc_shared<Entity>(relation_row_size, q);
    Entity *result = malloc_shared<Entity>(relation_row_size, q);
    Entity *t_delta = malloc_shared<Entity>(relation_row_size, q);
    Entity *hash_table = malloc_shared<Entity>(hash_table_row_size, q);
    long int t_delta_row_size = relation_row_size;
    long int result_row_size = relation_row_size;

    auto fill_event = q.fill(hash_table, negative_entity, hash_table_row_size);
    fill_event.wait();

    get_relation_from_file(relation, data_path,
                           relation_row_size, separator);
    initialize_t_delta_result_array(q, relation_row_size, relation, t_delta, result);

    build_hash_table(q, hash_table, hash_table_row_size, relation, relation_row_size);

    std::stable_sort(result, result + result_row_size, cmp());

    int iterations = 0;
    // Iterations until no new tuple is found
    while (true) {
        iterations++;
        // Get the projected join result in 2 pass
        // First pass is to create the offset array for each row of the t_delta as GPU cannot do dynamic allocation
        // Second pass is to insert projected join result
        int *join_result_offset = malloc_shared<int>(t_delta_row_size, q);
        get_join_result_offset(q, hash_table, hash_table_row_size, t_delta, t_delta_row_size, join_result_offset);
//        long int join_result_row_size = oneapi::dpl::reduce(oneapi::dpl::execution::make_device_policy(q),
//                                                            join_result_offset, join_result_offset + t_delta_row_size,
//                                                            0);
//        oneapi::dpl::exclusive_scan(oneapi::dpl::execution::make_device_policy(q),
//                                    join_result_offset, join_result_offset + t_delta_row_size, join_result_offset, 0);

        long int start_index = 0;
        for (long int i = 0; i < t_delta_row_size; i++) {
            long int temp = join_result_offset[i];
            join_result_offset[i] = start_index;
            start_index += temp;
        }
        long int join_result_row_size = start_index;

        Entity *join_result = malloc_shared<Entity>(join_result_row_size, q);
        get_join_result(q, hash_table, hash_table_row_size, t_delta, t_delta_row_size,
                        join_result_offset, join_result);
        // Deduplication of projected join result
        // Sort and then remove consecutive duplicate Entity
        std::stable_sort(join_result, join_result + join_result_row_size, cmp());
        // Unique operation
        long int projection_row_size =
                (std::unique(oneapi::dpl::execution::make_device_policy(q),
                             join_result,
                             join_result + join_result_row_size, is_equal())) -
                join_result;
        // Update the t_delta for next iteration
        free(t_delta, q);
        t_delta = malloc_shared<Entity>(projection_row_size, q);
        std::copy(oneapi::dpl::execution::make_device_policy(q),
                  join_result, join_result + projection_row_size, t_delta);
        t_delta_row_size = projection_row_size;
        // Union phase
        long int concatenated_row_size = t_delta_row_size + result_row_size;
        Entity *concatenated_result = malloc_shared<Entity>(concatenated_row_size, q);
        // merge two sorted array: previous result and join result
        std::merge(oneapi::dpl::execution::make_device_policy(q),
                   result, result + result_row_size,
                   t_delta, t_delta + t_delta_row_size,
                   concatenated_result, cmp());
        long int deduplicated_result_row_size =
                (std::unique(oneapi::dpl::execution::make_device_policy(q),
                             concatenated_result,
                             concatenated_result + concatenated_row_size, is_equal())) -
                concatenated_result;
        free(result, q);
        result = malloc_shared<Entity>(deduplicated_result_row_size, q);
        std::copy(oneapi::dpl::execution::make_device_policy(q),
                  concatenated_result, concatenated_result + deduplicated_result_row_size,
                  result);
        free(join_result, q);
        free(join_result_offset, q);
        free(concatenated_result, q);
//        std::cout << "Iteration " << iterations << ": "
//                                << "join_result = " << join_result_row_size << ", "
//                  << "projection = " << projection_row_size << ", "
//                  //              << "concatenated_row_size = " << concatenated_row_size << ", "
//                  << "deduplicated_result = " << deduplicated_result_row_size << ", "
//                  << std::endl;
        if (result_row_size == deduplicated_result_row_size) {
            break;
        }
        result_row_size = deduplicated_result_row_size;
    }
    free(relation, q);
    free(result, q);
    free(t_delta, q);
    free(hash_table, q);
    time_point_end = std::chrono::high_resolution_clock::now();
    double calculated_time = get_time_spent("", time_point_begin, time_point_end);;
    std::cout << std::endl;
    std::cout << "| Dataset | Number of rows | TC size | Iterations | Time (s) |" << std::endl;
    std::cout << "| --- | --- | --- | --- | --- | --- |" << std::endl;
    std::cout << "| " << dataset_name << " | " << relation_row_size << " | " << result_row_size;
    std::cout << std::fixed << " | " << iterations << " | ";
    std::cout << std::fixed << calculated_time << " |\n" << std::endl;

}


int main() {
    // Set locale for printing numbers with commas as thousands separator
    std::locale loc("");
    std::cout.imbue(loc);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    queue q;
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>()
              << std::endl;

    // Separator character for dataset values
    char separator = '\t';

    // Array of dataset names and paths, filename pattern: data_<number_of_rows>.txt
    std::string datasets[] = {
            "OL.cedge", "data/data_7035.txt",
            "SF.cedge", "data/data_223001.txt",
            "ego-Facebook", "data/data_88234.txt",
            "wiki-Vote", "data/data_103689.txt",
//            "p2p-Gnutella09", "data/data_26013.txt",
//            "p2p-Gnutella04", "data/data_39994.txt",
            "cal.cedge", "data/data_21693.txt",
            "TG.cedge", "data/data_23874.txt",
            "luxembourg_osm", "data/data_119666.txt",
//            "fe_sphere", "data/data_49152.txt",
//            "fe_body", "data/data_163734.txt",
            "cti", "data/data_48232.txt",
//            "fe_ocean", "data/data_409593.txt",
            "wing", "data/data_121544.txt",
//            "loc-Brightkite", "data/data_214078.txt",
            "delaunay_n16", "data/data_196575.txt",
//            "usroads", "data/data_165435.txt",
//            "SF.cedge", "data/data_223001.txt"
//            "data_4", "data/data_4.txt",
//            "data_5", "data/data_51.txt",
//            "data_5", "data/data_5.txt",
//            "data_10", "data/data_10.txt",
//            "data_22", "data/data_22.txt",
//            "data_25", "data/data_25.txt",
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
        compute_tc(q, data_path, separator, row_size, dataset_name);
    }
    return 0;
}