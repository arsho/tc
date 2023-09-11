//
// Created by arsho on 01/09/23.
//
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;


class tc_sycl;

int main() {
    int i, j, k;
    // Read "hipc_2019.txt" which has two tab separated integers in each line and store them in vector
    std::vector<int> input_relation;
    int n = 0;
    // Define the file path
    std::string file_path = "hipc_2019.txt";
    // Open the file
    std::ifstream file(file_path);
    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }

    // Read and store the file contents
    while (file >> i >> j) {
        input_relation.push_back(i);
        input_relation.push_back(j);
        n++;
    }
    // Close the file
    file.close();

    size_t number_of_rows = n;
    std::vector<int> reverse_relation;
    reverse_relation.resize(number_of_rows * 2);
    try {
        auto q = queue{};
        buffer input_buffer(input_relation);
        buffer reverse_buffer(reverse_relation);
        q.submit([&](handler &cgh) {
            accessor input_accessor(input_buffer, cgh, read_write);
            accessor reverse_accessor(reverse_buffer, cgh, read_write);
            cgh.parallel_for<class natural_join_kernel>(range<1>(number_of_rows), [=](id<1> idx) {
                reverse_accessor[idx * 2] = input_accessor[(idx * 2) + 1];
                reverse_accessor[(idx * 2) + 1] = input_accessor[idx * 2];
            });

        });
        q.wait();
        q.throw_asynchronous();
    } catch (const exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    int join_result_size = 0;
    try {
        auto q = queue{};
        buffer input_buffer(input_relation);
        buffer reverse_buffer(reverse_relation);
        buffer<int> join_result_size_buffer(&join_result_size, 1);
        q.submit([&](handler &cgh) {
            accessor input_accessor(input_buffer, cgh, read_only);
            accessor reverse_accessor(reverse_buffer, cgh, read_only);
            auto join_result_size_reduction = reduction(join_result_size_buffer, cgh, 0, plus<>());
            cgh.parallel_for(range<1>(number_of_rows),
                             join_result_size_reduction,
                             [=](id<1> idx, auto &join_result_size_temp) {
                                 int input_value = input_accessor[idx * 2];
                                 for (size_t i = 0; i < number_of_rows; i++) {
                                     int reverse_value = reverse_accessor[i * 2];
                                     if (input_value == reverse_value) {
                                         join_result_size_temp += 1;
                                     }
                                 }
                             });
        });
        q.wait();
        q.throw_asynchronous();
    } catch (const exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    std::cout << "Join result size is: " << join_result_size << std::endl;
    std::vector<int> result_relation;
    result_relation.resize(join_result_size * 2);


    int join_index = 0;
    try {
        auto q = queue{};
        buffer input_buffer(input_relation);
        buffer reverse_buffer(reverse_relation);
        buffer result_buffer(result_relation);
        q.submit([&](handler &cgh) {
            accessor input_accessor(input_buffer, cgh, read_only);
            accessor reverse_accessor(reverse_buffer, cgh, read_only);
            accessor result_accessor(reverse_buffer, cgh, read_write);
            cgh.parallel_for(range<1>(number_of_rows),
                             [=](id<1> idx, auto &join_index_temp) {
                                 int input_value = input_accessor[idx * 2];
                                 for (size_t i = 0; i < number_of_rows; i++) {
                                     int reverse_value = reverse_accessor[i * 2];
                                     if (input_value == reverse_value) {
                                         sycl::atomic_ref<int> atomic_join_index(&join_index);
                                         size_t result_idx = atomic_join_index.fetch_add_explicit(1, sycl::memory_order_relaxed);
                                         // Insert the join result into result_relation
                                         result_idx *= 2;
                                         result_accessor[result_idx] = input_value;
                                         result_accessor[result_idx + 1] = reverse_value;
                                     }
                                 }
                             });
        });
        q.wait();
        q.throw_asynchronous();
    } catch (const exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    // Print the result_relation
    std::cout << "Result relation is: " << std::endl;
    for (size_t i = 0; i < result_relation.size(); i += 2) {
        std::cout << result_relation[i] << ", " << result_relation[i + 1] << std::endl;
    }

    // Free the memory
    input_relation.clear();
    reverse_relation.clear();
    result_relation.clear();
    return 0;
}
////    // Define a SYCL kernel to perform the natural join
////    q.submit([&](handler &cgh) {
////        auto input_accessor = input_buffer.get_access<access::mode::read>(cgh);
////        auto reverse_accessor = reverse_buffer.get_access<access::mode::read>(cgh);
////        auto result_accessor = result_buffer.get_access<access::mode::discard_write>(cgh);
////
////        cgh.parallel_for<class natural_join_kernel>(range<1>(number_of_rows), [=](id<1> idx) {
////            int input_value = input_accessor[idx][0]; // Get the first column value from input
////            for (size_t i = 0; i < number_of_rows; i++) {
////                int reverse_value = reverse_accessor[i][0]; // Get the first column value from reverse
////                if (input_value == reverse_value) {
////                    result_accessor[idx] = input_accessor[idx]; // Add the matching row to the result
////                    break; // Stop searching for matches
////                }
////            }
////        });
////    });
////
////    // Retrieve the result from the buffer
////    result_relation = result_buffer.get_access<access::mode::read>();
////
////    // Print the result
////    for (const auto &row: result_relation) {
////        for (int val: row) {
////            std::cout << val << "\t";
////        }
////        std::cout << std::endl;
////    }
//
//
//
////                int input_value = input_accessor[idx][0]; // Get the first column value from input
////                for (size_t i = 0; i < number_of_rows; i++) {
////                    int reverse_value = reverse_accessor[i][0]; // Get the first column value from reverse
////                    if (input_value == reverse_value) {
////                        result_accessor[idx] = input_accessor[idx]; // Add the matching row to the result
////                        break; // Stop searching for matches
////                    }
////                }
//
//
////    try {
////        q.submit([&](handler &cgh) {
////            accessor input_accessor(input_buffer, cgh, read_write);
////            accessor reverse_accessor(input_buffer, cgh, read_write);
////            cgh.parallel_for<tc_sycl>(
////                    range{number_of_rows},
////                    [=](id<1> idx) {
////                        std::reverse(input_accessor[idx].begin(), input_accessor[idx].end());
////                    }
////            );
////        });
////        q.wait();
////        q.throw_asynchronous();
////    } catch (const exception &e) {
////        std::cout << "Exception: " << e.what() << std::endl;
////    }
//// Print this data
////std::cout << "Input relation is: " << std::endl;
////for (i = 0; i < n; i++) {
////std::cout << input_relation.at(i).at(0) << " " << input_relation.at(i).at(1) << std::endl;
////}
////std::cout << std::endl;
////std::cout << "Reverse relation is: " << std::endl;
////for (i = 0; i < n; i++) {
////std::cout << reverse_relation.at(i).at(0) << " " << reverse_relation.at(i).at(1) << std::endl;
////}
