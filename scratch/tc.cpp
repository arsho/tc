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
    std::vector <std::vector<int>> input_relation, reverse_relation, result_relation;
    std::vector<int> row;
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
        row.push_back(i);
        row.push_back(j);
        input_relation.push_back(row);
        std::reverse(row.begin(), row.end());
        reverse_relation.push_back(row);
        row.clear();
        n++;
    }
    // Close the file
    file.close();

    size_t number_of_rows = n;
    auto q = queue{};
    buffer input_buffer(input_relation);
    buffer reverse_buffer(reverse_relation);
    buffer result_buffer(result_relation); // Create a buffer for the result


    try {
        q.submit([&](handler &cgh) {
            accessor input_accessor(input_buffer, cgh, read_write);
            accessor reverse_accessor(input_buffer, cgh, read_write);
            accessor result_accessor(result_buffer, cgh, read_write);

            cgh.parallel_for<class natural_join_kernel>(range<1>(number_of_rows), [=](id<1> idx) {
                int input_value = input_accessor[idx][0]; // Get the first column value from input
                for (size_t i = 0; i < number_of_rows; i++) {
                    int reverse_value = reverse_accessor[i][0]; // Get the first column value from reverse
                    if (input_value == reverse_value) {
                        result_accessor[idx] = input_accessor[idx]; // Add the matching row to the result
                        break; // Stop searching for matches
                    }
                }
            });


//            cgh.parallel_for<tc_sycl>(
//                    range{number_of_rows},
//                    [=](id<1> idx) {
//                        std::reverse(input_accessor[idx].begin(), input_accessor[idx].end());
//                    }
//            );
        });
        q.wait();
        q.throw_asynchronous();
    } catch (const exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }


    // Define a SYCL kernel to perform the natural join
    q.submit([&](handler &cgh) {
        auto input_accessor = input_buffer.get_access<access::mode::read>(cgh);
        auto reverse_accessor = reverse_buffer.get_access<access::mode::read>(cgh);
        auto result_accessor = result_buffer.get_access<access::mode::discard_write>(cgh);

        cgh.parallel_for<class natural_join_kernel>(range<1>(number_of_rows), [=](id<1> idx) {
            int input_value = input_accessor[idx][0]; // Get the first column value from input
            for (size_t i = 0; i < number_of_rows; i++) {
                int reverse_value = reverse_accessor[i][0]; // Get the first column value from reverse
                if (input_value == reverse_value) {
                    result_accessor[idx] = input_accessor[idx]; // Add the matching row to the result
                    break; // Stop searching for matches
                }
            }
        });
    });

    // Retrieve the result from the buffer
    result_relation = result_buffer.get_access<access::mode::read>();

    // Print the result
    for (const auto &row: result_relation) {
        for (int val: row) {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }


    // Free the memory
    input_relation.clear();
    reverse_relation.clear();
    return 0;
}


//    try {
//        q.submit([&](handler &cgh) {
//            accessor input_accessor(input_buffer, cgh, read_write);
//            accessor reverse_accessor(input_buffer, cgh, read_write);
//            cgh.parallel_for<tc_sycl>(
//                    range{number_of_rows},
//                    [=](id<1> idx) {
//                        std::reverse(input_accessor[idx].begin(), input_accessor[idx].end());
//                    }
//            );
//        });
//        q.wait();
//        q.throw_asynchronous();
//    } catch (const exception &e) {
//        std::cout << "Exception: " << e.what() << std::endl;
//    }
// Print this data
//std::cout << "Input relation is: " << std::endl;
//for (i = 0; i < n; i++) {
//std::cout << input_relation.at(i).at(0) << " " << input_relation.at(i).at(1) << std::endl;
//}
//std::cout << std::endl;
//std::cout << "Reverse relation is: " << std::endl;
//for (i = 0; i < n; i++) {
//std::cout << reverse_relation.at(i).at(0) << " " << reverse_relation.at(i).at(1) << std::endl;
//}
