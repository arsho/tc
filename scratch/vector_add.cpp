//
// Created by arsho on 01/09/23.
//
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;


class vector_add;

int main() {
    int i, j, k;
    size_t n = 15;
    std::vector<double> a, b, c;
    a.resize(n);
    b.resize(n);
    c.resize(n);
    for (i = 0; i < n; i++) {
        a.at(i) = static_cast<double>(i);
        b.at(i) = static_cast<double>(i);
        c.at(i) = 0.0;
    }
    try {
        auto q = queue{};
        buffer buffer_a(a);
        buffer buffer_b(b);
        buffer buffer_c(c);
        q.submit([&](handler &cgh) {
            accessor accessor_a(buffer_a, cgh, read_only);
            accessor accessor_b(buffer_b, cgh, read_only);
            accessor accessor_c(buffer_c, cgh, write_only);

            cgh.parallel_for<vector_add>(
                    range{n},
                    [=](id<1> idx) {
                        accessor_c[idx] = accessor_a[idx] + accessor_b[idx];
                    }
            );
        });
        q.wait();
        q.throw_asynchronous();
    } catch (const exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    for (i = 0; i < n; i++) {
        std::cout << a.at(i) << " + " << b.at(i) << " = " << c.at(i) << std::endl;
    }
    a.clear();
    b.clear();
    c.clear();
    return 0;
}
