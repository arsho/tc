#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(dpct::err0 code, const char *file, int line,
                      bool abort = true) {
    
}