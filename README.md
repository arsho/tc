## Transitive Closure Computation
This repo contains the code, data, and instructions to compute transitive closure in different parallel programming models.


## Folder structure
```
.
├── cuda_implementation
│   ├── data_7035.txt: Sample dataset
│   ├── error_handler.cu
│   ├── kernels.cu
│   ├── Makefile
│   ├── tc_cuda.cu: Main file
│   └── utils.cu
├── gitignore.txt
├── README.md
└── sycl_implementation

```


## Dependencies
### Hardware
- The complete benchmark of the CUDA-based transitive closure computation experiment can be executed on an Nvidia A100 GPU with a minimum of 40 GB GPU memory. The ThetaGPU single-GPU node is a suitable choice.
- Partial benchmarks can be run on other Nvidia GPUs, but they may result in program termination for certain datasets due to limited GPU memory, leading to an instance of the `std::bad_alloc: cudaErrorMemoryAllocation: out of memory` error.

### NVIDIA CUDA Toolkit (version 11.4.2 or later)
- Download and install the NVIDIA CUDA Toolkit from the NVIDIA website: [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
- Follow the installation instructions for your operating system. Make sure to install version 11.4.2 or later.

## Run CUDA Implementation
- To build and run the `Makefile`, navigate to the `code` directory containing the `Makefile`, `tc_cuda.cu`, and `hashjoin.cu` files and run the following command:
```
cd cuda_implementation
make run
```
This will build the `tc_cuda.out` executable using the nvcc compiler and run the test target to execute the program.
- After successful run, the output will be like:
```shell
Benchmark for OL.cedge_initial
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge_initial | 7035 | 146120 | 64 | 960 x 512 | 0.0276 |


Initialization: 0.0005, Read: 0.0103
Hashtable rate: 528470545 keys/s, time: 0.0000
Join: 0.0037
Deduplication: 0.0055 (sort: 0.0030, unique: 0.0026)
Memory clear: 0.0017
Union: 0.0058 (merge: 0.0014)
Total: 0.0276
```

## CUDA to SYCL Migration
- Install Syclomatic:
```shell
cd ~/
mkdir syclomatic
cd syclomatic
wget https://github.com/oneapi-src/SYCLomatic/releases/download/20230725/linux_release.tgz
tar -xvf linux_release.tgz
cd bin
pwd
# Add the following line to .zshrc
export PATH="/home/arsho/syclomatic/bin:$PATH"
```
- In nvidia machine:
```shell
cd sycl_implementation
make clean
intercept-build make
c2s -p compile_commands.json --out-root tc_sycl
cp data_7035.txt tc_sycl
tar -cvf tc_sycl.tgz tc_sycl
scp tc_sycl.tgz idc:~/
```
- In Intel dev cloud node:
```shell
ssh idc
srun --pty bash
source /opt/intel/oneapi/setvars.sh
rm -rf tc_sycl
tar -xvf tc_sycl.tgz
cd tc_sycl
icpx -fsycl *.cpp
```
- In converted SYCL code, replace `std::reducet` to `std::reduce`
Error:
```shell
u107416@idc-beta-batch-pvc-node-03:~$ icpx -fsycl tc_cuda.dp.cpp 
tc_cuda.dp.cpp:652:30: error: 'decltype(offset)' (aka 'int *') is not a class, namespace, or enumeration
                            (decltype(offset)::value_type)0);
                             ^
In file included from tc_cuda.dp.cpp:1:
In file included from /opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/execution:65:
In file included from /opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/algorithm_impl.h:24:
In file included from /opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/algorithm_fwd.h:20:
In file included from /opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/iterator_defs.h:21:
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/utils.h:110:17: error: no matching function for call to object of type 'const is_equal'
        return !_M_pred(::std::forward<_Args>(__args)...);
                ^~~~~~~
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/utils_hetero.h:74:34: note: in instantiation of function template specialization 'oneapi::dpl::__internal::__not_pred<is_equal>::operator()<Entity &, Entity &>' requested here
            __predicate_result = __predicate(get<0>(__acc[__idx]), get<0>(__acc[__idx + (-1)]));
                                 ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h:686:43: note: in instantiation of function template specialization 'oneapi::dpl::__internal::__create_mask_unique_copy<oneapi::dpl::__internal::__not_pred<is_equal>, long>::operator()<unsigned long, const oneapi::dpl::__ranges::zip_view<oneapi::dpl::__ranges::guard_view<Entity *>, oneapi::dpl::__ranges::all_view<int, sycl::access::mode::read_write>>>' requested here
                __local_acc[__local_id] = __data_acc(__adjusted_global_id, __acc);
                                          ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h:721:9: note: in instantiation of function template specialization 'oneapi::dpl::unseq_backend::__scan<std::integral_constant<bool, true>, oneapi::dpl::execution::device_policy<> &, std::plus<long>, oneapi::dpl::unseq_backend::walk_n<oneapi::dpl::execution::device_policy<> &, oneapi::dpl::__internal::__no_op>, oneapi::dpl::unseq_backend::__scan_assigner, oneapi::dpl::unseq_backend::__mask_assigner<1>, oneapi::dpl::__internal::__create_mask_unique_copy<oneapi::dpl::__internal::__not_pred<is_equal>, long>, oneapi::dpl::unseq_backend::__no_init_value<long>>::scan_impl<sycl::nd_item<>, long, const sycl::local_accessor<long>, oneapi::dpl::__ranges::zip_view<oneapi::dpl::__ranges::guard_view<Entity *>, oneapi::dpl::__ranges::all_view<int, sycl::access::mode::read_write>>, const oneapi::dpl::__ranges::all_view<Entity, sycl::access::mode::write>, const sycl::accessor<long, 1, sycl::access::mode::discard_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t>, unsigned long, unsigned long, unsigned long>' requested here
        scan_impl(__item, __n, __local_acc, __acc, __out_acc, __wg_sums_acc, __size_per_wg, __wgroup_size,
        ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:347:21: note: in instantiation of function template specialization 'oneapi::dpl::unseq_backend::__scan<std::integral_constant<bool, true>, oneapi::dpl::execution::device_policy<> &, std::plus<long>, oneapi::dpl::unseq_backend::walk_n<oneapi::dpl::execution::device_policy<> &, oneapi::dpl::__internal::__no_op>, oneapi::dpl::unseq_backend::__scan_assigner, oneapi::dpl::unseq_backend::__mask_assigner<1>, oneapi::dpl::__internal::__create_mask_unique_copy<oneapi::dpl::__internal::__not_pred<is_equal>, long>, oneapi::dpl::unseq_backend::__no_init_value<long>>::operator()<sycl::nd_item<>, long, const sycl::local_accessor<long>, oneapi::dpl::__ranges::zip_view<oneapi::dpl::__ranges::guard_view<Entity *>, oneapi::dpl::__ranges::all_view<int, sycl::access::mode::read_write>>, const oneapi::dpl::__ranges::all_view<Entity, sycl::access::mode::write>, const sycl::accessor<long, 1, sycl::access::mode::discard_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t>, unsigned long, unsigned long, unsigned long>' requested here
                    __local_scan(__item, __n, __local_acc, __rng1, __rng2, __wg_sums_acc, __size_per_wg, __wgroup_size,
                    ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h:903:21: note: in instantiation of function template specialization 'oneapi::dpl::__internal::__pattern_scan_copy<oneapi::dpl::execution::device_policy<> &, Entity *, oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, Entity>, oneapi::dpl::__internal::__create_mask_unique_copy<oneapi::dpl::__internal::__not_pred<is_equal>, long>, oneapi::dpl::unseq_backend::__copy_by_mask<std::plus<long>, oneapi::dpl::__internal::__pstl_assign, std::integral_constant<bool, true>, 1>>' requested here
    auto __result = __pattern_scan_copy(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result_first,
                    ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h:953:24: note: in instantiation of function template specialization 'oneapi::dpl::__internal::__pattern_unique_copy<oneapi::dpl::execution::device_policy<> &, Entity *, oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, Entity>, is_equal>' requested here
    auto __copy_last = __pattern_unique_copy(__exec, __first, __last, __copy_first, __pred,
                       ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/glue_algorithm_impl.h:516:37: note: in instantiation of function template specialization 'oneapi::dpl::__internal::__pattern_unique<oneapi::dpl::execution::device_policy<>, Entity *, is_equal>' requested here
    return oneapi::dpl::__internal::__pattern_unique(
                                    ^
tc_cuda.dp.cpp:693:19: note: in instantiation of function template specialization 'oneapi::dpl::unique<oneapi::dpl::execution::device_policy<>, Entity *, is_equal>' requested here
            (std::unique(oneapi::dpl::execution::make_device_policy(q_ct1),
                  ^
tc_cuda.dp.cpp:100:10: note: candidate function not viable: 'this' argument has type 'const is_equal', but method is not marked const
    bool operator()(const Entity &lhs, const Entity &rhs) {
         ^
In file included from tc_cuda.dp.cpp:1:
In file included from /opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/execution:65:
In file included from /opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/algorithm_impl.h:36:
In file included from /opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h:23:
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/utils_hetero.h:74:34: error: SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
            __predicate_result = __predicate(get<0>(__acc[__idx]), get<0>(__acc[__idx + (-1)]));
                                 ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/utils.h:108:5: note: 'operator()<Entity &, Entity &>' declared here
    operator()(_Args&&... __args) const
    ^
/opt/intel/oneapi/dpl/2022.2.0/linux/include/oneapi/dpl/pstl/hetero/utils_hetero.h:68:5: note: called by 'operator()<unsigned long, const oneapi::dpl::__ranges::zip_view<oneapi::dpl::__ranges::guard_view<Entity *>, oneapi::dpl::__ranges::all_view<int, sycl::access::mode::read_write>>>'
    operator()(_Idx __idx, _Acc& __acc) const
    ^
3 errors generated.
```
- Example of converting a single CUDA file:
```shell
c2s vectoradd.cu --gen-helper-function --out-root sycl_vector_add
```
## Citation
We encourage you to cite our work if you have used our work. Use the following BibTeX citation:
- BibTeX:
```
@inproceedings {288749,
    author = {Ahmedur Rahman Shovon and Thomas Gilray and Kristopher Micinski and Sidharth Kumar},
    title = {Towards Iterative Relational Algebra on the {GPU}},
    booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC 23)},
    year = {2023},
    isbn = {978-1-939133-35-9},
    address = {Boston, MA},
    pages = {1009--1016},
    url = {https://www.usenix.org/conference/atc23/presentation/shovon},
    publisher = {USENIX Association},
    month = jul,
}
```

### References
- [SYCL Migration - Sorting Networks](https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/C%2B%2BSYCL/Jupyter/cuda-to-sycl-migration-training/02_SYCL_Migration_SortingNetworks/02_SYCL_Migration_SortingNetworks.ipynb)
- [SYCL supported Thrust API](https://oneapi-src.github.io/SYCLomatic/dev_guide/api-mapping-status.html#thrust-api)
- [SYCL in IDC](https://github.com/bjodom/idc#an-example-script)
- [Intel Dev Cloud](https://scheduler.cloud.intel.com/#/systems)
- [Intel One API Samples](https://github.com/oneapi-src/oneAPI-samples/tree/master)
- [Discord channel](https://discord.com/channels/579789537866154054/1126249130726019143)