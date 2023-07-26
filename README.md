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
srun --pty bash
source /opt/intel/oneapi/setvars.sh
tar -xvf tc_sycl.tgz
cd tc_sycl
icpx -fsycl *.cpp
```
Error:
```shell
kernels.dp.cpp:7:23: error: unknown type name 'Entity'
void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      ^
kernels.dp.cpp:19:24: error: use of undeclared identifier 'get_position'
        int position = get_position(key, hash_table_row_size);
                       ^
kernels.dp.cpp:34:32: error: unknown type name 'Entity'
void initialize_result_t_delta(Entity *result, Entity *t_delta,
                               ^
kernels.dp.cpp:34:48: error: unknown type name 'Entity'
void initialize_result_t_delta(Entity *result, Entity *t_delta,
                                               ^
kernels.dp.cpp:50:18: error: unknown type name 'Entity'
void copy_struct(Entity *source, long int source_rows, Entity *destination,
                 ^
kernels.dp.cpp:50:56: error: unknown type name 'Entity'
void copy_struct(Entity *source, long int source_rows, Entity *destination,
                                                       ^
kernels.dp.cpp:65:27: error: unknown type name 'Entity'
void negative_fill_struct(Entity *source, long int source_rows,
                          ^
kernels.dp.cpp:80:88: error: unknown type name 'Entity'
void get_reverse_relation(int *relation, long int relation_rows, int relation_columns, Entity *t_delta,
                                                                                       ^
kernels.dp.cpp:96:27: error: unknown type name 'Entity'
void get_join_result_size(Entity *hash_table, long int hash_table_row_size,
                          ^
kernels.dp.cpp:97:27: error: unknown type name 'Entity'
                          Entity *t_delta, long int relation_rows,
                          ^
kernels.dp.cpp:109:24: error: use of undeclared identifier 'get_position'
        int position = get_position(key, hash_table_row_size);
                       ^
kernels.dp.cpp:123:22: error: unknown type name 'Entity'
void get_join_result(Entity *hash_table, int hash_table_row_size,
                     ^
kernels.dp.cpp:124:22: error: unknown type name 'Entity'
                     Entity *t_delta, int relation_rows, int *offset, Entity *join_result,
                     ^
kernels.dp.cpp:124:71: error: unknown type name 'Entity'
                     Entity *t_delta, int relation_rows, int *offset, Entity *join_result,
                                                                      ^
kernels.dp.cpp:134:24: error: use of undeclared identifier 'get_position'
        int position = get_position(key, hash_table_row_size);
                       ^
kernels.dp.cpp:149:30: error: unknown type name 'Entity'
void get_join_result_size_ar(Entity *hash_table, long int hash_table_row_size,
                             ^
kernels.dp.cpp:162:24: error: use of undeclared identifier 'get_position'
        int position = get_position(key, hash_table_row_size);
                       ^
kernels.dp.cpp:176:25: error: unknown type name 'Entity'
void get_join_result_ar(Entity *hash_table, int hash_table_row_size,
                        ^
kernels.dp.cpp:177:68: error: unknown type name 'Entity'
                     int *t_delta, int relation_rows, int *offset, Entity *join_result,
                                                                   ^
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated.
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
- [CUDA — Memory Model blog](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)
- [CUDA - Pinned memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)
