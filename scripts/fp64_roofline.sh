run_cublas_test() {
    nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -lcublas -O3 ../src/roofline/cublas_gemm_double.cu -o ../src/roofline/cublas_gemm_double
    if [ $? -ne 0 ]; then
        echo "Compilation failed for cuBLAS test"
        exit 1
    fi

    log_file="../logs/roofline/fp64_GH200_smallsize_avg.log"
    csv_file="../logs/roofline/fp64_GH200_smallsize_avg.csv"
    echo "" >"$log_file"

    chmod +x ../src/roofline/cublas_gemm_double
    for i in {1..1024..1}; do
        echo "Running with INPUT_M=$i"
        ../src/roofline/cublas_gemm_double $i $i $i >>"$log_file" 2>&1
    done

    grep '\[hemeng_log\]' "$log_file" | sed 's/\[hemeng_log\]//' >"$csv_file"

    log_file="../logs/roofline/fp64_GH200_largesize_avg.log"
    csv_file="../logs/roofline/fp64_GH200_largesize_avg.csv"
    echo "" >"$log_file"

    chmod +x ../src/roofline/cublas_gemm_double
    for i in {1024..8192..16}; do
        echo "Running with INPUT_M=$i"
        ../src/roofline/cublas_gemm_double $i $i $i >>"$log_file" 2>&1
    done

    grep '\[hemeng_log\]' "$log_file" | sed 's/\[hemeng_log\]//' >"$csv_file"
}

run_cublasdx_test() {
    local src_dir=$1
    local log_prefix=$2

    cd "$src_dir" || {
        echo "Directory not found: $src_dir"
        exit 1
    }
    log_file="${log_prefix}.log"
    csv_file="${log_prefix}.csv"
    echo "" >"$log_file"

    for value in $(seq 1 100); do
        echo "Running cuBLASDx test with INPUT_M=$value and INPUT_K=$value..." | tee -a $log_file

        rm -rf fp64_single_gemm_performance

        nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -lcublas -O3 --ptxas-options=-v -lineinfo \
            -DCUBLASDX_EXAMPLE_ENABLE_SM_90 -DINPUT_M=$value -DINPUT_K=$value \
            -I./24.08/include/cublasdx/include/ \
            -I./24.08/external/cutlass/include/ -I./24.08/include \
            -o fp64_single_gemm_performance fp64_single_gemm_performance.cu

        if [ ! -f "./fp64_single_gemm_performance" ]; then
            echo "Compilation failed for cuBLASDx with INPUT_M=$value and INPUT_K=$value" | tee -a $log_file
            continue
        fi
        chmod +x ./fp64_single_gemm_performance
        ./fp64_single_gemm_performance >>$log_file 2>&1
    done

    grep '\[hemeng_log\]' "$log_file" | sed 's/\[hemeng_log\],//' >"$csv_file"
    cd - >/dev/null
}

run_cublas_test

run_cublasdx_test "../src/block_gemm/cuBLASDx" "../../../logs/roofline/fp64_GH200_cuBLASDx"
