#!/bin/bash

base_dir=$(pwd)
batchnum_list=(1000 10000)
block_sizes=(16 32 48 64 96 128)

declare -A block_to_num_rank=(
    [16]=1
    [32]=2
    [48]=3
    [64]=4
    [96]=6
    [128]=8
)
magma_log_dir="${base_dir}/../logs/batched_gemm/H200"
mkdir -p "$magma_log_dir"
magma_log_prefix="${magma_log_dir}/MAGMA_batch_fp64"
magma_log_file="${magma_log_prefix}.log"
magma_csv_file="${magma_log_prefix}.csv"
echo "" >"$magma_log_file"

kami_log_prefix="${magma_log_dir}/KAMI_batch_fp64"
kami_log_file="${kami_log_prefix}.log"
kami_csv_file="${kami_log_prefix}.csv"
echo "" >"$kami_log_file"

if [ -z "$CUDADIR" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDADIR="/usr/local/cuda"
        echo "CUDADIR set to: $CUDADIR"
    else
        echo "WARNING: /usr/local/cuda not found and CUDADIR is not set."
    fi
else
    echo "CUDADIR already set to: $CUDADIR"
fi

if [ -n "$CUDADIR" ]; then
    CUDA_INCLUDE="$CUDADIR/include"
    CUDA_LIB="$CUDADIR/lib64"
    echo "CUDA_INCLUDE = $CUDA_INCLUDE"
    echo "CUDA_LIB = $CUDA_LIB"
else
    echo "CUDA_INCLUDE and CUDA_LIB not set because CUDADIR is missing."
fi

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    . /opt/intel/oneapi/setvars.sh
    echo "oneAPI environment sourced successfully."
    echo "LD_LIBRARY_PATH is now: $LD_LIBRARY_PATH"
else
    echo "WARNING: /opt/intel/oneapi/setvars.sh not found. oneAPI environment not set."
fi

if [ -n "$MKLROOT" ]; then
    MKL_INCLUDE="$MKLROOT/include"
    MKL_LIB="$MKLROOT/lib/intel64"
    echo "MKLROOT = $MKLROOT"
    echo "MKL_INCLUDE = $MKL_INCLUDE"
    echo "MKL_LIB = $MKL_LIB"
else
    echo "WARNING: MKLROOT not set. MKL_INCLUDE and MKL_LIB will be empty."
fi

MAGMA_DIR=$(find ~ -type d -name "magma-2.9.0" 2>/dev/null | head -n 1)
if [ -z "$MAGMA_DIR" ]; then
    echo "MAGMA 2.9.0 directory not found under ~, skipping MAGMA tests." | tee -a "$magma_log_file"
else
    echo "Found MAGMA 2.9.0 at: $MAGMA_DIR" | tee -a "$magma_log_file"
    cd "${base_dir}/../src/batched_gemm/MAGMA"
    echo "Compiling testing_dgemm_batched..." | tee -a "$magma_log_file"
    g++ -O3 -fPIC -fopenmp -DNDEBUG -DADD_ -Wall -Wno-strict-aliasing -Wshadow -DMAGMA_WITH_MKL -std=c++11 \
    -I"$CUDA_INCLUDE" \
    -I"$MKL_INCLUDE" \
    -I"$MAGMA_DIR/include" \
    -I"$MAGMA_DIR/testing" \
    ./testing_dgemm_batched.cpp -o testing_dgemm_batched \
    -L"$MAGMA_DIR/testing" \
    -L"$MAGMA_DIR/lib" \
    -L"$MAGMA_DIR/testing/lin" \
    -L"$CUDA_LIB" \
    -L"$MKL_LIB" \
    -ltest -lmagma -llapacktest -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core \
    -lpthread -lstdc++ -lm -lgfortran -lcublas -lcusparse -lcudart -lcudadevrt \
    -Wl,-rpath,"$MAGMA_DIR/lib" -Wl,-rpath,"$MKL_LIB" >>"$magma_log_file" 2>&1
    if [ ! -f ./testing_dgemm_batched ]; then
        echo "Compilation of testing_dgemm_batched failed." | tee -a "$magma_log_file"
    else
        echo "Compilation successful. Running testing_dgemm_batched..." | tee -a "$magma_log_file"
        for batch in "${batchnum_list[@]}"; do
            for n in "${block_sizes[@]}"; do
                echo "Running testing_dgemm_batched with batch=$batch, n=$n" | tee -a "$magma_log_file"
                ./testing_dgemm_batched --batch "$batch" -n "$n" >>"$magma_log_file" 2>&1
                echo "----------------------------------------" >>"$magma_log_file"
            done
        done
        echo "testing_dgemm_batched tests completed." | tee -a "$magma_log_file"
    fi
fi

cd "$base_dir" || exit

for batchnum in "${batchnum_list[@]}"; do
    for block_size in "${block_sizes[@]}"; do
        num_rank_block=${block_to_num_rank[$block_size]}
        if [ "$block_size" -eq 16 ]; then
            if [ "$batchnum" -eq 1000 ]; then
                exe="batched_gemm_2d_double_mma"
                exe_dir="${base_dir}/../src/batched_gemm/KAMI/2d"
            else
                exe="batched_gemm_3d_double_mma"
                exe_dir="${base_dir}/../src/batched_gemm/KAMI/3d"
            fi
        else
            exe="batched_gemm_1d_double_mma"
            exe_dir="${base_dir}/../src/batched_gemm/KAMI/1d"
        fi
        echo "Running $exe with M/N/K=$block_size, Batch=$batchnum, NUM_RANK_BLOCK=$num_rank_block" | tee -a "$kami_log_file"
        cd "$exe_dir" || {
            echo "Directory not found: $exe_dir" | tee -a "$kami_log_file"
            exit 1
        }
        rm -f $exe
        if [ "$block_size" -eq 128 ]; then
            src_file="batched_gemm_1d_double_mma_128.cu"
            extra_define="-DK_ALL_BLOCK=$block_size"
        else
            src_file="${exe}.cu"
            extra_define="-DK_BLOCK=$block_size"
        fi
        nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
            -DM_BLOCK=$block_size -DN_BLOCK=$block_size $extra_define \
            -DNUM_BATCHES=$batchnum -DNUM_RANK_BLOCK=$num_rank_block \
            -o $exe $src_file >>"$kami_log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "Compilation successful. Running..." | tee -a "$kami_log_file"
            ./$exe >>"$kami_log_file" 2>&1
        else
            echo "Compilation failed for M/N/K=$block_size, skipping..." | tee -a "$kami_log_file"
        fi

        cd "$base_dir"
        sleep 2
    done
done
grep '\[hemeng_log\]' "$kami_log_file" | sed 's/\[hemeng_log\],//' >"$kami_csv_file"
grep '\[hemeng_log\]' "$magma_log_file" | sed 's/\[hemeng_log\],//' >"$magma_csv_file"
echo "All tests completed."
echo "MAGMA logs: $magma_log_file, CSV: $magma_csv_file"
echo "KAMI logs: $kami_log_file, CSV: $kami_csv_file"
