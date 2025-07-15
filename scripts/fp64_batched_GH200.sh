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

magma_log_dir="${base_dir}/../logs/batched_gemm/GH200"
mkdir -p "$magma_log_dir"
magma_log_prefix="${magma_log_dir}/MAGMA_batch_fp64"
magma_log_file="${magma_log_prefix}.log"
magma_csv_file="${magma_log_prefix}.csv"
echo "" >"$magma_log_file"

kami_log_prefix="${magma_log_dir}/KAMI_batch_fp64"
kami_log_file="${kami_log_prefix}.log"
kami_csv_file="${kami_log_prefix}.csv"
echo "" >"$kami_log_file"

export CUDADIR=/usr/local/cuda
. /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2025.0/lib/intel64:$LD_LIBRARY_PATH

MAGMA_DIR=$(find ~ -type d -name "magma-2.*" 2>/dev/null | head -n 1)
if [ -z "$MAGMA_DIR" ]; then
    echo "MAGMA directory not found under ~" | tee -a "$magma_log_file"
    exit 1
fi

echo "Found MAGMA directory at: $MAGMA_DIR" | tee -a "$magma_log_file"

magma_src_dir="${base_dir}/../src/batched_gemm/MAGMA"
cd "$magma_src_dir" || {
    echo "MAGMA source dir not found: $magma_src_dir" | tee -a "$magma_log_file"
    exit 1
}

echo "Compiling testing_dgemm_batched..." | tee -a "$magma_log_file"
g++ -O3 -fPIC -fopenmp -DNDEBUG -DADD_ -Wall -Wno-strict-aliasing -Wshadow -DMAGMA_WITH_MKL -std=c++11 \
    -I/usr/local/cuda/include \
    -I/opt/intel/oneapi/mkl/2025.0/include \
    -I"$MAGMA_DIR/include" \
    -I"$MAGMA_DIR/testing" \
    ./testing_dgemm_batched.cpp -o testing_dgemm_batched \
    -L"$MAGMA_DIR/testing" \
    -L"$MAGMA_DIR/lib" \
    -L"$MAGMA_DIR/testing/lin" \
    -L/usr/local/cuda/lib64 \
    -L/opt/intel/oneapi/mkl/2025.0/lib/intel64 \
    -ltest -lmagma -llapacktest -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core \
    -lpthread -lstdc++ -lm -lgfortran -lcublas -lcusparse -lcudart -lcudadevrt \
    -Wl,-rpath,"$MAGMA_DIR/lib" -Wl,-rpath,/opt/intel/oneapi/mkl/2025.0/lib/intel64 >>"$magma_log_file" 2>&1

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
