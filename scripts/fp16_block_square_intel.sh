#!/bin/bash

run_block_test() {
    local dir=$1
    local log_path=$2
    local mode=$3

    if [ ! -d "$dir" ]; then
        echo "Directory does not exist: $dir"
        exit 1
    fi
    local curr_dir=$(pwd)
    cd "$dir"
    echo "" >"${log_path}.log"

    declare -A M_TO_NUM_RANK
    case "$mode" in
    1d)
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=1 [128]=2 [192]=3)
        ;;
    2d)
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=1 [128]=2 [192]=2)
        ;;
    3d)
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=2 [128]=2 [192]=2)
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
    esac

    for m_block in 16 32 64 128 192; do
        num_rank=${M_TO_NUM_RANK[$m_block]}
        if [ -z "$num_rank" ]; then
            echo "No NUM_RANK_BLOCK found for M_BLOCK=${m_block}"
            continue
        fi

        echo "Testing: M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"
        make clean
        make -j40 INPUT_M=${m_block} INPUT_N=${m_block} INPUT_K=${m_block} num_block=${num_rank} >>"${log_path}.log" 2>&1
    done

    grep '\[hemeng_log\]' "${log_path}.log" | sed 's/\[hemeng_log\],//' >"${log_path}.csv"
    cd - >/dev/null
}

run_block_test "../src/block_gemm/intel/1d" "../../../../logs/block_gemm/square/intel/fp16_block_square_1d_intel" 1d
run_block_test "../src/block_gemm/intel/2d" "../../../../logs/block_gemm/square/intel/fp16_block_square_2d_intel" 2d
run_block_test "../src/block_gemm/intel/3d" "../../../../logs/block_gemm/square/intel/fp16_block_square_3d_intel" 3d

cd ../src/block_gemm/intel/basic || exit 1
log_file="../../../../logs/block_gemm/square/intel/fp16_basic_intel"
mkdir -p "$(dirname "${log_file}")"
echo "" >"${log_file}.log"

for m in 16 32 64 128 192; do
    echo "Running basic_gemm ${m}"
    ./basic_gemm "$m" >>"${log_file}.log" 2>&1
done

grep '\[hemeng_log\]' "${log_file}.log" | sed 's/\[hemeng_log\],//' >"${log_file}.csv"
