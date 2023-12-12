MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static

INPUT_WIDTH=999999
REPEAT=10

sed "s/LEN/${INPUT_WIDTH}/g;s/SCHEDULE/pr/g;s/REPEAT/${REPEAT}/g" benchmark_1d.mlir  | $MLIR_PATH/bin/mlir-opt --sparsifier | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench1d_result_pr.txt

sed "s/LEN/${INPUT_WIDTH}/g;s/SCHEDULE/rp/g;s/REPEAT/${REPEAT}/g" benchmark_1d.mlir  | $MLIR_PATH/bin/mlir-opt --sparsifier | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench1d_result_rp.txt
