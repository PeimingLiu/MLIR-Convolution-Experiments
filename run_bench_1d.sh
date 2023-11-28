MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static

$MLIR_PATH/bin/mlir-opt benchmark_1d.mlir --sparsifier | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so
