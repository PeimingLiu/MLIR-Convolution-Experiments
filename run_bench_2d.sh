
MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static

sed 's/WIDTH/200/g;s/HEIGHT/200/g' benchmark_2d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so
