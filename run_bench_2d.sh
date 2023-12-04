MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
INPUT_WIDTH=200
INPUT_HEIGHT=200

sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/SCHEDULE/prpr/g" benchmark_2d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench2d_result_prpr.txt

sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/SCHEDULE/prrp/g" benchmark_2d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench2d_result_prrp.txt

sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/SCHEDULE/pprr/g" benchmark_2d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench2d_result_pprr.txt
