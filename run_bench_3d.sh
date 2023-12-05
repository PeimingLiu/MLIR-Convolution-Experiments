MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
INPUT_WIDTH=5
INPUT_HEIGHT=5
INPUT_DEPTH=5
REPEAT=5

sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/DEPTH/${INPUT_DEPTH}/g;s/SCHEDULE/ppprrr/g;s/REPEAT/${REPEAT}/g" benchmark_3d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench3d_result_ppprrr.txt

sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/DEPTH/${INPUT_DEPTH}/g;s/SCHEDULE/prprrp/g;s/REPEAT/${REPEAT}/g" benchmark_3d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench3d_result_prprrp.txt

sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/DEPTH/${INPUT_DEPTH}/g;s/SCHEDULE/rrrppp/g;s/REPEAT/${REPEAT}/g" benchmark_3d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench3d_result_rrrppp.txt
