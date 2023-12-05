MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
INPUT_WIDTH=20
INPUT_HEIGHT=20
INPUT_DEPTH=20
REPEAT=1

ORDERS=(ppprrr  prprrp  rrrppp  prprpr  pprrrp  prrprp)

for ORD in ${ORDERS[*]};
do
  sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/DEPTH/${INPUT_DEPTH}/g;s/SCHEDULE/${ORD}/g;s/REPEAT/${REPEAT}/g" benchmark_3d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so > bench3d_result_${ORD}.txt
done
