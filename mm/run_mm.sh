export TENSOR0=bcsstk16.mtx

MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
ORDERS=(prpr prrp pprr rrpp)

for ORD in ${ORDERS[*]};
do
  echo $ORD
  sed "s/SCHEDULE/${ORD}/g" bench_mm.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so
done
