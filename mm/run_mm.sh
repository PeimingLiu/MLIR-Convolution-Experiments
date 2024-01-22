MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
ORDERS=(prpr prrp pprr rrpp)

ms=$(ls *.mtx)

for MAT in $ms;
do
  export TENSOR0=$MAT
  echo $MAT
  for ORD in ${ORDERS[*]};
  do
    echo $ORD
    sed "s/SCHEDULE/${ORD}/g" bench_mm.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so
  done
done
