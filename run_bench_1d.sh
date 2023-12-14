MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
INPUT_WIDTH=999999
REPEAT=5

ORDERS=(pr rp)
for ORD in ${ORDERS[*]};
do
  echo $ORD > ./result/bench1d_result_${ORD}.txt
  sed "s/LEN/${INPUT_WIDTH}/g;s/SCHEDULE/${ORD}/g;s/REPEAT/${REPEAT}/g" benchmark_1d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so >> ./result/bench1d_result_${ORD}.txt

  exec 5< ./result/bench1d_result_${ORD}.txt
  # empty file.
  >./result/latex/1D/${ORD}_d.txt
  >./result/latex/1D/${ORD}_c.txt
  read line <&5 # read order
  while read sparsity <&5 ; do
        read d <&5
        read c <&5
        echo "${sparsity} ${d}" >> ./result/latex/1D/${ORD}_d.txt
        echo "${sparsity} ${c}" >> ./result/latex/1D/${ORD}_c.txt
  done
done
