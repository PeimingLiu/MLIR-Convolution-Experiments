MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
INPUT_WIDTH=5
INPUT_HEIGHT=5
INPUT_DEPTH=5
REPEAT=3  # At least three

ORDERS=(ppprrr prprrp rrrppp prprpr prpprr prrprp)

for ORD in ${ORDERS[*]};
do
  echo $ORD > ./result/bench3d_result_${ORD}.txt
  sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/DEPTH/${INPUT_DEPTH}/g;s/SCHEDULE/${ORD}/g;s/REPEAT/${REPEAT}/g" benchmark_3d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so >> ./result/bench3d_result_${ORD}.txt

  exec 5< ./result/bench3d_result_${ORD}.txt

  >./result/latex/3D/${ORD}_ddd.txt
  >./result/latex/3D/${ORD}_ccc.txt
  >./result/latex/3D/${ORD}_dcc.txt
  >./result/latex/3D/${ORD}_ddc.txt
  read line <&5 # read order
  while read sparsity <&5 ; do
        read ddd <&5
        read ccc <&5
        read dcc <&5
        read ddc <&5
        echo "${sparsity} ${ddd}" >> ./result/latex/3D/${ORD}_ddd.txt
        echo "${sparsity} ${ccc}" >> ./result/latex/3D/${ORD}_ccc.txt
        echo "${sparsity} ${dcc}" >> ./result/latex/3D/${ORD}_dcc.txt
        echo "${sparsity} ${ddc}" >> ./result/latex/3D/${ORD}_ddc.txt
  done
done
