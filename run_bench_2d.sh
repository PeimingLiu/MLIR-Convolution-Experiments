MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static
INPUT_WIDTH=999
INPUT_HEIGHT=999
REPEAT=10

ORDERS=(prpr prrp pprr rrpp)

for ORD in ${ORDERS[*]};
do
  echo $ORD > ./result/bench2d_result_${ORD}.txt
  sed "s/WIDTH/${INPUT_WIDTH}/g;s/HEIGHT/${INPUT_HEIGHT}/g;s/SCHEDULE/${ORD}/g;s/REPEAT/${REPEAT}/g" benchmark_2d.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier="enable-runtime-library=true" | mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so >> ./result/bench2d_result_${ORD}.txt

  exec 5< ./result/bench2d_result_${ORD}.txt

  >./result/latex/2D/${ORD}_dd.txt
  >./result/latex/2D/${ORD}_cc.txt
  >./result/latex/2D/${ORD}_dc.txt
  read line <&5 # read order
  while read sparsity <&5 ; do
        read dd <&5
        read cc <&5
        read dc <&5
        echo "${sparsity} ${dd}" >> ./result/latex/2D/${ORD}_dd.txt
        echo "${sparsity} ${cc}" >> ./result/latex/2D/${ORD}_cc.txt
        echo "${sparsity} ${dc}" >> ./result/latex/2D/${ORD}_dc.txt
  done
done
