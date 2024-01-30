MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static

export TENSOR0="./ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group1.smtx.tns"


echo "DDDS"
for ((c=998; c<=999; c++)); do
sed "s/FORMAT/DDDS/g;s/SPARSITY/${c}/g" ./hyper_sparse_resnet.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier=$OPT | $MLIR_PATH/bin/mlir-cpu-runner -O1 -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so,$MLIR_PATH/lib/libmlir_async_runtime.so
done

echo "DSSS"
for ((c=999; c<=999; c++)); do
sed "s/FORMAT/DSSS/g;s/SPARSITY/${c}/g" ./hyper_sparse_resnet.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier=$OPT | $MLIR_PATH/bin/mlir-cpu-runner -O1 -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so,$MLIR_PATH/lib/libmlir_async_runtime.so
done


# echo "COO"
# for ((c=999; c<=999; c++)); do
# sed "s/FORMAT/COO/g;s/SPARSITY/${c}/g" ./hyper_sparse_resnet.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier=$OPT | $MLIR_PATH/bin/mlir-cpu-runner -O1 -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so,$MLIR_PATH/lib/libmlir_async_runtime.so
# done
