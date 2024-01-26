MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static

export TENSOR0="./ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group1.smtx.tns"

echo "DDDS"
sed "s/FORMAT/DDDS/g" ./hyper_sparse_resnet.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier | $MLIR_PATH/bin/mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so

echo "DSSS"
sed "s/FORMAT/DSSS/g" ./hyper_sparse_resnet.mlir | $MLIR_PATH/bin/mlir-opt --sparsifier | $MLIR_PATH/bin/mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so
