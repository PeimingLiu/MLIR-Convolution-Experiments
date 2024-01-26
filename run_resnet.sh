MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-static

export TENSOR0="./ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group1.smtx.tns" \
       TENSOR1="./ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group1.smtx.tns" \
       TENSOR2="./ResNet50/0.8/tns/bottleneck_1_block_group1_1_1.smtx.tns" \
       TENSOR3="./ResNet50/0.8/tns/bottleneck_2_block_group1_1_1.smtx.tns" \
       TENSOR4="./ResNet50/0.8/tns/bottleneck_3_block_group1_1_1.smtx.tns" \
       TENSOR5="./ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group2.smtx.tns" \
       TENSOR6="./ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group2.smtx.tns" \
       TENSOR7="./ResNet50/0.8/tns/bottleneck_2_block_group_projection_block_group2.smtx.tns" \
       TENSOR8="./ResNet50/0.8/tns/bottleneck_1_block_group2_1_1.smtx.tns" \
       TENSOR9="./ResNet50/0.8/tns/bottleneck_2_block_group2_1_1.smtx.tns" \
       TENSOR10="./ResNet50/0.8/tns/bottleneck_3_block_group2_1_1.smtx.tns" \
       TENSOR11="./ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group3.smtx.tns" \
       TENSOR12="./ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group3.smtx.tns" \
       TENSOR13="./ResNet50/0.8/tns/bottleneck_2_block_group_projection_block_group3.smtx.tns" \
       TENSOR14="./ResNet50/0.8/tns/bottleneck_3_block_group_projection_block_group3.smtx.tns" \
       TENSOR15="./ResNet50/0.8/tns/bottleneck_1_block_group3_1_1.smtx.tns" \
       TENSOR16="./ResNet50/0.8/tns/bottleneck_2_block_group3_1_1.smtx.tns" \
       TENSOR17="./ResNet50/0.8/tns/bottleneck_3_block_group3_1_1.smtx.tns" \
       TENSOR18="./ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group4.smtx.tns" \
       TENSOR19="./ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group4.smtx.tns" \
       TENSOR20="./ResNet50/0.8/tns/bottleneck_2_block_group_projection_block_group4.smtx.tns" \
       TENSOR21="./ResNet50/0.8/tns/bottleneck_3_block_group_projection_block_group4.smtx.tns" \
       TENSOR22="./ResNet50/0.8/tns/bottleneck_1_block_group4_1_1.smtx.tns" \
       TENSOR23="./ResNet50/0.8/tns/bottleneck_2_block_group4_1_1.smtx.tns" \
       TENSOR24="./ResNet50/0.8/tns/bottleneck_3_block_group4_1_1.smtx.tns"

#$MLIR_PATH/bin/mlir-opt --sparsifier="parallelization-strategy=any-storage-any-loop" ./input_sparse_benchmark.mlir | mlir-cpu-runner -O0 -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so,$MLIR_PATH/lib/libmlir_async_runtime.so
$MLIR_PATH/bin/mlir-opt --sparsifier="parallelization-strategy=none" ./input_sparse_benchmark.mlir | $MLIR_PATH/bin/mlir-cpu-runner -O0 -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so
