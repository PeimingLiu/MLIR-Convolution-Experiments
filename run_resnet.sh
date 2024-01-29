MLIR_PATH=/usr/local/google/home/peiming/projects/llvm-project/build-openmp

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

#%benchmark , %N , %H , %W , %R , %S , %STRIDE , %PAD , %C , %M
CONFIG=(
         "s/BEN/0/g;s/N_VAL/1/g;s/H_VAL/112/g;s/W_VAL/112/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/2/g;s/C_VAL/64/g;s/M_VAL/256/g"
         "s/BEN/1/g;s/N_VAL/1/g;s/H_VAL/56/g;s/W_VAL/56/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/64/g;s/M_VAL/64/g"
         "s/BEN/2/g;s/N_VAL/1/g;s/H_VAL/56/g;s/W_VAL/56/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/256/g;s/M_VAL/64/g"
         "s/BEN/3/g;s/N_VAL/1/g;s/H_VAL/56/g;s/W_VAL/56/g;s/R_VAL/3/g;s/S_VAL/3/g;s/STRIDE/1/g;s/C_VAL/64/g;s/M_VAL/64/g"
         "s/BEN/4/g;s/N_VAL/1/g;s/H_VAL/56/g;s/W_VAL/56/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/64/g;s/M_VAL/256/g"
         "s/BEN/5/g;s/N_VAL/1/g;s/H_VAL/56/g;s/W_VAL/56/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/2/g;s/C_VAL/256/g;s/M_VAL/512/g"
         "s/BEN/6/g;s/N_VAL/1/g;s/H_VAL/56/g;s/W_VAL/56/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/256/g;s/M_VAL/128/g"
         "s/BEN/7/g;s/N_VAL/1/g;s/H_VAL/56/g;s/W_VAL/56/g;s/R_VAL/3/g;s/S_VAL/3/g;s/STRIDE/2/g;s/C_VAL/128/g;s/M_VAL/128/g"
         "s/BEN/8/g;s/N_VAL/1/g;s/H_VAL/28/g;s/W_VAL/28/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/512/g;s/M_VAL/128/g"
         "s/BEN/9/g;s/N_VAL/1/g;s/H_VAL/28/g;s/W_VAL/28/g;s/R_VAL/3/g;s/S_VAL/3/g;s/STRIDE/1/g;s/C_VAL/128/g;s/M_VAL/128/g"
         "s/BEN/10/g;s/N_VAL/1/g;s/H_VAL/28/g;s/W_VAL/28/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/128/g;s/M_VAL/512/g"
         "s/BEN/11/g;s/N_VAL/1/g;s/H_VAL/28/g;s/W_VAL/28/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/2/g;s/C_VAL/512/g;s/M_VAL/1024/g"
         "s/BEN/12/g;s/N_VAL/1/g;s/H_VAL/28/g;s/W_VAL/28/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/512/g;s/M_VAL/256/g"
         "s/BEN/13/g;s/N_VAL/1/g;s/H_VAL/28/g;s/W_VAL/28/g;s/R_VAL/3/g;s/S_VAL/3/g;s/STRIDE/2/g;s/C_VAL/256/g;s/M_VAL/256/g"
         "s/BEN/14/g;s/N_VAL/1/g;s/H_VAL/14/g;s/W_VAL/14/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/256/g;s/M_VAL/1024/g"
         "s/BEN/15/g;s/N_VAL/1/g;s/H_VAL/14/g;s/W_VAL/14/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/1024/g;s/M_VAL/256/g"
         "s/BEN/16/g;s/N_VAL/1/g;s/H_VAL/14/g;s/W_VAL/14/g;s/R_VAL/3/g;s/S_VAL/3/g;s/STRIDE/1/g;s/C_VAL/256/g;s/M_VAL/256/g"
         "s/BEN/17/g;s/N_VAL/1/g;s/H_VAL/14/g;s/W_VAL/14/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/256/g;s/M_VAL/1024/g"
         "s/BEN/18/g;s/N_VAL/1/g;s/H_VAL/14/g;s/W_VAL/14/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/2/g;s/C_VAL/1024/g;s/M_VAL/2048/g"
         "s/BEN/19/g;s/N_VAL/1/g;s/H_VAL/14/g;s/W_VAL/14/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/1024/g;s/M_VAL/512/g"
         "s/BEN/20/g;s/N_VAL/1/g;s/H_VAL/14/g;s/W_VAL/14/g;s/R_VAL/3/g;s/S_VAL/3/g;s/STRIDE/2/g;s/C_VAL/512/g;s/M_VAL/512/g"
         "s/BEN/21/g;s/N_VAL/1/g;s/H_VAL/7/g;s/W_VAL/7/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/512/g;s/M_VAL/2048/g"
         "s/BEN/22/g;s/N_VAL/1/g;s/H_VAL/7/g;s/W_VAL/7/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/2048/g;s/M_VAL/512/g"
         "s/BEN/23/g;s/N_VAL/1/g;s/H_VAL/7/g;s/W_VAL/7/g;s/R_VAL/3/g;s/S_VAL/3/g;s/STRIDE/1/g;s/C_VAL/512/g;s/M_VAL/512/g"
         "s/BEN/24/g;s/N_VAL/1/g;s/H_VAL/7/g;s/W_VAL/7/g;s/R_VAL/1/g;s/S_VAL/1/g;s/STRIDE/1/g;s/C_VAL/512/g;s/M_VAL/2048/g"
       )

echo "multi"
for cfg in ${CONFIG[*]}; do

  sed $cfg ./input_sparse_benchmark.mlir | $MLIR_PATH/bin/mlir-opt --canonicalize | $MLIR_PATH/bin/mlir-opt --sparsifier="parallelization-strategy=any-storage-any-loop" | mlir-cpu-runner -O3 -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so,$MLIR_PATH/lib/libmlir_async_runtime.so,$MLIR_PATH/lib/libomp.so

done

echo "single"
for cfg in ${CONFIG[*]}; do

  sed $cfg ./input_sparse_benchmark.mlir | $MLIR_PATH/bin/mlir-opt --canonicalize | $MLIR_PATH/bin/mlir-opt --sparsifier="parallelization-strategy=none" | $MLIR_PATH/bin/mlir-cpu-runner  -O3 -e entry -entry-point-result=void -shared-libs=$MLIR_PATH/lib/libmlir_c_runner_utils.so,$MLIR_PATH/lib/libmlir_runner_utils.so

done
