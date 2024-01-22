// DEFINE: %{compile} = mlir-opt %s --sparsifier
// DEFINE: %{env} = \
// DEFINE: TENSOR0="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group1.smtx.tns" \
// DEFINE: TENSOR1="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group1.smtx.tns" \
// DEFINE: TENSOR2="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group1_1_1.smtx.tns" \
// DEFINE: TENSOR3="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_2_block_group1_1_1.smtx.tns" \
// DEFINE: TENSOR4="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_3_block_group1_1_1.smtx.tns" \
// DEFINE: TENSOR5="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group2.smtx.tns" \
// DEFINE: TENSOR6="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group2.smtx.tns" \
// DEFINE: TENSOR7="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_2_block_group_projection_block_group2.smtx.tns" \
// DEFINE: TENSOR8="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group2_1_1.smtx.tns" \
// DEFINE: TENSOR9="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_2_block_group2_1_1.smtx.tns" \
// DEFINE: TENSOR10="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_3_block_group2_1_1.smtx.tns" \
// DEFINE: TENSOR11="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group3.smtx.tns" \
// DEFINE: TENSOR12="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group3.smtx.tns" \
// DEFINE: TENSOR13="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_2_block_group_projection_block_group3.smtx.tns" \
// DEFINE: TENSOR14="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_3_block_group_projection_block_group3.smtx.tns" \
// DEFINE: TENSOR15="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group3_1_1.smtx.tns" \
// DEFINE: TENSOR16="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_2_block_group3_1_1.smtx.tns" \
// DEFINE: TENSOR17="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_3_block_group3_1_1.smtx.tns" \
// DEFINE: TENSOR18="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group4.smtx.tns" \
// DEFINE: TENSOR19="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group_projection_block_group4.smtx.tns" \
// DEFINE: TENSOR20="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_2_block_group_projection_block_group4.smtx.tns" \
// DEFINE: TENSOR21="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_3_block_group_projection_block_group4.smtx.tns" \
// DEFINE: TENSOR22="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_1_block_group4_1_1.smtx.tns" \
// DEFINE: TENSOR23="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_2_block_group4_1_1.smtx.tns" \
// DEFINE: TENSOR24="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_3_block_group4_1_1.smtx.tns"
// DEFINE: %{run} = \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
//
// RUN: %{compile} | env %{env} %{run}

!Filename = !llvm.ptr

#DD = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
}>

#DDDS = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : dense, d1 : dense, d2 : dense, d3 : compressed)
}>

#SSSS = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : compressed)
}>

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @rtclock() -> (f64)

  func.func private @printMemref1dF32(%ptr : memref<?xf32>) attributes { llvm.emit_c_interface }

  //
  // Helper method to print values array. The transfer actually
  // reads more than required to verify size of buffer as well.
  //
  func.func @dump(%arg0: memref<?xf32>) {
    call @printMemref1dF32(%arg0) : (memref<?xf32>) -> ()
    return
  }

  func.func @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> tensor<?x?x?x?xf32> {
    %buf = bufferization.alloc_tensor(%s1, %s2, %s3, %s4) : tensor<?x?x?x?xf32>
    %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %ret : tensor<?x?x?x?xf32>
  }

  func.func @get_sparse_4d_tensor(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %sparsity : index) -> tensor<?x?x?x?xf32> {
    %tnsr = tensor.generate %s1, %s2, %s3, %s4 {
    ^bb0(%i : index, %j : index, %k : index, %l : index):
      %prime1 = arith.constant 73856093 : index
      %prime2 = arith.constant 19349663 : index
      %prime3 = arith.constant 83492791 : index
      %prime4 = arith.constant 49979687 : index
      %ii = arith.muli %i, %prime1 : index
      %jj = arith.muli %j, %prime2 : index
      %kk = arith.muli %k, %prime3 : index
      %ll = arith.muli %l, %prime4 : index
      %m1 = arith.addi %ii, %jj : index
      %m2 = arith.addi %m1, %kk : index
      %m3 = arith.addi %m2, %ll : index
      %c100 = arith.constant 100 : index
      %hash = arith.remui %m3, %c100 : index
      %b = arith.cmpi uge, %hash, %sparsity : index

      %f1 = arith.constant 1.0 : f32
      %f0 = arith.constant 0.0 : f32
      %insert = scf.if %b -> f32 {
        scf.yield %f1 : f32
      }  else {
        scf.yield %f0 : f32
      }
      tensor.yield %insert : f32
    } : tensor<?x?x?x?xf32>
    return %tnsr : tensor<?x?x?x?xf32>
  }

  func.func @conv_2d_dual_sparse(%arg0: tensor<?x?x?x?xf32, #SSSS>, %arg1: tensor<?x?x?x?xf32, #SSSS>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %ret = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                      strides = dense<1> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32, #SSSS>, tensor<?x?x?x?xf32, #SSSS>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %ret : tensor<?x?x?x?xf32>
  }

  func.func @conv_2d_filter_sparse(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32, #SSSS>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %ret = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                      strides = dense<1> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32, #SSSS>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %ret : tensor<?x?x?x?xf32>
  }

  func.func @conv_input_sparse(%arg0: tensor<?x?x?x?xf32, #SSSS>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>, %str : index) -> tensor<?x?x?x?xf32> {
    %c1 = arith.constant 1 : index
    %is_one = arith.cmpi eq, %str, %c1 : index
    %ret = scf.if %is_one -> tensor<?x?x?x?xf32> {
      %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                        strides = dense<1> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32, #SSSS>, tensor<?x?x?x?xf32>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      scf.yield %result : tensor<?x?x?x?xf32>
    } else {
      %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                        strides = dense<2> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32, #SSSS>, tensor<?x?x?x?xf32>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      scf.yield %result : tensor<?x?x?x?xf32>
    }
    return %ret : tensor<?x?x?x?xf32>
  }

  func.func @conv_dense(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>, %str : index) -> tensor<?x?x?x?xf32> {
    %c1 = arith.constant 1 : index
    %is_one = arith.cmpi eq, %str, %c1 : index

    %ret = scf.if %is_one -> tensor<?x?x?x?xf32> {
      %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                        strides = dense<1> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      scf.yield %result : tensor<?x?x?x?xf32>
    } else {
      %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                        strides = dense<2> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      scf.yield %result : tensor<?x?x?x?xf32>
    }
    return %ret : tensor<?x?x?x?xf32>
  }

  func.func @runBenchmark(%benchmark : index, %N : index, %H : index, %W : index, %R : index, %S : index,
  %STR : index, %PAD : index, %C : index, %M : index) {
    // vector.print %benchmark : index
    // Compute output shape
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %Pad2 = arith.muli %PAD, %c2 : index
    %HPad = arith.addi %H, %Pad2 : index
    %WPad = arith.addi %W, %Pad2 : index
    %HPadMinusR = arith.subi %HPad, %R : index
    %HPadMinusRDivStr = arith.divui  %HPadMinusR, %STR : index
    %WPadMinusS = arith.subi %WPad, %S : index
    %WPadMinusSDivStr = arith.divui %WPadMinusS, %STR : index
    %P = arith.addi %HPadMinusRDivStr, %c1 : index
    %Q = arith.addi %WPadMinusSDivStr, %c1: index

    // Construct filter of size RxSxCxM.
    %file_name = call @getTensorFilename(%benchmark) : (index) -> (!Filename)
    %filter = sparse_tensor.new %file_name : !Filename to tensor<?x?xf32, #DD>
    %dense_filter = sparse_tensor.convert %filter : tensor<?x?xf32, #DD> to tensor<?x?xf32>
    %filter_shape = tensor.from_elements %R, %S, %C, %M : tensor<4xindex>
    %reshaped_filter = tensor.reshape %dense_filter(%filter_shape) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
    %sparse_filter = sparse_tensor.convert %reshaped_filter: tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32, #SSSS>

    // %filter_vals = sparse_tensor.values %sparseFilter : tensor<?x?x?x?xf32, #DDDS> to memref<?xf32>
    // call @dump(%filter_vals) : (memref<?xf32>) -> ()

    // Construct input.
    %input_sparsity = arith.constant 0 : index
    %input = call @get_sparse_4d_tensor(%N, %H, %W, %C, %input_sparsity) :(index, index, index, index, index) -> (tensor<?x?x?x?xf32>)
    %sparse_input = sparse_tensor.convert %input: tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32, #SSSS>

    // Construct output.
    %output_elem = arith.constant 0.0 : f32
    %output = call @alloc_4d_filled_f32(%N, %P, %Q, %M, %output_elem) :(index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)

    // Run sparse conv
    %start = func.call @rtclock() : () -> f64
    %ret = func.call @conv_input_sparse(%sparse_input, %reshaped_filter, %output, %STR) : (tensor<?x?x?x?xf32, #SSSS>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index) -> (tensor<?x?x?x?xf32>)
    %end = func.call @rtclock() : () -> f64
    %time = arith.subf %end, %start : f64
    vector.print %time : f64

    bufferization.dealloc_tensor %filter: tensor<?x?xf32, #DD>
    bufferization.dealloc_tensor %filter_shape : tensor<4xindex>
    bufferization.dealloc_tensor %reshaped_filter: tensor<?x?x?x?xf32>
    bufferization.dealloc_tensor %sparse_filter: tensor<?x?x?x?xf32, #SSSS>
    bufferization.dealloc_tensor %input: tensor<?x?x?x?xf32>
    bufferization.dealloc_tensor %sparse_input: tensor<?x?x?x?xf32, #SSSS>
    bufferization.dealloc_tensor %output : tensor<?x?x?x?xf32>
    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    %c17 = arith.constant 17 : index
    %c18 = arith.constant 18 : index
    %c19 = arith.constant 19 : index
    %c20 = arith.constant 20 : index
    %c21 = arith.constant 21 : index
    %c22 = arith.constant 22 : index
    %c23 = arith.constant 23 : index
    %c24 = arith.constant 24 : index
    %c28 = arith.constant 28 : index
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c112 = arith.constant 112 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index

    call @runBenchmark(%c0, %c1, %c112, %c112, %c1, %c1, %c2, %c0, %c64, %c256) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c1, %c1, %c56, %c56, %c1, %c1, %c1, %c0, %c64, %c64) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c2, %c1, %c56, %c56, %c1, %c1, %c1, %c0, %c256, %c64) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c3, %c1, %c56, %c56, %c3, %c3, %c1, %c1, %c64, %c64) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c4, %c1, %c56, %c56, %c1, %c1, %c1, %c0, %c64, %c256) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c5, %c1, %c56, %c56, %c1, %c1, %c2, %c0, %c256, %c512) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c6, %c1, %c56, %c56, %c1, %c1, %c1, %c0, %c256, %c128) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c7, %c1, %c56, %c56, %c3, %c3, %c2, %c1, %c128, %c128) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c8, %c1, %c28, %c28, %c1, %c1, %c1, %c0, %c512, %c128) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c9, %c1, %c28, %c28, %c3, %c3, %c1, %c1, %c128, %c128) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c10, %c1, %c28, %c28, %c1, %c1, %c1, %c0, %c128, %c512) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c11, %c1, %c28, %c28, %c1, %c1, %c2, %c0, %c512, %c1024) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c12, %c1, %c28, %c28, %c1, %c1, %c1, %c0, %c512, %c256) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c13, %c1, %c28, %c28, %c3, %c3, %c2, %c1, %c256, %c256) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c14, %c1, %c14, %c14, %c1, %c1, %c1, %c0, %c256, %c1024) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c15, %c1, %c14, %c14, %c1, %c1, %c1, %c0, %c1024, %c256) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c16, %c1, %c14, %c14, %c3, %c3, %c1, %c1, %c256, %c256) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c17, %c1, %c14, %c14, %c1, %c1, %c1, %c0, %c256, %c1024) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c18, %c1, %c14, %c14, %c1, %c1, %c2, %c0, %c1024, %c2048) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c19, %c1, %c14, %c14, %c1, %c1, %c1, %c0, %c1024, %c512) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c20, %c1, %c14, %c14, %c3, %c3, %c2, %c1, %c512, %c512) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c21, %c1, %c7, %c7, %c1, %c1, %c1, %c0, %c512, %c2048) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c22, %c1, %c7, %c7, %c1, %c1, %c1, %c0, %c2048, %c512) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c23, %c1, %c7, %c7, %c3, %c3, %c1, %c1, %c512, %c512) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    call @runBenchmark(%c24, %c1, %c7, %c7, %c1, %c1, %c1, %c0, %c512, %c2048) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    return
  }
}