// DEFINE: %{compile} = mlir-opt %s --sparsifier
// DEFINE: %{env} = \
// DEFINE: TENSOR0="%mlir_src_dir/test/Integration/data/ResNet50/0.8/tns/bottleneck_projection_block_group_projection_block_group1.smtx.tns" \
//
// RUN: %{compile} | env %{env} %{run}

!Filename = !llvm.ptr

#DD = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
}>

#COO = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : compressed(nonunique), d1 : singleton(nonunique), d2 : singleton(nonunique), d3 : singleton)
}>

#DDDS = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : dense, d1 : dense, d2 : dense, d3 : compressed)
}>

#DSSS = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : dense, d1 : compressed, d2 : compressed, d3 : compressed)
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
      %c1000 = arith.constant 1000 : index
      %hash = arith.remui %m3, %c1000 : index
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

  func.func @conv_input_sparse(%arg0: tensor<?x?x?x?xf32, #FORMAT>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>, %str : index) -> tensor<?x?x?x?xf32> {
    %c1 = arith.constant 1 : index
    %is_one = arith.cmpi eq, %str, %c1 : index
    %ret = scf.if %is_one -> tensor<?x?x?x?xf32> {
      %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                          strides = dense<1> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32, #FORMAT>, tensor<?x?x?x?xf32>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      scf.yield %result : tensor<?x?x?x?xf32>
    } else {
      %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                          strides = dense<2> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32, #FORMAT>, tensor<?x?x?x?xf32>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      scf.yield %result : tensor<?x?x?x?xf32>
    }
    return %ret : tensor<?x?x?x?xf32>
  }


 func.func @runBenchmark(
  %benchmark : index, %N : index, %H : index, %W : index,
  %R : index, %S : index, %STR : index, %PAD : index, %C : index, %M : index) {
    // vector.print %benchmark : index
    // Compute output shape
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %Pad2 = arith.constant 0 : index
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

    // Construct input.
    %input_sparsity = arith.constant SPARSITY : index // 99.9%
    %input = call @get_sparse_4d_tensor(%N, %H, %W, %C, %input_sparsity) :(index, index, index, index, index) -> (tensor<?x?x?x?xf32>)
    %sparse_input = sparse_tensor.convert %input: tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32, #FORMAT>

    // Construct output.
    %output_elem = arith.constant 0.0 : f32
    %output = call @alloc_4d_filled_f32(%N, %P, %Q, %M, %output_elem) :(index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)

    // Warmup
    %tmp = func.call @conv_input_sparse(%sparse_input, %reshaped_filter, %output, %STR) : (tensor<?x?x?x?xf32, #FORMAT>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index) -> (tensor<?x?x?x?xf32>)

    // Run sparse conv
    %start = func.call @rtclock() : () -> f64
    %ret = func.call @conv_input_sparse(%sparse_input, %reshaped_filter, %output, %STR) : (tensor<?x?x?x?xf32, #FORMAT>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index) -> (tensor<?x?x?x?xf32>)
    %end = func.call @rtclock() : () -> f64
    %time = arith.subf %end, %start : f64
    vector.print %time : f64

    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c112 = arith.constant 112 : index
    %c256 = arith.constant 256 : index

    call @runBenchmark(%c0, %c1, %c256, %c256, %c1, %c1, %c2, %c0, %c64, %c256) :
      (index, index, index, index, index, index, index, index , index, index) -> ()
    return
  }
}
