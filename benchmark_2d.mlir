// DEFINE: %{compile} = mlir-opt %s --sparse-compiler
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
//
// RUN: %{compile} | %{run}

!Filename = !llvm.ptr

#DC = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#CC = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @rtclock() -> (f64)

  func.func @alloc_2d_filled_f32(%s1 : index, %f : f32) -> tensor<?x?xf32> {
    %buf = bufferization.alloc_tensor(%s1, %s1) : tensor<?x?xf32>
    %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %ret : tensor<?x?xf32>
  }

  func.func @get_sparse_2d_tensor(%s1 : index, %sparsity : index) -> tensor<?x?xf32> {
    %tnsr = tensor.generate %s1, %s1 {
    ^bb0(%i : index, %j: index):
      %prime1 = arith.constant 73856093 : index
      %ij = arith.muli %i, %j : index
      %ii = arith.muli %ij, %prime1 : index
      %c100 = arith.constant 100 : index
      %hash = arith.remui %ii, %c100 : index
      %b = arith.cmpi uge, %hash, %sparsity : index
      %f1 = arith.constant 1.0 : f32
      %f0 = arith.constant 0.0 : f32
      %insert = scf.if %b -> f32 {
        scf.yield %f1 : f32
      }  else {
        scf.yield %f0 : f32
      }
      tensor.yield %insert : f32
    } : tensor<?x?xf32>
    return %tnsr : tensor<?x?xf32>
  }

  func.func @conv_2d_DC_dense(%arg0: tensor<?x?xf32, #DC>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %ret = linalg.conv_2d ins (%arg0, %arg1: tensor<?x?xf32, #DC>, tensor<?x?xf32>)
                          outs (%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
    return %ret : tensor<?x?xf32>
  }

  func.func @conv_2d_CC_dense(%arg0: tensor<?x?xf32, #CC>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %ret = linalg.conv_2d ins (%arg0, %arg1: tensor<?x?xf32, #CC>, tensor<?x?xf32>)
                          outs (%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
    return %ret : tensor<?x?xf32>
  }

  func.func @conv_2d_dense_dense(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %ret = linalg.conv_2d ins (%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                          outs (%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
    return %ret : tensor<?x?xf32>
  }

  func.func @runBenchmark(%IL : index, %FL: index) {
    // vector.print %benchmark : index
    // Compute output shape
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %f0 = arith.constant 0.0 : f64
    %f5 = arith.constant 5.0 : f64
    %c101 = arith.constant 101 : index
    %tmp = arith.subi %IL, %FL : index
    %OL = arith.addi %tmp, %c1 : index

    %filter_sparsity = arith.constant 0 : index
    %filter = func.call @get_sparse_2d_tensor(%FL, %filter_sparsity) :(index, index) -> (tensor<?x?xf32>)
    %output_elem = arith.constant 0.0 : f32


    // Run sparse conv
    scf.for %input_sparsity = %c0 to %c101 step %c1 {
      // Construct input.
      %dense_input = func.call @get_sparse_2d_tensor(%IL, %input_sparsity) :(index, index) -> (tensor<?x?xf32>)
      %CC_input = sparse_tensor.convert %dense_input: tensor<?x?xf32> to tensor<?x?xf32, #CC>
      %DC_input = sparse_tensor.convert %dense_input: tensor<?x?xf32> to tensor<?x?xf32, #DC>

      // Run sparse conv
      %dense_time, %CC_time, %DC_time = scf.for %iv = %c0 to %c5 step %c1
        iter_args(%dense_sum = %f0, %CC_sum = %f0, %DC_sum =  %f0) -> (f64, f64, f64) {

        %dense_output = func.call @alloc_2d_filled_f32(%OL, %output_elem) :(index, f32) -> (tensor<?x?xf32>)
        %dense_start = func.call @rtclock() : () -> f64
        %dense_ret = func.call @conv_2d_dense_dense(%dense_input, %filter, %dense_output)
               : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
        %dense_end = func.call @rtclock() : () -> f64
        bufferization.dealloc_tensor %dense_ret : tensor<?x?xf32>
        %dense_time = arith.subf %dense_end, %dense_start : f64
        %dense_next = arith.addf %dense_sum, %dense_time : f64


        %CC_output = func.call @alloc_2d_filled_f32(%OL, %output_elem) :(index, f32) -> (tensor<?x?xf32>)
        %CC_start = func.call @rtclock() : () -> f64
        %CC_ret = func.call @conv_2d_CC_dense(%CC_input, %filter, %CC_output)
               : (tensor<?x?xf32, #CC>, tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
        %CC_end = func.call @rtclock() : () -> f64
        bufferization.dealloc_tensor %CC_ret : tensor<?x?xf32>
        %CC_time = arith.subf %CC_end, %CC_start : f64
        %CC_next = arith.addf %CC_sum, %CC_time : f64

        %DC_output = func.call @alloc_2d_filled_f32(%OL, %output_elem) :(index, f32) -> (tensor<?x?xf32>)
        %DC_start = func.call @rtclock() : () -> f64
        %DC_ret = func.call @conv_2d_DC_dense(%DC_input, %filter, %DC_output)
               : (tensor<?x?xf32, #DC>, tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
        %DC_end = func.call @rtclock() : () -> f64
        bufferization.dealloc_tensor %DC_ret : tensor<?x?xf32>
        %DC_time = arith.subf %DC_end, %DC_start : f64
        %DC_next = arith.addf %DC_sum, %DC_time : f64

        scf.yield %dense_next, %CC_next, %DC_next : f64, f64, f64
      }

      bufferization.dealloc_tensor %dense_input : tensor<?x?xf32>
      bufferization.dealloc_tensor %CC_input : tensor<?x?xf32, #CC>
      bufferization.dealloc_tensor %DC_input : tensor<?x?xf32, #DC>
      vector.print %input_sparsity : index // also the sparsity level

      %dense_average_time = arith.divf %dense_time, %f5 : f64
      vector.print %dense_average_time : f64
      %CC_average_time = arith.divf %CC_time, %f5 : f64
      vector.print %CC_average_time : f64
      %DC_average_time = arith.divf %DC_time, %f5 : f64
      vector.print %DC_average_time : f64

    }
    return
  }

  func.func @entry() {
    %c1024 = arith.constant 400 : index
    %c3 = arith.constant 3 : index
    call @runBenchmark(%c1024, %c3) :  (index, index) -> ()
    return
  }
}
