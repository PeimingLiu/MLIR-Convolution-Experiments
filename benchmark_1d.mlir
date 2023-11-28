// DEFINE: %{compile} = mlir-opt %s --sparse-compiler
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
//
// RUN: %{compile} | %{run}

!Filename = !llvm.ptr

#INPUT = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @rtclock() -> (f64)

  func.func @alloc_1d_filled_f32(%s1 : index, %f : f32) -> tensor<?xf32> {
    %buf = bufferization.alloc_tensor(%s1) : tensor<?xf32>
    %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?xf32>) -> tensor<?xf32>
    return %ret : tensor<?xf32>
  }

  func.func @get_sparse_1d_tensor(%s1 : index, %sparsity : index) -> tensor<?xf32> {
    %tnsr = tensor.generate %s1 {
    ^bb0(%i : index):
      %prime1 = arith.constant 73856093 : index
      %ii = arith.muli %i, %prime1 : index
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
    } : tensor<?xf32>
    return %tnsr : tensor<?xf32>
  }

  func.func @conv_1d_sparse_dense(%arg0: tensor<?xf32, #INPUT>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %ret = linalg.conv_1d ins (%arg0, %arg1: tensor<?xf32, #INPUT>, tensor<?xf32>)
                          outs (%arg2: tensor<?xf32>) -> tensor<?xf32>
    return %ret : tensor<?xf32>
  }

  func.func @conv_1d_dense_dense(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %ret = linalg.conv_1d ins (%arg0, %arg1: tensor<?xf32>, tensor<?xf32>)
                          outs (%arg2: tensor<?xf32>) -> tensor<?xf32>
    return %ret : tensor<?xf32>
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
    %c100 = arith.constant 100 : index
    %tmp = arith.subi %IL, %FL : index
    %OL = arith.addi %tmp, %c1 : index

    %filter_sparsity = arith.constant 0 : index
    %filter = func.call @get_sparse_1d_tensor(%FL, %filter_sparsity) :(index, index) -> (tensor<?xf32>)
    %output_elem = arith.constant 0.0 : f32


    // Run sparse conv
    scf.for %input_sparsity = %c0 to %c100 step %c1 {
      // Construct input.
      %dense_input = func.call @get_sparse_1d_tensor(%IL, %input_sparsity) :(index, index) -> (tensor<?xf32>)
      %sparse_input = sparse_tensor.convert %dense_input: tensor<?xf32> to tensor<?xf32, #INPUT>

      // Run sparse conv
      %dense_time, %sparse_time = scf.for %iv = %c0 to %c5 step %c1
        iter_args(%dense_sum = %f0, %sparse_sum = %f0) -> (f64, f64) {
        %dense_output = func.call @alloc_1d_filled_f32(%OL, %output_elem) :(index, f32) -> (tensor<?xf32>)
        %dense_start = func.call @rtclock() : () -> f64
        %dense_ret = func.call @conv_1d_dense_dense(%dense_input, %filter, %dense_output)
               : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
        %dense_end = func.call @rtclock() : () -> f64
        bufferization.dealloc_tensor %dense_ret : tensor<?xf32>

        %sparse_output = func.call @alloc_1d_filled_f32(%OL, %output_elem) :(index, f32) -> (tensor<?xf32>)
        %sparse_start = func.call @rtclock() : () -> f64
        %sparse_ret = func.call @conv_1d_sparse_dense(%sparse_input, %filter, %sparse_output)
               : (tensor<?xf32, #INPUT>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
        %sparse_end = func.call @rtclock() : () -> f64
        bufferization.dealloc_tensor %sparse_ret : tensor<?xf32>

        %dense_time = arith.subf %dense_end, %dense_start : f64
        %dense_next = arith.addf %dense_sum, %dense_time : f64

        %sparse_time = arith.subf %sparse_end, %sparse_start : f64
        %sparse_next = arith.addf %sparse_sum, %sparse_time : f64

        scf.yield %dense_next, %sparse_next : f64, f64
      }

      bufferization.dealloc_tensor %dense_input : tensor<?xf32>
      bufferization.dealloc_tensor %sparse_input : tensor<?xf32, #INPUT>

      %dense_average_time = arith.divf %dense_time, %f5 : f64
      %sparse_average_time = arith.divf %sparse_time, %f5 : f64

      vector.print %iv : index
      vector.print %dense_average_time : f64
      vector.print %sparse_average_time : f64

    }
    return
  }

  func.func @entry() {
    %c1024 = arith.constant 4096 : index
    %c3 = arith.constant 3 : index
    call @runBenchmark(%c1024, %c3) :  (index, index) -> ()
    return
  }
}
