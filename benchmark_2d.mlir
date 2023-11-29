// DEFINE: %{compile} = mlir-opt %s --sparse-compiler
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
//
// RUN: %{compile} | %{run}

!Filename = !llvm.ptr
!Generator = !llvm.ptr

#DD = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : dense) }>
#DC = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#CC = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>


module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @rtclock() -> (f64)

  func.func private @rtsrand(index) -> (!Generator)
  func.func private @rtrand(!Generator, index) -> (index)
  func.func private @rtdrand(!Generator) -> ()

  func.func @alloc_2d_filled_f32(%w : index, %h : index, %f : f32) -> tensor<?x?xf32> {
    %buf = bufferization.alloc_tensor(%w, %h) : tensor<?x?xf32>
    %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %ret : tensor<?x?xf32>
  }

  func.func private @printMemref2dF32(%ptr : tensor<?x?xf32>) attributes { llvm.emit_c_interface }
  func.func @dump(%arg0: tensor<?x?xf32>) {
    call @printMemref2dF32(%arg0) : (tensor<?x?xf32>) -> ()
    return
  }

  func.func @get_sparse_2d_tensor(%w : index, %h : index, %sparsity : index, %g : !Generator) -> tensor<?x?xf32> {
    %tnsr = tensor.generate %w, %h {
    ^bb0(%i : index, %j: index):
      %c99 = arith.constant 99 : index
      %ri = func.call @rtrand(%g, %c99) : (!Generator, index) -> (index)
      %b = arith.cmpi uge, %ri, %sparsity : index
      %r = arith.index_cast %ri : index to i64
      %f1 = arith.constant 1.0 : f32
      %f0 = arith.constant 0.0 : f32
      %insert = scf.if %b -> f32 {
        scf.yield %f1 : f32
      }  else {
        scf.yield %f0 : f32
      }
      tensor.yield %insert : f32
    } : tensor<?x?xf32>

//    func.call @dump(%tnsr) : (tensor<?x?xf32>) -> ()
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

  func.func @runBenchmark(%IW : index, %IH : index, %FL: index) {
    // Compute output shape
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c49 = arith.constant 49 : index
    %c50 = arith.constant 50 : index
    %f0 = arith.constant 0.0 : f64
    %f5 = arith.constant 5.0 : f64
    %c101 = arith.constant 101 : index
    %tmpW = arith.subi %IW, %FL : index
    %tmpH = arith.subi %IH, %FL : index
    %OW = arith.addi %tmpW, %c1 : index
    %OH = arith.addi %tmpH, %c1 : index

    %g = func.call @rtsrand(%c0) : (index) ->(!Generator)

    %filter_sparsity = arith.constant 0 : index
    %filter = func.call @get_sparse_2d_tensor(%FL, %FL, %filter_sparsity, %g) :(index, index, index, !Generator) -> (tensor<?x?xf32>)
    %output_elem = arith.constant 0.0 : f32


    // Run sparse conv
    scf.for %input_sparsity = %c49 to %c50 step %c1 {
      // Construct input.
      %dense_input = func.call @get_sparse_2d_tensor(%IW, %IH, %input_sparsity, %g) :(index, index, index, !Generator) -> (tensor<?x?xf32>)
      %CC_input = sparse_tensor.convert %dense_input: tensor<?x?xf32> to tensor<?x?xf32, #CC>
      // %nnz = sparse_tensor.number_of_entries %CC_input : tensor<?x?xf32, #CC>
      // vector.print %nnz : index
      %DC_input = sparse_tensor.convert %dense_input: tensor<?x?xf32> to tensor<?x?xf32, #DC>

      // Run sparse conv
      %dense_time, %CC_time, %DC_time = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%dense_sum = %f0, %CC_sum = %f0, %DC_sum =  %f0) -> (f64, f64, f64) {

        %dense_output = func.call @alloc_2d_filled_f32(%OW, %OH, %output_elem) :(index, index, f32) -> (tensor<?x?xf32>)
        %dense_start = func.call @rtclock() : () -> f64
        %dense_ret = func.call @conv_2d_dense_dense(%dense_input, %filter, %dense_output)
               : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
        %dense_end = func.call @rtclock() : () -> f64
//        func.call @dump(%dense_ret) : (tensor<?x?xf32>) -> ()
        bufferization.dealloc_tensor %dense_ret : tensor<?x?xf32>
        %dense_time = arith.subf %dense_end, %dense_start : f64
        %dense_next = arith.addf %dense_sum, %dense_time : f64


        %CC_output = func.call @alloc_2d_filled_f32(%OW, %OH, %output_elem) :(index, index, f32) -> (tensor<?x?xf32>)
        %CC_start = func.call @rtclock() : () -> f64
        %CC_ret = func.call @conv_2d_CC_dense(%CC_input, %filter, %CC_output)
               : (tensor<?x?xf32, #CC>, tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
        %CC_end = func.call @rtclock() : () -> f64
//        func.call @dump(%CC_ret) : (tensor<?x?xf32>) -> ()
        bufferization.dealloc_tensor %CC_ret : tensor<?x?xf32>
        %CC_time = arith.subf %CC_end, %CC_start : f64
        %CC_next = arith.addf %CC_sum, %CC_time : f64

        %DC_output = func.call @alloc_2d_filled_f32(%OW, %OH, %output_elem) :(index, index, f32) -> (tensor<?x?xf32>)
        %DC_start = func.call @rtclock() : () -> f64
        %DC_ret = func.call @conv_2d_DC_dense(%DC_input, %filter, %DC_output)
               : (tensor<?x?xf32, #DC>, tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
        %DC_end = func.call @rtclock() : () -> f64
//        func.call @dump(%DC_ret) : (tensor<?x?xf32>) -> ()
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

    func.call @rtdrand(%g) : (!Generator) ->()
    return
  }

  func.func @entry() {
    %w = arith.constant WIDTH : index
    %h = arith.constant HEIGHT : index
    %c3 = arith.constant 3 : index
    call @runBenchmark(%w, %h, %c3) :  (index, index, index) -> ()
    return
  }
}
