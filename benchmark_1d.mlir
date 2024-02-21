// DEFINE: %{compile} = mlir-opt %s --sparse-compiler
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
//
// RUN: %{compile} | %{run}

!Filename = !llvm.ptr
!Generator = !llvm.ptr

#map = affine_map<(d0, d1) -> (d1 + d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>

#pr = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d1 + d0)>,
    affine_map<(d0, d1) -> (d1)>,
    affine_map<(d0, d1) -> (d0)>
  ],
  iterator_types = ["parallel", "reduction"]
}

#rp = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d1 + d0)>,
    affine_map<(d0, d1) -> (d0)>,
    affine_map<(d0, d1) -> (d1)>
  ],
  iterator_types = ["reduction", "parallel"]
}


#INPUT = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @rtclock() -> (f64)

  func.func private @rtsrand(index) -> (!Generator)
  func.func private @rtrand(!Generator, index) -> (index)
  func.func private @rtdrand(!Generator) -> ()

  func.func @alloc_1d_filled_f64(%s1 : index, %f : f64) -> tensor<?xf64> {
    %buf = bufferization.alloc_tensor(%s1) : tensor<?xf64>
    %ret = linalg.fill ins(%f : f64) outs(%buf : tensor<?xf64>) -> tensor<?xf64>
    return %ret : tensor<?xf64>
  }

  func.func @get_sparse_1d_tensor(%w : index, %sparsity : index, %g : !Generator) -> tensor<?xf64> {
    %tnsr = tensor.generate %w {
    ^bb0(%i : index):
      %c99 = arith.constant 99 : index
      %ri = func.call @rtrand(%g, %c99) : (!Generator, index) -> (index)
      %b = arith.cmpi uge, %ri, %sparsity : index
      %r = arith.index_cast %ri : index to i64
      %f1 = arith.constant 1.0 : f64
      %f0 = arith.constant 0.0 : f64
      %insert = scf.if %b -> f64 {
        scf.yield %f1 : f64
      }  else {
        scf.yield %f0 : f64
      }
      tensor.yield %insert : f64
    } : tensor<?xf64>

    // func.call @dump(%tnsr) : (tensor<?x?xf64>) -> ()
    return %tnsr : tensor<?xf64>
  }

  // Generalizes linalg.conv_1d to specifies loop schedules.
  func.func @conv_1d_sparse_dense(%arg0: tensor<?xf64, #INPUT>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>) -> tensor<?xf64> {
     %0 = linalg.generic #SCHEDULE
          ins(%arg0, %arg1 : tensor<?xf64, #INPUT>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) attrs =  {sorted = true} {
     ^bb0(%in: f64, %in_0: f64, %out: f64):
       %1 = arith.mulf %in, %in_0 : f64
       %2 = arith.addf %out, %1 : f64
       linalg.yield %2 : f64
     } -> tensor<?xf64>
     return %0 : tensor<?xf64>
  }

  func.func @conv_1d_dense_dense(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<?xf64>) -> tensor<?xf64> {
     %0 = linalg.conv_1d
          ins(%arg0, %arg1 : tensor<?xf64>, tensor<?xf64>) outs(%arg2 : tensor<?xf64>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
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
    %c101 = arith.constant 101 : index
    %tmp = arith.subi %IL, %FL : index
    %OL = arith.addi %tmp, %c1 : index

    %g = func.call @rtsrand(%c0) : (index) ->(!Generator)

    %filter_sparsity = arith.constant 0 : index
    %filter = func.call @get_sparse_1d_tensor(%FL, %filter_sparsity, %g) :(index, index, !Generator) -> (tensor<?xf64>)
    %output_elem = arith.constant 0.0 : f64


    // Run sparse conv
    scf.for %input_sparsity = %c0 to %c101 step %c1 {
      // Construct input.
      %dense_input = func.call @get_sparse_1d_tensor(%IL, %input_sparsity, %g) :(index, index, !Generator) -> (tensor<?xf64>)
      %sparse_input = sparse_tensor.convert %dense_input: tensor<?xf64> to tensor<?xf64, #INPUT>

      %repeat = arith.constant REPEAT : index
      // Run sparse conv
      %dense_time, %sparse_time = scf.for %iv = %c0 to %repeat step %c1
        iter_args(%dense_sum = %f0, %sparse_sum = %f0) -> (f64, f64) {
        %dense_output = func.call @alloc_1d_filled_f64(%OL, %output_elem) :(index, f64) -> (tensor<?xf64>)
        %dense_start = func.call @rtclock() : () -> f64
        %dense_ret = func.call @conv_1d_dense_dense(%dense_input, %filter, %dense_output)
               : (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) -> (tensor<?xf64>)
        %dense_end = func.call @rtclock() : () -> f64
        bufferization.dealloc_tensor %dense_ret : tensor<?xf64>

        %sparse_output = func.call @alloc_1d_filled_f64(%OL, %output_elem) :(index, f64) -> (tensor<?xf64>)
        %sparse_start = func.call @rtclock() : () -> f64
        %sparse_ret = func.call @conv_1d_sparse_dense(%sparse_input, %filter, %sparse_output)
               : (tensor<?xf64, #INPUT>, tensor<?xf64>, tensor<?xf64>) -> (tensor<?xf64>)
        %sparse_end = func.call @rtclock() : () -> f64
        bufferization.dealloc_tensor %sparse_ret : tensor<?xf64>

        %dense_time = arith.subf %dense_end, %dense_start : f64
        %dense_next = arith.addf %dense_sum, %dense_time : f64

        %sparse_time = arith.subf %sparse_end, %sparse_start : f64
        %sparse_next = arith.addf %sparse_sum, %sparse_time : f64

        scf.yield %dense_next, %sparse_next : f64, f64
      }

      bufferization.dealloc_tensor %dense_input : tensor<?xf64>
      bufferization.dealloc_tensor %sparse_input : tensor<?xf64, #INPUT>

      %irep = arith.index_castui %repeat : index to i64
      %frep = arith.uitofp %irep : i64 to f64
      %f1000 = arith.constant 1000.0 : f64

      %dense_average_time = arith.divf %dense_time, %frep : f64
      %sparse_average_time = arith.divf %sparse_time, %frep : f64

      %dense_average_time_ms = arith.mulf %dense_average_time, %f1000 : f64
      %sparse_average_time_ms = arith.mulf %sparse_average_time, %f1000 : f64

      vector.print %input_sparsity : index
      vector.print %dense_average_time_ms : f64
      vector.print %sparse_average_time_ms : f64

    }
    return
  }

  func.func @entry() {
    %l = arith.constant LEN : index
    %c3 = arith.constant 3 : index
    call @runBenchmark(%l, %c3) :  (index, index) -> ()
    return
  }
}
