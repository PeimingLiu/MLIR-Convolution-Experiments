// DEFINE: %{compile} = mlir-opt %s --sparse-compiler
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
//
// RUN: %{compile} | %{run}

!Filename = !llvm.ptr
!Generator = !llvm.ptr

#DDC = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0: dense, d1 : dense, d2 : compressed) }>
#DCC = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0: dense, d1 : compressed, d2 : compressed) }>
#CCC = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0: compressed, d1 : compressed, d2 : compressed) }>

#ppprrr = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d2 + d5)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
}

#prprrp = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d1, d2 + d3, d4 + d5)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d5)>
  ],
  iterator_types = ["parallel", "reduction", "parallel", "reduction", "reduction", "parallel"]
}


module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @rtclock() -> (f64)

  func.func private @rtsrand(index) -> (!Generator)
  func.func private @rtrand(!Generator, index) -> (index)
  func.func private @rtdrand(!Generator) -> ()

  func.func @alloc_3d_filled_f64(%w : index, %h : index, %d : index,  %f : f64) -> tensor<?x?x?xf64> {
    %buf = bufferization.alloc_tensor(%w, %h, %d) : tensor<?x?x?xf64>
    %ret = linalg.fill ins(%f : f64) outs(%buf : tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    return %ret : tensor<?x?x?xf64>
  }

  func.func private @printMemref3dF32(%ptr : tensor<?x?x?xf64>) attributes { llvm.emit_c_interface }
  func.func @dump(%arg0: tensor<?x?x?xf64>) {
    call @printMemref3dF32(%arg0) : (tensor<?x?x?xf64>) -> ()
    return
  }

  func.func @get_sparse_3d_tensor(%w : index, %h : index,  %d : index,  %sparsity : index, %g : !Generator) -> tensor<?x?x?xf64> {
    %tnsr = tensor.generate %w, %h, %d {
    ^bb0(%i : index, %j: index, %k: index):
      %c99 = arith.constant 99 : index
      %ri = func.call @rtrand(%g, %c99) : (!Generator, index) -> (index)
      %b = arith.cmpi uge, %ri, %sparsity : index
      %r = arith.index_cast %ri : index to i64
      %f1 = arith.constant 1.0 : f64
      %f0 = arith.constant 0.0 : f64
      %insert = scf.if %b -> f64 {
        %fr = arith.uitofp %r : i64 to f64
        scf.yield %fr : f64
      }  else {
        scf.yield %f0 : f64
      }
      tensor.yield %insert : f64
    } : tensor<?x?x?xf64>

    // func.call @dump(%tnsr) : (tensor<?x?x?xf64>) -> ()
    return %tnsr : tensor<?x?x?xf64>
  }

  func.func @conv_3d_DDC_dense_SCHEDULE(%arg0: tensor<?x?x?xf64, #DDC>, %arg1: tensor<?x?x?xf64>, %arg2: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?x?xf64, #DDC>, tensor<?x?x?xf64>) outs(%arg2 : tensor<?x?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x?xf64>
    return %0 : tensor<?x?x?xf64>
  }

  func.func @conv_3d_DCC_dense_SCHEDULE(%arg0: tensor<?x?x?xf64, #DCC>, %arg1: tensor<?x?x?xf64>, %arg2: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?x?xf64, #DCC>, tensor<?x?x?xf64>) outs(%arg2 : tensor<?x?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x?xf64>
    return %0 : tensor<?x?x?xf64>
  }

  func.func @conv_3d_CCC_dense_SCHEDULE(%arg0: tensor<?x?x?xf64, #CCC>, %arg1: tensor<?x?x?xf64>, %arg2: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?x?xf64, #CCC>, tensor<?x?x?xf64>) outs(%arg2 : tensor<?x?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x?xf64>
    return %0 : tensor<?x?x?xf64>
  }

  func.func @conv_3d_dense_dense_SCHEDULE(%arg0: tensor<?x?x?xf64>, %arg1: tensor<?x?x?xf64>, %arg2: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%arg2 : tensor<?x?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x?xf64>
    return %0 : tensor<?x?x?xf64>
  }

  func.func @runBenchmark(%IW : index, %IH : index, %ID: index, %FL: index) {
    // Compute output shape
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c49 = arith.constant 49 : index
    %c50 = arith.constant 50 : index
    %f0 = arith.constant 0.0 : f64
    %f5 = arith.constant 5.0 : f64
    %c70 = arith.constant 70 : index
    %c71 = arith.constant 71 : index
    %c101 = arith.constant 101 : index
    %tmpW = arith.subi %IW, %FL : index
    %tmpH = arith.subi %IH, %FL : index
    %tmpD = arith.subi %ID, %FL : index
    %OW = arith.addi %tmpW, %c1 : index
    %OH = arith.addi %tmpH, %c1 : index
    %OD = arith.addi %tmpD, %c1 : index

    %g = func.call @rtsrand(%c0) : (index) ->(!Generator)

    %filter_sparsity = arith.constant 0 : index
    %filter = func.call @get_sparse_3d_tensor(%FL, %FL, %FL, %filter_sparsity, %g) :(index, index, index, index, !Generator) -> (tensor<?x?x?xf64>)
    %output_elem = arith.constant 0.0 : f64


    // Run sparse conv
    scf.for %input_sparsity = %c0 to %c101 step %c1 {
      // Construct input.
      %dense_input = func.call @get_sparse_3d_tensor(%IW, %IH, %ID, %input_sparsity, %g) :(index, index, index, index, !Generator) -> (tensor<?x?x?xf64>)
      %CCC_input = sparse_tensor.convert %dense_input: tensor<?x?x?xf64> to tensor<?x?x?xf64, #CCC>
      // %nnz = sparse_tensor.number_of_entries %CCC_input : tensor<?x?x?xf64, #CCC>
      // vector.print %nnz : index
      %DCC_input = sparse_tensor.convert %dense_input: tensor<?x?x?xf64> to tensor<?x?x?xf64, #DCC>

      %repeat = arith.constant REPEAT : index
      %dense_time_t = arith.constant dense<0.0> : vector<REPEATxf64>
      %CCC_time_t = arith.constant dense<0.0> : vector<REPEATxf64>
      %DCC_time_t = arith.constant dense<0.0> : vector<REPEATxf64>

      // Run sparse conv
      %dense_time, %CCC_time, %DCC_time = scf.for %iv = %c0 to %repeat step %c1
        iter_args(%dense_sum = %dense_time_t, %CCC_sum = %CCC_time_t, %DCC_sum = %DCC_time_t) -> (vector<REPEATxf64>, vector<REPEATxf64>, vector<REPEATxf64>) {

        %dense_output = func.call @alloc_3d_filled_f64(%OW, %OH, %OD, %output_elem) :(index, index, index, f64) -> (tensor<?x?x?xf64>)
        %dense_start = func.call @rtclock() : () -> f64
        %dense_ret = func.call @conv_3d_dense_dense_SCHEDULE(%dense_input, %filter, %dense_output)
               : (tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>)
        %dense_end = func.call @rtclock() : () -> f64
        func.call @dump(%dense_ret) : (tensor<?x?x?xf64>) -> ()
        bufferization.dealloc_tensor %dense_ret : tensor<?x?x?xf64>
        %dense_time = arith.subf %dense_end, %dense_start : f64
        %dense_next = vector.insertelement %dense_time, %dense_sum[%iv:index] : vector<REPEATxf64>

        %CCC_output = func.call @alloc_3d_filled_f64(%OW, %OH, %OD, %output_elem) :(index, index, index, f64) -> (tensor<?x?x?xf64>)
        %CCC_start = func.call @rtclock() : () -> f64
        %CCC_ret = func.call @conv_3d_CCC_dense_SCHEDULE(%CCC_input, %filter, %CCC_output)
               : (tensor<?x?x?xf64, #CCC>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>)
        %CCC_end = func.call @rtclock() : () -> f64
        func.call @dump(%CCC_ret) : (tensor<?x?x?xf64>) -> ()
        bufferization.dealloc_tensor %CCC_ret : tensor<?x?x?xf64>
        %CCC_time = arith.subf %CCC_end, %CCC_start : f64
        %CCC_next = vector.insertelement %CCC_time, %CCC_sum[%iv:index] : vector<REPEATxf64>

        %DCC_output = func.call @alloc_3d_filled_f64(%OW, %OH, %OD, %output_elem) :(index, index, index, f64) -> (tensor<?x?x?xf64>)
        %DCC_start = func.call @rtclock() : () -> f64
        %DCC_ret = func.call @conv_3d_DCC_dense_SCHEDULE(%DCC_input, %filter, %DCC_output)
               : (tensor<?x?x?xf64, #DCC>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> (tensor<?x?x?xf64>)
        %DCC_end = func.call @rtclock() : () -> f64
        func.call @dump(%DCC_ret) : (tensor<?x?x?xf64>) -> ()
        bufferization.dealloc_tensor %DCC_ret : tensor<?x?x?xf64>
        %DCC_time = arith.subf %DCC_end, %DCC_start : f64
        %DCC_next = vector.insertelement %DCC_time, %DCC_sum[%iv:index] : vector<REPEATxf64>

        scf.yield %dense_next, %CCC_next, %DCC_next : vector<REPEATxf64>, vector<REPEATxf64>, vector<REPEATxf64>
      }

      bufferization.dealloc_tensor %dense_input : tensor<?x?x?xf64>
      bufferization.dealloc_tensor %CCC_input : tensor<?x?x?xf64, #CCC>
      bufferization.dealloc_tensor %DCC_input : tensor<?x?x?xf64, #DCC>
      vector.print %input_sparsity : index // also the sparsity level

      %dense_time_min = vector.reduction <minf>, %dense_time : vector<REPEATxf64> into f64
      %CCC_time_min = vector.reduction <minf>, %CCC_time : vector<REPEATxf64> into f64
      %DCC_time_min = vector.reduction <minf>, %DCC_time : vector<REPEATxf64> into f64

      %dense_time_max = vector.reduction <maxf>, %dense_time : vector<REPEATxf64> into f64
      %CCC_time_max = vector.reduction <maxf>, %CCC_time : vector<REPEATxf64> into f64
      %DCC_time_max = vector.reduction <maxf>, %DCC_time : vector<REPEATxf64> into f64

      %dense_time_sum = vector.reduction <add>, %dense_time, %f0 : vector<REPEATxf64> into f64
      %CCC_time_sum = vector.reduction <add>, %CCC_time, %f0 : vector<REPEATxf64> into f64
      %DCC_time_sum = vector.reduction <add>, %DCC_time, %f0 : vector<REPEATxf64> into f64

      %dense_time_exc = arith.addf %dense_time_max, %dense_time_min : f64
      %dense_time_res = arith.subf %dense_time_sum, %dense_time_exc : f64
      vector.print %dense_time_res : f64

      %CCC_time_exc = arith.addf %CCC_time_max, %CCC_time_min : f64
      %CCC_time_res = arith.subf %CCC_time_sum, %CCC_time_exc : f64
      vector.print %CCC_time_res : f64

      %DCC_time_exc = arith.addf %DCC_time_max, %DCC_time_min : f64
      %DCC_time_res = arith.subf %DCC_time_sum, %DCC_time_exc : f64
      vector.print %DCC_time_res : f64
    }

    func.call @rtdrand(%g) : (!Generator) ->()
    return
  }

  func.func @entry() {
    %w = arith.constant WIDTH : index
    %h = arith.constant HEIGHT : index
    %d = arith.constant DEPTH : index
    %c3 = arith.constant 3 : index
    call @runBenchmark(%w, %h, %d, %c3) :  (index, index, index, index) -> ()
    return
  }
}
