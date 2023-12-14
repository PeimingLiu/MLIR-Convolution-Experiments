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

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d1, d3 + d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>

#prrp = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0 + d1, d3 + d2)>,
    affine_map<(d0, d1, d2, d3) -> (d1, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d3)>
  ],
  iterator_types = ["parallel", "reduction", "reduction", "parallel"]
}

#prpr = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0 + d1, d2 + d3)>,
    affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d2)>
  ],
  iterator_types = ["parallel", "reduction", "parallel", "reduction"]
}

#pprr = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"]
}

#rrpp = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>
  ],
  iterator_types = ["reduction", "reduction", "parallel", "parallel"]
}

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @rtclock() -> (f64)

  func.func private @rtsrand(index) -> (!Generator)
  func.func private @rtrand(!Generator, index) -> (index)
  func.func private @rtdrand(!Generator) -> ()

  func.func @alloc_2d_filled_f64(%w : index, %h : index, %f : f64) -> tensor<?x?xf64> {
    %buf = bufferization.alloc_tensor(%w, %h) : tensor<?x?xf64>
    %ret = linalg.fill ins(%f : f64) outs(%buf : tensor<?x?xf64>) -> tensor<?x?xf64>
    return %ret : tensor<?x?xf64>
  }

  func.func private @printMemrefF64(%ptr : tensor<*xf64>) attributes { llvm.emit_c_interface }
  func.func @dump(%arg0: tensor<?x?xf64>) {
    %mem = tensor.cast %arg0 : tensor<?x?xf64> to tensor<*xf64>
    call @printMemrefF64(%mem) : (tensor<*xf64>) -> ()
    return
  }

  func.func @get_sparse_2d_tensor(%w : index, %h : index, %sparsity : index, %g : !Generator) -> tensor<?x?xf64> {
    %tnsr = tensor.generate %w, %h {
    ^bb0(%i : index, %j: index):
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
    } : tensor<?x?xf64>

    // func.call @dump(%tnsr) : (tensor<?x?xf64>) -> ()
    return %tnsr : tensor<?x?xf64>
  }

  func.func @conv_2d_DC_dense_SCHEDULE(%arg0: tensor<?x?xf64, #DC>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?xf64, #DC>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @conv_2d_CC_dense_SCHEDULE(%arg0: tensor<?x?xf64, #CC>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?xf64, #CC>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @conv_2d_dense_dense_SCHEDULE(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%arg2 : tensor<?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @runBenchmark(%IW : index, %IH : index, %FL: index) {
    // Compute output shape
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
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
    %OW = arith.addi %tmpW, %c1 : index
    %OH = arith.addi %tmpH, %c1 : index

    %g = func.call @rtsrand(%c0) : (index) ->(!Generator)

    %filter_sparsity = arith.constant 0 : index
    %filter = func.call @get_sparse_2d_tensor(%FL, %FL, %filter_sparsity, %g) :(index, index, index, !Generator) -> (tensor<?x?xf64>)
    %output_elem = arith.constant 0.0 : f64


    // Run sparse conv
    scf.for %input_sparsity = %c0 to %c101 step %c1 {
      // Construct input.
      %dense_input = func.call @get_sparse_2d_tensor(%IW, %IH, %input_sparsity, %g) :(index, index, index, !Generator) -> (tensor<?x?xf64>)
      %CC_input = sparse_tensor.convert %dense_input: tensor<?x?xf64> to tensor<?x?xf64, #CC>
      // %nnz = sparse_tensor.number_of_entries %CC_input : tensor<?x?xf64, #CC>
      // vector.print %nnz : index
      %DC_input = sparse_tensor.convert %dense_input: tensor<?x?xf64> to tensor<?x?xf64, #DC>

      %repeat = arith.constant REPEAT : index
      %dense_time_t = arith.constant dense<0.0> : vector<REPEATxf64>
      %CC_time_t = arith.constant dense<0.0> : vector<REPEATxf64>
      %DC_time_t = arith.constant dense<0.0> : vector<REPEATxf64>

      // Run sparse conv
      %dense_time, %CC_time, %DC_time = scf.for %iv = %c0 to %repeat step %c1
        iter_args(%dense_sum = %dense_time_t, %CC_sum = %CC_time_t, %DC_sum = %DC_time_t) -> (vector<REPEATxf64>, vector<REPEATxf64>, vector<REPEATxf64>) {

        %dense_output = func.call @alloc_2d_filled_f64(%OW, %OH, %output_elem) :(index, index, f64) -> (tensor<?x?xf64>)
        %dense_start = func.call @rtclock() : () -> f64
        %dense_ret = func.call @conv_2d_dense_dense_SCHEDULE(%dense_input, %filter, %dense_output)
               : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> (tensor<?x?xf64>)
        %dense_end = func.call @rtclock() : () -> f64
        // func.call @dump(%dense_ret) : (tensor<?x?xf64>) -> ()
        bufferization.dealloc_tensor %dense_ret : tensor<?x?xf64>
        %dense_time = arith.subf %dense_end, %dense_start : f64
        %dense_next = vector.insertelement %dense_time, %dense_sum[%iv:index] : vector<REPEATxf64>

        %CC_output = func.call @alloc_2d_filled_f64(%OW, %OH, %output_elem) :(index, index, f64) -> (tensor<?x?xf64>)
        %CC_start = func.call @rtclock() : () -> f64
        %CC_ret = func.call @conv_2d_CC_dense_SCHEDULE(%CC_input, %filter, %CC_output)
               : (tensor<?x?xf64, #CC>, tensor<?x?xf64>, tensor<?x?xf64>) -> (tensor<?x?xf64>)
        %CC_end = func.call @rtclock() : () -> f64
        // func.call @dump(%CC_ret) : (tensor<?x?xf64>) -> ()
        bufferization.dealloc_tensor %CC_ret : tensor<?x?xf64>
        %CC_time = arith.subf %CC_end, %CC_start : f64
        %CC_next = vector.insertelement %CC_time, %CC_sum[%iv:index] : vector<REPEATxf64>

        %DC_output = func.call @alloc_2d_filled_f64(%OW, %OH, %output_elem) :(index, index, f64) -> (tensor<?x?xf64>)
        %DC_start = func.call @rtclock() : () -> f64
        %DC_ret = func.call @conv_2d_DC_dense_SCHEDULE(%DC_input, %filter, %DC_output)
               : (tensor<?x?xf64, #DC>, tensor<?x?xf64>, tensor<?x?xf64>) -> (tensor<?x?xf64>)
        %DC_end = func.call @rtclock() : () -> f64
        // func.call @dump(%DC_ret) : (tensor<?x?xf64>) -> ()
        bufferization.dealloc_tensor %DC_ret : tensor<?x?xf64>
        %DC_time = arith.subf %DC_end, %DC_start : f64
        %DC_next = vector.insertelement %DC_time, %DC_sum[%iv:index] : vector<REPEATxf64>

        scf.yield %dense_next, %CC_next, %DC_next : vector<REPEATxf64>, vector<REPEATxf64>, vector<REPEATxf64>
      }

      bufferization.dealloc_tensor %dense_input : tensor<?x?xf64>
      bufferization.dealloc_tensor %CC_input : tensor<?x?xf64, #CC>
      bufferization.dealloc_tensor %DC_input : tensor<?x?xf64, #DC>
      vector.print %input_sparsity : index // also the sparsity level

      %dense_time_min = vector.reduction <minf>, %dense_time : vector<REPEATxf64> into f64
      %CC_time_min = vector.reduction <minf>, %CC_time : vector<REPEATxf64> into f64
      %DC_time_min = vector.reduction <minf>, %DC_time : vector<REPEATxf64> into f64

      %dense_time_max = vector.reduction <maxf>, %dense_time : vector<REPEATxf64> into f64
      %CC_time_max = vector.reduction <maxf>, %CC_time : vector<REPEATxf64> into f64
      %DC_time_max = vector.reduction <maxf>, %DC_time : vector<REPEATxf64> into f64

      %dense_time_sum = vector.reduction <add>, %dense_time, %f0 : vector<REPEATxf64> into f64
      %CC_time_sum = vector.reduction <add>, %CC_time, %f0 : vector<REPEATxf64> into f64
      %DC_time_sum = vector.reduction <add>, %DC_time, %f0 : vector<REPEATxf64> into f64

      %dense_time_exc = arith.addf %dense_time_max, %dense_time_min : f64
      %dense_time_res = arith.subf %dense_time_sum, %dense_time_exc : f64

      %CC_time_exc = arith.addf %CC_time_max, %CC_time_min : f64
      %CC_time_res = arith.subf %CC_time_sum, %CC_time_exc : f64

      %DC_time_exc = arith.addf %DC_time_max, %DC_time_min : f64
      %DC_time_res = arith.subf %DC_time_sum, %DC_time_exc : f64

      %rep = arith.subi %repeat, %c2 : index
      %irep = arith.index_castui %rep : index to i64
      %frep = arith.uitofp %irep : i64 to f64
      %f1000 = arith.constant 1000.0 : f64

      %dense_average_time = arith.divf %dense_time_res, %frep : f64
      %CC_average_time = arith.divf %CC_time_res, %frep : f64
      %DC_average_time = arith.divf %DC_time_res, %frep : f64

      %dense_average_time_ms = arith.mulf %dense_average_time, %f1000 : f64
      %CC_average_time_ms = arith.mulf %CC_average_time, %f1000 : f64
      %DC_average_time_ms = arith.mulf %DC_average_time, %f1000 : f64

      vector.print %dense_average_time_ms : f64
      vector.print %CC_average_time_ms : f64
      vector.print %DC_average_time_ms : f64
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
