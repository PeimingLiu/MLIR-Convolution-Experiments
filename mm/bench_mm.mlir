// RUN(conv2d-dd-lib, 5): export TENSOR0=${MMREPO}/fidapm37.mtx ; ${PMLIR}/mlir-opt --sparsifier=enable-runtime-library=true %s | ${PMLIR}/mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=${PMLIB}/libmlir_c_runner_utils.so
// RUN(conv2d-dd-cod, 5): export TENSOR0=${MMREPO}/fidapm37.mtx ; ${PMLIR}/mlir-opt --sparsifier=enable-runtime-library=false %s | ${PMLIR}/mlir-cpu-runner -e entry -entry-point-result=void -shared-libs=${PMLIB}/libmlir_c_runner_utils.so

// TODO: use dynamic sized tensor to allow different shapes of input matrix

!Filename = !llvm.ptr

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

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

  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @conv_2d_SCHEDULE(%arg0: tensor<?x?xf64, #CSR>,
                                       %arg1: tensor<3x3xf64>,
                                       %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #SCHEDULE
    ins(%arg0, %arg1 : tensor<?x?xf64, #CSR>, tensor<3x3xf64>)
    outs(%arg2 : tensor<?x?xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }


  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // A typical edge detection filter.
    %filter = arith.constant dense<[
      [  1.0,  0.0, -1.0 ],
      [  0.0,  0.0,  0.0 ],
      [ -1.0,  0.0,  1.0 ]
    ]> : tensor<3x3xf64>

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %input = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #CSR>

    %d0 = tensor.dim %input, %c0 : tensor<?x?xf64, #CSR>
    %d1 = tensor.dim %input, %c1 : tensor<?x?xf64, #CSR>
    %o0 = arith.subi %d0, %c2 : index
    %o1 = arith.subi %d1, %c2 : index

    %output = tensor.empty(%o0, %o1) : tensor<?x?xf64>

    // Call and time kernel.
    %t0 = call @rtclock() : () -> f64
    %ret = call @conv_2d_SCHEDULE(%input, %filter, %output) : (tensor<?x?xf64, #CSR>, tensor<3x3xf64>, tensor<?x?xf64>) -> (tensor<?x?xf64>)
    %t1 = call @rtclock() : () -> f64

    // Report time.
    %tt = arith.subf %t1, %t0 : f64
    vector.print %tt : f64

    // Release the resources.
    bufferization.dealloc_tensor %input : tensor<?x?xf64, #CSR>

    return
  }
}
