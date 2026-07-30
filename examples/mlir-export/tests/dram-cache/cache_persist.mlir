// Fixture for test-dram-cache-persist.
//
// The KV cache design rests on one property: a DRAM memref passed as an argument is written in place
// and keeps its contents between calls. This is the smallest function that exercises it - append
// %src into cell %slot of %cache, nothing else.
//
// Note the ", 1" on every memref: memory space 1 is DRAM. Without it the compiler rejects the
// module with "all memrefs should already be in DRAM memory space".
module {
  func.func @forward(%src:   memref<4xf16, 1>   {txe.name = "input_0"},
                     %slot:  index              {txe.name = "input_1"},
                     %cache: memref<8x4xf16, 1> {txe.name = "cache"})
      attributes {llvm.emit_c_interface} {
    %cell = memref.subview %cache[%slot, 0] [1, 4] [1, 1]
          : memref<8x4xf16, 1> to memref<4xf16, strided<[1], offset: ?>, 1>
    memref.copy %src, %cell
          : memref<4xf16, 1> to memref<4xf16, strided<[1], offset: ?>, 1>
    return
  }
}
