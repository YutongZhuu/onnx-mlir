#include "src/Accelerators/Accelerator.hpp"
#include "src/Accelerators/MyAccel/MyAccelAccelerator.hpp"

#include <cstdlib>

namespace onnx_mlir::accel {

// The release binary exposes NNPA as accelerator enum value 1. A compiler
// built with only MyAccel also assigns MyAccel value 1, so this preload shim
// replaces accelerator initialization while preserving the binary ABI.
void initAccelerators(llvm::ArrayRef<Accelerator::Kind> kinds) {
  // Do not inject compiler libraries into child tools such as llc/clang.
  unsetenv("LD_PRELOAD");
  if (!kinds.empty())
    createMyAccel()->setName("MyAccel");
}

} // namespace onnx_mlir::accel
