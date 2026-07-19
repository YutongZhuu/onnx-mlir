set(MYACCEL_ENABLED 1 BOOL PARENT_SCOPE)
option(MYACCEL_USE_XRT "Build MyAccel runtime with XRT FPGA support" OFF)

add_onnx_mlir_library(MyAccelRuntime STATIC
  Runtime/MyConv.c
  Runtime/MyAccelXrt.cpp
  EXCLUDE_FROM_OM_LIBS
  INCLUDE_DIRS PRIVATE
    ${ONNX_MLIR_SRC_ROOT}/include
    Runtime
)
set_target_properties(MyAccelRuntime PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  COMPILE_OPTIONS "-O3"
)
target_compile_options(MyAccelRuntime PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>"
)

add_onnx_mlir_library(OMMyAccelAccel
  MyAccelAccelerator.cpp
  EXCLUDE_FROM_OM_LIBS
  DEPENDS MyAccelRuntime
  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
    ${ONNX_MLIR_SRC_ROOT}
  LINK_LIBS PUBLIC
    onnx
    OMAccelerator
    OMKrnlOps
    OMONNXOps
    OMShapeInferenceOpInterface
    OMCompilerPasses
    OMKrnlToLLVM
    OMONNXToKrnl
)
