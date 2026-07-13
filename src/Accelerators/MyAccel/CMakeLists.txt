set(MYACCEL_ENABLED 1 BOOL PARENT_SCOPE)

add_onnx_mlir_library(MyAccelRuntime STATIC
  Runtime/MyConv.c
  EXCLUDE_FROM_OM_LIBS
  INCLUDE_DIRS PRIVATE
    ${ONNX_MLIR_SRC_ROOT}/include
)
set_target_properties(MyAccelRuntime PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  COMPILE_OPTIONS "-O3"
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
