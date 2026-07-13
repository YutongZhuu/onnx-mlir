#ifndef ONNX_MLIR_MYACCEL_COMPILER_OPTIONS_H
#define ONNX_MLIR_MYACCEL_COMPILER_OPTIONS_H

// MyAccel does not add instrumentation/profile/report enum values, but the
// accelerator registry requires each extension to define these hooks.
#define INSTRUMENTSTAGE_ENUM_MyAccel
#define INSTRUMENTSTAGE_CL_ENUM_MyAccel
#define PROFILEIR_CL_ENUM_MyAccel
#define OPTREPORT_ENUM_MyAccel
#define OPTREPORT_CL_ENUM_MyAccel

#endif
