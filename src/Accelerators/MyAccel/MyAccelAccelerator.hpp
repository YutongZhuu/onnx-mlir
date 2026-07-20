#ifndef ONNX_MLIR_MYACCEL_ACCELERATOR_H
#define ONNX_MLIR_MYACCEL_ACCELERATOR_H

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir::accel {

Accelerator *createMyAccel();

class MyAccelAccelerator final : public Accelerator {
public:
  static MyAccelAccelerator *getInstance();
  static bool classof(const Accelerator *accel) {
    return accel->getKind() == Accelerator::Kind::MyAccel;
  }

  uint64_t getVersionNumber() const final { return 0x00010000; }
  void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &, mlir::PassManager &,
      onnx_mlir::EmissionTargetType &, std::string) const final;
  void registerDialects(mlir::DialectRegistry &) const final {}
  void registerPasses(int) const final {}
  void configurePasses() const final {}
  mlir::MemRefType convertTensorTypeToMemRefType(
      const mlir::TensorType) const final { return nullptr; }
  void conversionTargetONNXToKrnl(mlir::ConversionTarget &) const final {}
  void rewritePatternONNXToKrnl(mlir::RewritePatternSet &,
      mlir::TypeConverter &, mlir::MLIRContext *) const final;
  void conversionTargetKrnlToLLVM(mlir::ConversionTarget &) const final {}
  void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &,
      mlir::LLVMTypeConverter &, mlir::MLIRContext *) const final;

private:
  MyAccelAccelerator();
};

} // namespace onnx_mlir::accel
#endif
