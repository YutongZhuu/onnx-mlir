#include "src/Accelerators/MyAccel/MyAccelAccelerator.hpp"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir::accel {
namespace {

// Rewrite 1: select static, rank-4, f32 ONNX Conv operations and replace them
// with a call boundary understood by the accelerator runtime.
struct ConvToMyAccelCall final : OpConversionPattern<ONNXConvOp> {
  ConvToMyAccelCall(TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern(converter, ctx, PatternBenefit(100)) {}

  LogicalResult matchAndRewrite(ONNXConvOp conv, ONNXConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto xType = dyn_cast<RankedTensorType>(conv.getX().getType());
    auto wType = dyn_cast<RankedTensorType>(conv.getW().getType());
    auto yType = dyn_cast<RankedTensorType>(conv.getY().getType());
    if (!xType || !wType || !yType || xType.getRank() != 4 ||
        wType.getRank() != 4 || yType.getRank() != 4 ||
        !xType.hasStaticShape() || !wType.hasStaticShape() ||
        !yType.hasStaticShape() || !xType.getElementType().isF32() ||
        !wType.getElementType().isF32() || !yType.getElementType().isF32())
      return failure();

    Operation *op = conv.getOperation();
    Location loc = ONNXLoc<ONNXConvOp>(op);
    ValueRange operands = adaptor.getOperands();
    MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
        rewriter, loc);
    ONNXConvOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    if (failed(shapeHelper.computeShape()))
      return failure();
    std::vector<Value> outputs = allocForONNXOp<ONNXConvOp>(
        conv, rewriter, typeConverter, shapeHelper);

    auto getArrayValue = [](std::optional<ArrayAttr> attr, size_t index,
                             int64_t defaultValue) {
      if (!attr || index >= attr->size())
        return defaultValue;
      return cast<IntegerAttr>((*attr)[index]).getInt();
    };
    auto call = KrnlCallOp::create(
        rewriter, loc, "my_conv_f32", outputs, op, operands, false);
    // Flatten arrays because KrnlCall's LLVM rewrite accepts scalar integers.
    call->setAttr("dilation_h", rewriter.getI64IntegerAttr(
        getArrayValue(conv.getDilations(), 0, 1)));
    call->setAttr("dilation_w", rewriter.getI64IntegerAttr(
        getArrayValue(conv.getDilations(), 1, 1)));
    call->setAttr("group", rewriter.getI64IntegerAttr(conv.getGroup()));
    call->setAttr("pad_left", rewriter.getI64IntegerAttr(
        getArrayValue(conv.getPads(), 1, 0)));
    call->setAttr("pad_top", rewriter.getI64IntegerAttr(
        getArrayValue(conv.getPads(), 0, 0)));
    call->setAttr("stride_h", rewriter.getI64IntegerAttr(
        getArrayValue(conv.getStrides(), 0, 1)));
    call->setAttr("stride_w", rewriter.getI64IntegerAttr(
        getArrayValue(conv.getStrides(), 1, 1)));
    rewriter.replaceOp(op, outputs);
    return success();
  }
};
} // namespace

Accelerator *createMyAccel() { return MyAccelAccelerator::getInstance(); }

MyAccelAccelerator *MyAccelAccelerator::getInstance() {
  static MyAccelAccelerator instance;
  return &instance;
}

MyAccelAccelerator::MyAccelAccelerator()
    : Accelerator(Accelerator::Kind::MyAccel) {
  acceleratorTargets.push_back(this);
  addCompilerConfig(CCM_SHARED_LIB_DEPS, {"MyAccelRuntime"}, true);
}

void MyAccelAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &target,
    std::string outputName) const {
  onnx_mlir::addPasses(module, pm, target, outputName);
}

void MyAccelAccelerator::rewritePatternONNXToKrnl(
    RewritePatternSet &patterns, TypeConverter &converter,
    MLIRContext *ctx) const {
  patterns.add<ConvToMyAccelCall>(converter, ctx);
}

// Rewrite 2: lower krnl.call to an LLVM declaration/call. The implementation
// also marshals memrefs into OMTensor pointers for the stable C runtime ABI.
void MyAccelAccelerator::rewritePatternKrnlToLLVM(
    RewritePatternSet &patterns, LLVMTypeConverter &converter,
    MLIRContext *ctx) const {
  krnl::populateLoweringKrnlCallOpPattern(converter, patterns, ctx);
}

} // namespace onnx_mlir::accel
