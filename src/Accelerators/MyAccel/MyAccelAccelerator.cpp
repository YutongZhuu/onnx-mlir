#include "src/Accelerators/MyAccel/MyAccelAccelerator.hpp"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir::accel {
namespace {

static bool isStaticScalarLike(Value value, Type elementType) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getElementType() != elementType)
    return false;
  return type.getNumElements() == 1;
}

// If all Conv operands are produced by per-tensor QDQ nodes, call the INT8
// runtime directly with the quantized tensors and their quantization metadata.
// The runtime returns the exact FP32 value represented by the three input
// DequantizeLinear operations; the existing output QuantizeLinear is left in
// the graph so this rewrite does not change the public model interface.
static LogicalResult rewriteQDQConv(ONNXConvOp conv,
    ONNXConvOpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    const TypeConverter *typeConverter) {
  auto xDQ = conv.getX().getDefiningOp<ONNXDequantizeLinearOp>();
  auto wDQ = conv.getW().getDefiningOp<ONNXDequantizeLinearOp>();
  auto bDQ = conv.getB().getDefiningOp<ONNXDequantizeLinearOp>();
  if (!xDQ || !wDQ || !bDQ)
    return failure();

  auto xType = dyn_cast<RankedTensorType>(xDQ.getX().getType());
  auto wType = dyn_cast<RankedTensorType>(wDQ.getX().getType());
  auto bType = dyn_cast<RankedTensorType>(bDQ.getX().getType());
  auto yType = dyn_cast<RankedTensorType>(conv.getY().getType());
  auto i8Type = rewriter.getI8Type();
  auto i32Type = rewriter.getI32Type();
  auto f32Type = rewriter.getF32Type();
  if (!xType || !wType || !bType || !yType || xType.getRank() != 4 ||
      wType.getRank() != 4 || bType.getRank() != 1 || yType.getRank() != 4 ||
      !xType.hasStaticShape() || !wType.hasStaticShape() ||
      !bType.hasStaticShape() || !yType.hasStaticShape() ||
      xType.getElementType() != i8Type || wType.getElementType() != i8Type ||
      bType.getElementType() != i32Type || !yType.getElementType().isF32() ||
      !isStaticScalarLike(xDQ.getXScale(), f32Type) ||
      !isStaticScalarLike(xDQ.getXZeroPoint(), i8Type) ||
      !isStaticScalarLike(wDQ.getXScale(), f32Type) ||
      !isStaticScalarLike(wDQ.getXZeroPoint(), i8Type) ||
      !isStaticScalarLike(bDQ.getXScale(), f32Type) ||
      !isStaticScalarLike(bDQ.getXZeroPoint(), i32Type))
    return failure();

  Operation *op = conv.getOperation();
  Location loc = ONNXLoc<ONNXConvOp>(op);
  MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
      rewriter, loc);
  ONNXConvOpShapeHelper shapeHelper(op, adaptor.getOperands(), &create.krnlIE);
  if (failed(shapeHelper.computeShape()))
    return failure();
  std::vector<Value> outputs = allocForONNXOp<ONNXConvOp>(
      conv, rewriter, typeConverter, shapeHelper);

  SmallVector<Value, 9> originalQuantizedOperands = {
      xDQ.getX(), xDQ.getXScale(), xDQ.getXZeroPoint(),
      wDQ.getX(), wDQ.getXScale(), wDQ.getXZeroPoint(),
      bDQ.getX(), bDQ.getXScale(), bDQ.getXZeroPoint()};
  SmallVector<Value, 9> convertedQuantizedOperands;
  for (Value operand : originalQuantizedOperands) {
    Value converted = rewriter.getRemappedValue(operand);
    if (!converted)
      return failure();
    convertedQuantizedOperands.push_back(converted);
  }

  auto getArrayValue = [](std::optional<ArrayAttr> attr, size_t index,
                           int64_t defaultValue) {
    if (!attr || index >= attr->size())
      return defaultValue;
    return cast<IntegerAttr>((*attr)[index]).getInt();
  };
  auto call = KrnlCallOp::create(rewriter, loc, "my_conv_qdq_i8_f32",
      outputs, op, convertedQuantizedOperands, false);
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

// Rewrite 1: select static, rank-4, f32 ONNX Conv operations and replace them
// with a call boundary understood by the accelerator runtime.
struct ConvToMyAccelCall final : OpConversionPattern<ONNXConvOp> {
  ConvToMyAccelCall(TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern(converter, ctx, PatternBenefit(100)) {}

  LogicalResult matchAndRewrite(ONNXConvOp conv, ONNXConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (succeeded(rewriteQDQConv(conv, adaptor, rewriter, typeConverter)))
      return success();

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
