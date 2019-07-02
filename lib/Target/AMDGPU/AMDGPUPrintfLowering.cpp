//=== PrintfLowering.cpp -- implement printf on ROCm ===//
//===----------------------------------------------------------------------===//
// August 2019.
//
//   This pass converts each call to printf into a sequence of writes
//   to the asynchronous stream:
//   - The format string is copied into the stream.
//   - Each argument is copied after the format string.
//   - A char* argument is assumed to be a null-terminated string when copying.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "printflowering"

namespace {
class LLVM_LIBRARY_VISIBILITY AMDGPUPrintfLowering : public ModulePass {
public:
  static char ID;
  AMDGPUPrintfLowering();
  bool runOnModule(Module &M) override;
  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  void initAnalysis(Module &M) {}
};
} // namespace

char AMDGPUPrintfLowering::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUPrintfLowering, "amdgpu-printf-lowering",
                      "AMDGPU Printf lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUPrintfLowering, "amdgpu-printf-lowering",
                    "AMDGPU Printf Lowering for Asynchronous Streams", false, false)

char &llvm::AMDGPUPrintfLoweringID = AMDGPUPrintfLowering::ID;

namespace llvm {
ModulePass *createAMDGPUPrintfLoweringPass() {
  return new AMDGPUPrintfLowering();
}
} // namespace llvm

AMDGPUPrintfLowering::AMDGPUPrintfLowering() : ModulePass(ID) {
  initializeAMDGPUPrintfLoweringPass(*PassRegistry::getPassRegistry());
}

typedef enum {
  ARGTYPE_INVALID = 0,
  ARGTYPE_INT32 = 1,
  ARGTYPE_INT64 = 2,
  ARGTYPE_FLOAT64 = 3,
  ARGTYPE_CSTRING = 4,
  ARGTYPE_POINTER = 5
} PrintfTypeID;

static bool isCString(const Value *Arg) {
  auto Ty = Arg->getType();
  auto PtrTy = dyn_cast<PointerType>(Ty);
  if (!PtrTy)
    return false;

  auto IntTy = dyn_cast<IntegerType>(PtrTy->getElementType());
  if (!IntTy)
    return false;

  return IntTy->getBitWidth() == 8;
}

static uint64_t getArgTypeId(const Value *Arg) {
  auto Ty = Arg->getType();
  if (auto IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getBitWidth()) {
    case 32:
      return ARGTYPE_INT32;
    case 64:
      return ARGTYPE_INT64;
    }
  } else if (Ty->getTypeID() == Type::DoubleTyID) {
    return ARGTYPE_FLOAT64;
  } else if (auto PtrTy = dyn_cast<PointerType>(Ty)) {
    return ARGTYPE_POINTER;
  }

  llvm_unreachable("unexpected type");
  return ARGTYPE_INVALID;
}

static Value *fitArgInto64Bits(IRBuilder<> &Builder, Value *Arg,
                               uint64_t typeID) {
  auto Int64Ty = Builder.getInt64Ty();
  switch (typeID) {
  case ARGTYPE_INT64:
    return Arg;
  case ARGTYPE_INT32:
    return Builder.CreateZExt(Arg, Int64Ty);
  case ARGTYPE_FLOAT64:
    return Builder.CreateBitCast(Arg, Int64Ty);
  case ARGTYPE_CSTRING:
  case ARGTYPE_POINTER:
    return Builder.CreatePtrToInt(Arg, Int64Ty);
  default:
    llvm_unreachable("unexpected type ID");
    return nullptr;
  }
}

static FunctionCallee getFnPrintfBegin(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto F = M->getOrInsertFunction("__ockl_printf_begin", Int64Ty, Int64Ty,
                                  Int64Ty, Int64Ty);
  return F;
}

static FunctionCallee getFnAppendBytes(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto CharPtrTy = Builder.getInt8PtrTy();
  auto IntTy = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto F = M->getOrInsertFunction("__ockl_printf_append_bytes", Int64Ty,
                                  Int64Ty, CharPtrTy, Int64Ty, IntTy);
  return F;
}

static FunctionCallee getFnAppendArg(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto F = M->getOrInsertFunction("__ockl_printf_append_arg", Int64Ty, Int64Ty,
                                  Int64Ty, Int64Ty, Int32Ty);
  return F;
}

static Value *callPrintfBegin(IRBuilder<> &Builder, Value *Version,
                              Value *NumArgs, Value *FmtLength) {
  auto Fn = getFnPrintfBegin(Builder);
  return Builder.CreateCall(Fn, {Version, NumArgs, FmtLength});
}

static Value *callAppendBytes(IRBuilder<> &Builder, Value *Desc, Value *Str,
                              Value *Len, bool isLast) {
  auto IsLastInt32 = Builder.getInt32(isLast);
  auto Fn = getFnAppendBytes(Builder);
  return Builder.CreateCall(Fn, {Desc, Str, Len, IsLastInt32});
}

static Value *callAppendArg(IRBuilder<> &Builder, Value *Desc, uint64_t typeID,
                            Value *Arg, bool isLast) {
  auto IsLastInt32 = Builder.getInt32(isLast);
  auto TypeIdValue = Builder.getInt64(typeID);
  Arg = fitArgInto64Bits(Builder, Arg, typeID);
  return Builder.CreateCall(getFnAppendArg(Builder),
                            {Desc, TypeIdValue, Arg, IsLastInt32});
}

static Value *computeStringLength(IRBuilder<> &Builder, Value *CharString) {
  auto *Prev = Builder.GetInsertBlock();
  Module *M = Prev->getModule();

  const DataLayout &DL = M->getDataLayout();
  auto CharZero = Builder.getInt8(0);
  auto One = Builder.getIntN(DL.getPointerSizeInBits(), 1);
  auto Zero = Builder.getIntN(DL.getPointerSizeInBits(), 0);
  auto SizeTy = One->getType();

  auto Prefix = CharString->getName();
  // Check for null pointer
  BasicBlock *IfNull = Prev->splitBasicBlock(Builder.GetInsertPoint(),
                                             Prefix + ".strlen.null");
  auto CmpNull =
      new ICmpInst(Prev->getTerminator(), CmpInst::ICMP_EQ, CharString,
                   Constant::getNullValue(CharString->getType()));
  Prev->getTerminator()->eraseFromParent();
  BasicBlock *While =
      BasicBlock::Create(M->getContext(), Prefix + ".strlen.while",
                         Prev->getParent(), IfNull);
  BranchInst::Create(IfNull, While, CmpNull, Prev);

  // A while-loop that checks for end of string.
  Builder.SetInsertPoint(While);
  auto Str = Builder.CreateAddrSpaceCast(CharString, Builder.getInt8PtrTy());

  auto PHI = Builder.CreatePHI(Str->getType(), 2);
  PHI->addIncoming(Str, Prev);
  auto GEPNext = Builder.CreateGEP(PHI, One);
  PHI->addIncoming(GEPNext, While);

  // Loop-exit
  BasicBlock *WhileDone = BasicBlock::Create(
      M->getContext(), Prefix + ".strlen.while.done",
      Prev->getParent(), IfNull);
  BranchInst::Create(IfNull, WhileDone);
  auto Data = Builder.CreateLoad(PHI);
  auto Cmp = Builder.CreateICmpEQ(Data, CharZero);
  Builder.CreateCondBr(Cmp, WhileDone, While);

  // Pointer arithmetic at loop exit:
  // len = end - start + 1 ... this includes the nul character.
  Builder.SetInsertPoint(WhileDone, WhileDone->begin());

  auto Begin = Builder.CreatePtrToInt(CharString, SizeTy);
  auto End = Builder.CreatePtrToInt(PHI, SizeTy);
  auto Len = Builder.CreateSub(End, Begin);
  Len = Builder.CreateAdd(Len, One);

  Builder.SetInsertPoint(IfNull, IfNull->begin());
  PHI = Builder.CreatePHI(Len->getType(), 2);
  PHI->addIncoming(Len, WhileDone);
  PHI->addIncoming(Zero, Prev);

  return PHI;
}

static Value *processArg(IRBuilder<> &Builder, Value *Desc, Value *Arg,
                         bool SpecIsCString, bool isLast) {
  if (SpecIsCString && isCString(Arg)) {
    auto Length = computeStringLength(Builder, Arg);
    Desc = callAppendArg(Builder, Desc, ARGTYPE_CSTRING, Length, false);
    return callAppendBytes(Builder, Desc, Arg, Length, isLast);
  }
  auto TypeId = getArgTypeId(Arg);
  return callAppendArg(Builder, Desc, TypeId, Arg, isLast);
}

static void collectPrintfsFromModule(SmallVectorImpl<Value *> &Printfs,
                                     Module &M) {
  for (auto &MF : M) {
    if (MF.isDeclaration())
      continue;
    for (auto &BB : MF) {
      for (auto &Instr : BB) {
        CallInst *CI = dyn_cast<CallInst>(&Instr);
        if (CI && CI->getCalledFunction() &&
            CI->getCalledFunction()->getName() == "printf") {
          Printfs.push_back(CI);
        }
      }
    }
  }
}

static void removePrintf(Module &M) {
  if (Function *F = M.getFunction("printf"))
    F->eraseFromParent();
}

static void locateCStrings(SparseBitVector<64> &BV, Value *Ptr) {
    Ptr->stripPointerCasts()->dump();
    auto Const = dyn_cast<Constant>(Ptr->stripPointerCasts());
  if (!Const)
    return;
  GlobalVariable *GVar = dyn_cast<GlobalVariable>(Const);
  if (!GVar || !GVar->hasInitializer())
      return;
  auto Init = GVar->getInitializer();
  auto CA = dyn_cast<ConstantDataArray>(Init);
  if (!CA || !CA->isString())
    return;
  auto Fmt = CA->getAsCString();

  static const char ConvSpecifiers[] = "diouxXfFeEgGaAcspnCS";
  size_t SpecPos = 0;
  // Skip the first argument, the format string.
  unsigned ArgIdx = 1;

  while ((SpecPos = Fmt.find_first_of('%', SpecPos)) != StringRef::npos) {
    if (Fmt[SpecPos + 1] == '%') {
      ++SpecPos;
      continue;
    }
    auto SpecEnd = Fmt.find_first_of(ConvSpecifiers, SpecPos);
    if (SpecEnd == StringRef::npos)
      return;
    auto Spec = Fmt.slice(SpecPos, SpecEnd + 1);
    ArgIdx += Spec.count('*');
    if (Fmt[SpecEnd] == 's') {
      BV.set(ArgIdx);
    }
    SpecPos = SpecEnd + 1;
    ++ArgIdx;
  }
}

static void lowerPrintf(IRBuilder<> &Builder, Value *P) {
  CallInst *CI = cast<CallInst>(P);
  auto NumOps = CI->getNumArgOperands();
  assert(NumOps >= 1);

  Builder.SetInsertPoint(CI);
  Builder.SetCurrentDebugLocation(CI->getDebugLoc());

  auto NumArgs = Builder.getIntN(64, NumOps - 1);
  auto Fmt = CI->getArgOperand(0);
  auto FmtLength = computeStringLength(Builder, Fmt);
  SparseBitVector<64> SpecIsCString;
  locateCStrings(SpecIsCString, Fmt);

  auto Desc =
      callPrintfBegin(Builder, Builder.getIntN(64, 0), NumArgs, FmtLength);
  Desc = callAppendBytes(Builder, Desc, Fmt, FmtLength, NumOps == 1);

  // Write out the actual arguments following the format string.
  for (unsigned int i = 1; i != NumOps; ++i) {
    bool isLast = i == NumOps - 1;
    bool isCString = SpecIsCString.test(i);
    Desc = processArg(Builder, Desc, CI->getArgOperand(i), isCString, isLast);
  }

  Desc =
      Builder.CreateIntCast(Desc, Builder.getInt32Ty(), /* isSigned = */ true);
  P->replaceAllUsesWith(Desc);
}

bool AMDGPUPrintfLowering::runOnModule(Module &M) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);

  SmallVector<Value *, 32> Printfs;
  collectPrintfsFromModule(Printfs, M);
  if (Printfs.empty()) {
    return false;
  }

  for (auto P : Printfs) {
    lowerPrintf(Builder, P);
  }

  // erase the printf calls
  for (auto P : Printfs) {
    CallInst *CI = dyn_cast<CallInst>(P);
    CI->eraseFromParent();
  }

  removePrintf(M);

  return true;
}

bool AMDGPUPrintfLowering::doInitialization(Module &M) { return false; }

bool AMDGPUPrintfLowering::doFinalization(Module &M) { return false; }
