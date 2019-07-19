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
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "printflowering"

namespace {
class LLVM_LIBRARY_VISIBILITY AMDGPUPrintfLowering : public ModulePass {
public:
  static char ID;
  AMDGPUPrintfLowering() : ModulePass(ID) {};
private:
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};
} // namespace

char AMDGPUPrintfLowering::ID = 0;

INITIALIZE_PASS(AMDGPUPrintfLowering, "amdgpu-printf-lowering",
                      "AMDGPU: Lower printf to hostcall", false, false)

char &llvm::AMDGPUPrintfLoweringID = AMDGPUPrintfLowering::ID;

namespace llvm {
ModulePass *createAMDGPUPrintfLoweringPass() {
  return new AMDGPUPrintfLowering();
}
} // namespace llvm

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

static Value *fitArgInto64Bits(IRBuilder<> &Builder, Value *Arg) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Ty = Arg->getType();
  if (auto IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getBitWidth()) {
    case 32:
      return Builder.CreateZExt(Arg, Int64Ty);
    case 64:
      return Arg;
    }
  } else if (Ty->getTypeID() == Type::DoubleTyID) {
    return Builder.CreateBitCast(Arg, Int64Ty);
  } else if (auto PtrTy = dyn_cast<PointerType>(Ty)) {
    return Builder.CreatePtrToInt(Arg, Int64Ty);
  }

  llvm_unreachable("unexpected type");
  return Builder.getInt64(0);
}

static FunctionCallee getFnPrintfBegin(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto F = M->getOrInsertFunction("__ockl_printf_begin", Int64Ty, Int64Ty);
  return F;
}

static Value *callPrintfBegin(IRBuilder<> &Builder, Value *Version) {
  auto Fn = getFnPrintfBegin(Builder);
  return Builder.CreateCall(Fn, Version);
}

static FunctionCallee getFnAppendArgs(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto F = M->getOrInsertFunction("__ockl_printf_append_args", Int64Ty, Int64Ty,
                                  Int32Ty,
                                  Int64Ty, Int64Ty, Int64Ty, Int64Ty,
                                  Int64Ty, Int64Ty, Int64Ty, Int32Ty);
  return F;
}

static Value *callAppendArgs(IRBuilder<> &Builder, Value *Desc, int NumArgs,
                             Value *Arg0, Value *Arg1, Value *Arg2, Value *Arg3,
                             Value *Arg4, Value *Arg5, Value *Arg6, bool IsLast) {
  auto IsLastValue = Builder.getInt32(IsLast);
  auto NumArgsValue = Builder.getInt32(NumArgs);
  return Builder.CreateCall(getFnAppendArgs(Builder),
                            {Desc, NumArgsValue, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, IsLastValue});
}

static FunctionCallee getFnAppendStringN(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto CharPtrTy = Builder.getInt8PtrTy();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto F = M->getOrInsertFunction("__ockl_printf_append_string_n", Int64Ty, Int64Ty, CharPtrTy, Int64Ty, Int32Ty);
  return F;
}

static Value *callAppendStringN(IRBuilder<> &Builder, Value *Desc, Value *Str,
                                         Value *Length, bool isLast) {
  auto IsLastInt32 = Builder.getInt32(isLast);
  auto Fn = getFnAppendStringN(Builder);
  return Builder.CreateCall(Fn, {Desc, Str, Length, IsLastInt32});
}

static Value *appendArg(IRBuilder<> &Builder, Value *Desc, Value *Arg, bool IsLast)
{
    auto Arg0 = fitArgInto64Bits(Builder, Arg);
    auto Zero = Builder.getInt64(0);
    return callAppendArgs(Builder, Desc, 1, Arg0,
                          Zero, Zero, Zero, Zero, Zero, Zero, IsLast);
}

static Value *appendString(IRBuilder<> &Builder,
                           const TargetLibraryInfo *TLI,
                           Value *Desc, Value *Arg, bool IsLast)
{
  auto M = Builder.GetInsertBlock()->getModule();
  auto DL = M->getDataLayout();
  auto Length = llvm::emitStrLen(Arg, Builder, DL, TLI);
  Length = Builder.CreateAdd(Length, Builder.getInt64(1));
  return callAppendStringN(Builder, Desc, Arg, Length, IsLast);
}

static Value *processArg(IRBuilder<> &Builder,
                         const TargetLibraryInfo *TLI,
                         Value *Desc, Value *Arg,
                         bool SpecIsCString, bool IsLast) {
  if (SpecIsCString && isCString(Arg)) {
    return appendString(Builder, TLI, Desc, Arg, IsLast);
  }
  return appendArg(Builder, Desc, Arg, IsLast);
}

static void locateCStrings(SparseBitVector<64> &BV, Value *Fmt) {
  StringRef Str;
  if (!getConstantStringInfo(Fmt, Str) || Str.empty())
    return;

  static const char ConvSpecifiers[] = "diouxXfFeEgGaAcspn";
  size_t SpecPos = 0;
  // Skip the first argument, the format string.
  unsigned ArgIdx = 1;

  while ((SpecPos = Str.find_first_of('%', SpecPos)) != StringRef::npos) {
    if (Str[SpecPos + 1] == '%') {
      SpecPos += 2;
      continue;
    }
    auto SpecEnd = Str.find_first_of(ConvSpecifiers, SpecPos);
    if (SpecEnd == StringRef::npos)
      return;
    auto Spec = Str.slice(SpecPos, SpecEnd + 1);
    ArgIdx += Spec.count('*');
    if (Str[SpecEnd] == 's') {
      BV.set(ArgIdx);
    }
    SpecPos = SpecEnd + 1;
    ++ArgIdx;
  }
}

static void lowerPrintf(IRBuilder<> &Builder, const TargetLibraryInfo *TLI,
                        CallInst *CI) {
  auto NumOps = CI->getNumArgOperands();
  assert(NumOps >= 1);

  Builder.SetInsertPoint(CI);
  Builder.SetCurrentDebugLocation(CI->getDebugLoc());

  auto Fmt = CI->getArgOperand(0);
  SparseBitVector<64> SpecIsCString;
  locateCStrings(SpecIsCString, Fmt);

  auto Desc =
      callPrintfBegin(Builder, Builder.getIntN(64, 0));
  Desc = appendString(Builder, TLI, Desc, Fmt, NumOps == 1);

  // Write out the actual arguments following the format string.
  for (unsigned int i = 1; i != NumOps; ++i) {
    bool IsLast = i == NumOps - 1;
    bool IsCString = SpecIsCString.test(i);
    Desc = processArg(Builder, TLI, Desc, CI->getArgOperand(i), IsCString, IsLast);
  }

  Desc = Builder.CreateTrunc(Desc, Builder.getInt32Ty());
  CI->replaceAllUsesWith(Desc);
}

static void collectPrintfsFromModule(SmallVectorImpl<CallInst *> &Printfs,
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

bool AMDGPUPrintfLowering::runOnModule(Module &M) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);
  auto TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  SmallVector<CallInst *, 32> Printfs;
  collectPrintfsFromModule(Printfs, M);
  if (Printfs.empty()) {
    return false;
  }

  for (auto P : Printfs) {
    lowerPrintf(Builder, TLI, P);
    P->eraseFromParent();
  }

  return true;
}
