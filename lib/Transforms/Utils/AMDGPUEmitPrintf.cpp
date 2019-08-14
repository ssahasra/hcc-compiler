//===- AMDGPUEmitPrintf.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility function to lower a printf call into a series of device
// library calls on the AMDGPU target.
//
// WARNING: This file knows about certain library functions. It recognizes them
// by name, and hardwires knowledge of their semantics.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-emit-printf"

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
  return M->getOrInsertFunction("__ockl_printf_begin", Int64Ty, Int64Ty);
}

static Value *callPrintfBegin(IRBuilder<> &Builder, Value *Version) {
  auto Fn = getFnPrintfBegin(Builder);
  return Builder.CreateCall(Fn, Version);
}

static FunctionCallee getFnAppendArgs(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  return M->getOrInsertFunction("__ockl_printf_append_args", Int64Ty, Int64Ty,
                                  Int32Ty,
                                  Int64Ty, Int64Ty, Int64Ty, Int64Ty,
                                  Int64Ty, Int64Ty, Int64Ty, Int32Ty);
}

static Value *callAppendArgs(IRBuilder<> &Builder, Value *Desc, int NumArgs,
                             Value *Arg0, Value *Arg1, Value *Arg2, Value *Arg3,
                             Value *Arg4, Value *Arg5, Value *Arg6, bool IsLast) {
  auto IsLastValue = Builder.getInt32(IsLast);
  auto NumArgsValue = Builder.getInt32(NumArgs);
  return Builder.CreateCall(getFnAppendArgs(Builder),
                            {Desc, NumArgsValue, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, IsLastValue});
}

static Value *appendArg(IRBuilder<> &Builder, Value *Desc, Value *Arg, bool IsLast)
{
    auto Arg0 = fitArgInto64Bits(Builder, Arg);
    auto Zero = Builder.getInt64(0);
    return callAppendArgs(Builder, Desc, 1, Arg0,
                          Zero, Zero, Zero, Zero, Zero, Zero, IsLast);
}

static Value* getStrlenWithNull(IRBuilder<> &Builder, Value *Ptr) {
  auto *Prev = Builder.GetInsertBlock();
  Module *M = Prev->getModule();

  auto CharZero = Builder.getInt8(0);
  auto One = Builder.getInt64(1);
  auto Zero = Builder.getInt64(0);
  auto Int64Ty = Builder.getInt64Ty();

  // Check for null pointer
  BasicBlock *Return = nullptr;
  if (Prev->getTerminator()) {
      Return = Prev->splitBasicBlock(Builder.GetInsertPoint(), Prev->getName() + ".strlen.return");
      Prev->getTerminator()->eraseFromParent();
  } else {
      Return = BasicBlock::Create(M->getContext(), Prev->getName() + ".strlen.return", Prev->getParent());
  }
  BasicBlock *While = BasicBlock::Create(M->getContext(), Prev->getName() + ".strlen.while",
                                         Prev->getParent(), Return);
  BasicBlock *WhileDone = BasicBlock::Create(M->getContext(), Prev->getName() + ".strlen.while.done", Prev->getParent(), Return);

  // Don't compute length if the pointer is null
  Builder.SetInsertPoint(Prev);
  auto CmpNull = Builder.CreateICmpEQ(Ptr, Constant::getNullValue(Ptr->getType()));
  BranchInst::Create(Return, While, CmpNull, Prev);

  // A while-loop that checks for end of string.
  Builder.SetInsertPoint(While);
  auto Str = Builder.CreateAddrSpaceCast(Ptr, Builder.getInt8PtrTy());

  auto PHI = Builder.CreatePHI(Str->getType(), 2);
  PHI->addIncoming(Str, Prev);
  auto GEPNext = Builder.CreateGEP(PHI, One);
  PHI->addIncoming(GEPNext, While);

  // Loop-exit
  auto Data = Builder.CreateLoad(PHI);
  auto Cmp = Builder.CreateICmpEQ(Data, CharZero);
  Builder.CreateCondBr(Cmp, WhileDone, While);
  // Pointer arithmetic at loop exit:
  // len = end - start + 1 ... this includes the null terminator
  Builder.SetInsertPoint(WhileDone, WhileDone->begin());
  auto Begin = Builder.CreatePtrToInt(Ptr, Int64Ty);
  auto End = Builder.CreatePtrToInt(PHI, Int64Ty);
  auto Len = Builder.CreateSub(End, Begin);
  Len = Builder.CreateAdd(Len, One);

  // Join with the early return. Length is zero if the pointer was
  // null. Strictly speaking, the zero does not matter since
  // __ockl_printf_append_string_n ignores the length if the pointer
  // is null.
  BranchInst::Create(Return, WhileDone);
  Builder.SetInsertPoint(Return, Return->begin());
  PHI = Builder.CreatePHI(Len->getType(), 2);
  PHI->addIncoming(Len, WhileDone);
  PHI->addIncoming(Zero, Prev);

  return PHI;
}

static FunctionCallee getFnAppendStringN(IRBuilder<> &Builder) {
  auto Int64Ty = Builder.getInt64Ty();
  auto CharPtrTy = Builder.getInt8PtrTy();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  return M->getOrInsertFunction("__ockl_printf_append_string_n", Int64Ty, Int64Ty, CharPtrTy, Int64Ty, Int32Ty);
}

static Value *callAppendStringN(IRBuilder<> &Builder, Value *Desc, Value *Str,
                                         Value *Length, bool isLast) {
  auto IsLastInt32 = Builder.getInt32(isLast);
  auto Fn = getFnAppendStringN(Builder);
  return Builder.CreateCall(Fn, {Desc, Str, Length, IsLastInt32});
}

static Value *appendString(IRBuilder<> &Builder,
                           Value *Desc, Value *Arg, bool IsLast)
{
  auto Length = getStrlenWithNull(Builder, Arg);
  return callAppendStringN(Builder, Desc, Arg, Length, IsLast);
}

static Value *processArg(IRBuilder<> &Builder,
                         Value *Desc, Value *Arg,
                         bool SpecIsCString, bool IsLast) {
  if (SpecIsCString && isCString(Arg)) {
    return appendString(Builder, Desc, Arg, IsLast);
  }
  return appendArg(Builder, Desc, Arg, IsLast);
}

static void locateCStrings(SparseBitVector<8> &BV, Value *Fmt) {
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

Value* llvm::emitAMDGPUPrintfCall(IRBuilder<> &Builder, ArrayRef<Value*> Args) {
  auto NumOps = Args.size();
  assert(NumOps >= 1);

  auto Fmt = Args[0];
  SparseBitVector<8> SpecIsCString;
  locateCStrings(SpecIsCString, Fmt);

  auto Desc =
      callPrintfBegin(Builder, Builder.getIntN(64, 0));
  Desc = appendString(Builder, Desc, Fmt, NumOps == 1);

  // Write out the actual arguments following the format string.
  for (unsigned int i = 1; i != NumOps; ++i) {
    bool IsLast = i == NumOps - 1;
    bool IsCString = SpecIsCString.test(i);
    Desc = processArg(Builder, Desc, Args[i], IsCString, IsLast);
  }

  return Builder.CreateTrunc(Desc, Builder.getInt32Ty());
}
