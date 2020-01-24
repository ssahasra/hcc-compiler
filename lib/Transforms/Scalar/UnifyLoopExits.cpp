#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Scalar.h"

#define DEBUG_TYPE "unify-loop-exits"

using namespace llvm;
using namespace llvm::PatternMatch;

using BBPredicates = DenseMap<BasicBlock *, Value *>;
using PredMap = DenseMap<BasicBlock *, BBPredicates>;
using BBVector = SmallVector<BasicBlock *, 8>;
using BBSet = SmallPtrSet<BasicBlock *, 8>;
using BBSetVector = SetVector<BasicBlock *, BBVector, BBSet>;

namespace {
struct UnifyLoopExitsPass : public FunctionPass {
  static char ID;
  UnifyLoopExitsPass()
      : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &F);
};
} // end anonymous namepsace

char UnifyLoopExitsPass::ID = 0;

static RegisterPass<UnifyLoopExitsPass>
X("unify-loop-exits", "Unify loop exits",
  false /* Only looks at CFG */,
  false /* Analysis Pass */);

Pass *llvm::createUnifyLoopExitsPass() {
  return new UnifyLoopExitsPass();
}

static Value *invert(Value *Condition) {
  // First: Check if it's a constant
  if (Constant *C = dyn_cast<Constant>(Condition))
    return ConstantExpr::getNot(C);

  // Second: If the condition is already inverted, return the original value
  Value *NotCondition;
  if (match(Condition, m_Not(m_Value(NotCondition))))
    return NotCondition;

  if (Instruction *Inst = dyn_cast<Instruction>(Condition)) {
    // Third: Check all the users for an invert
    BasicBlock *Parent = Inst->getParent();
    for (User *U : Condition->users())
      if (Instruction *I = dyn_cast<Instruction>(U))
        if (I->getParent() == Parent && match(I, m_Not(m_Specific(Condition))))
          return I;

    // Last option: Create a new instruction
    return BinaryOperator::CreateNot(Condition, "", Parent->getTerminator());
  }

  if (Argument *Arg = dyn_cast<Argument>(Condition)) {
    BasicBlock &EntryBlock = Arg->getParent()->getEntryBlock();
    return BinaryOperator::CreateNot(Condition,
                                     Arg->getName() + ".inv",
                                     EntryBlock.getTerminator());
  }

  llvm_unreachable("Unhandled condition to invert");
}

static void squeeze(const BBSetVector &Predecessors, const BBSetVector &Successors) {
  auto F = Predecessors.front()->getParent();
  auto &Context = F->getContext();
  auto BoolTrue = ConstantInt::getTrue(Context);
  auto BoolFalse = ConstantInt::getFalse(Context);
  PredMap Predicates;

  for (auto Predecessor : Predecessors) {
    auto Branch = cast<BranchInst>(Predecessor->getTerminator());
    for (auto Successor : successors(Predecessor)) {
      if (!Successors.count(Successor)) continue;
      if (!Branch->isConditional()) {
        Predicates[Successor][Predecessor] = BoolTrue;
        continue;
      }
      if (Branch->getSuccessor(0) == Successor) {
        Predicates[Successor][Predecessor] = Branch->getCondition();
        continue;
      }
      Predicates[Successor][Predecessor] = invert(Branch->getCondition());
    }
  }

  for (auto Predecessor : Predecessors) {
    auto Branch = cast<BranchInst>(Predecessor->getTerminator());
    if (Branch->getNumSuccessors() != 1) {
      assert(Branch->getNumSuccessors() == 2);
      auto Succ0 = Branch->getSuccessor(0);
      auto Succ1 = Branch->getSuccessor(1);
      if (Successors.count(Succ0) && !Successors.count(Succ1)) {
        Predicates[Succ0][Predecessor] = BoolTrue;
      } else if (Successors.count(Succ1) && !Successors.count(Succ0)) {
        Predicates[Succ1][Predecessor] = BoolTrue;
      }
    }
    for (auto Successor : Successors) {
      if (Predicates[Successor].count(Predecessor) == 0) {
        Predicates[Successor][Predecessor] = BoolFalse;
      }
    }
  }

  auto PreHeader = BasicBlock::Create(Context, "PreHeader", F);

  assert(Predecessors.size() > 1);

  for (auto Predecessor : Predecessors) {
    auto Branch = cast<BranchInst>(Predecessor->getTerminator());
    if (!Branch->isConditional()) {
      Branch->setSuccessor(0, PreHeader);
      continue;
    }
    auto Succ0 = Branch->getSuccessor(0);
    auto Succ1 = Branch->getSuccessor(1);
    if (Successors.count(Succ0)) {
      if (Successors.count(Succ1)) {
        Branch->eraseFromParent();
        BranchInst::Create(PreHeader, Predecessor);
        continue;
      }
      Branch->setSuccessor(0, PreHeader);
      continue;
    }
    assert(Successors.count(Succ1));
    Branch->setSuccessor(1, PreHeader);
  }

  DenseMap<PHINode*, PHINode*> MovedPhis;
  for (auto Successor : Successors) {
    if (pred_empty(Successor)) {
      while (!Successor->empty() && isa<PHINode>(Successor->begin())) {
        auto Phi = cast<PHINode>(Successor->begin());
        Phi->moveBefore(*PreHeader, PreHeader->begin());

        for (auto Predecessor : Predecessors) {
          if (-1 == Phi->getBasicBlockIndex(Predecessor)) {
            Phi->addIncoming(UndefValue::get(Phi->getType()), Predecessor);
          }
        }
      }
      continue;
    }

    SmallVector<BasicBlock *, 4> MovedPredecessors;
    SmallVector<BasicBlock *, 4> NotPredecessors;
    auto &Preds = Predicates[Successor];
    for (auto P : Preds) {
      if (P.second != BoolFalse)
        MovedPredecessors.push_back(P.first);
      else
        NotPredecessors.push_back(P.first);
    }

    for (auto &RefI : *Successor) {
      auto I = &RefI;
      if (!isa<PHINode>(I)) break;
      auto Phi = cast<PHINode>(I);
      auto NewPhi = PHINode::Create(Phi->getType(), MovedPredecessors.size(), "", PreHeader);
      MovedPhis[Phi] = NewPhi;
      for (auto P : MovedPredecessors) {
        auto V = Phi->removeIncomingValue(P, false);
        NewPhi->addIncoming(V, P);
      }
      assert(Phi->getNumIncomingValues() != 0);
      for (auto P : NotPredecessors) {
        NewPhi->addIncoming(UndefValue::get(Phi->getType()), P);
      }
      assert(NewPhi->getNumIncomingValues() == Preds.size());
    }
  }

  BBPredicates Guards;
  for (int i = 0, e = Successors.size() - 1; i != e; ++i) {
    auto Successor = Successors[i];
    auto Phi = PHINode::Create(BoolTrue->getType(), Predecessors.size(), "", PreHeader);
    Guards[Successor] = Phi;
    auto Preds = Predicates[Successor];
    LLVM_DEBUG(dbgs() << "Guard for " << Successor->getName() << ": ";
               for (auto P : Preds) {
                 dbgs() << " " << P.first->getName();
               }
               dbgs() << "\n");
    Predicates.erase(Successor);
    for (auto P : Preds) {
      Phi->addIncoming(P.second, P.first);
      if (isa<Constant>(P.second)) {
        continue;
      }
      for (auto &OtherPreds : Predicates) {
        auto &X = OtherPreds.second[P.first];
        if (isa<Constant>(X)) {
          continue;
        }
        X = BoolTrue;
      }
    }
  }

  auto GuardBlock = PreHeader;
  if (Successors.size() == 1) {
    auto Successor = Successors[0];
    BranchInst::Create(Successor, GuardBlock);
    for (auto &RefI : *Successor) {
      auto I = &RefI;
      if (!isa<PHINode>(I)) break;
      auto Phi = cast<PHINode>(I);
      assert(MovedPhis.count(Phi));
      Phi->addIncoming(MovedPhis[Phi], GuardBlock);
    }
    return;
  }

  int i = 0;
  if (Successors.size() > 2) {
    for (int e = Successors.size() - 2; i != e; ++i) {
      auto Successor = Successors[i];
      auto Next = BasicBlock::Create(Context, "CycleSuccessor", F);
      assert(Guards.count(Successor));
      BranchInst::Create(Successor, Next, Guards[Successor], GuardBlock);
      for (auto &RefI : *Successor) {
        auto I = &RefI;
        if (!isa<PHINode>(I)) break;
        auto Phi = cast<PHINode>(I);
        assert(MovedPhis.count(Phi));
        Phi->addIncoming(MovedPhis[Phi], GuardBlock);
      }
      GuardBlock = Next;
    }
  }

  BranchInst::Create(Successors[i], Successors[i+1], Guards[Successors[i]], GuardBlock);

  for (int e = i + 2; i != e; ++i) {
    auto Successor = Successors[i];
    for (auto &RefI : *Successor) {
      auto I = &RefI;
      if (!isa<PHINode>(I)) break;
      auto Phi = cast<PHINode>(I);
      assert(MovedPhis.count(Phi));
      Phi->addIncoming(MovedPhis[Phi], GuardBlock);
    }
  }
}

static void unifyLoopExits(const LoopInfo &LI, Loop *L) {
  BBVector ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  BBSetVector Predecessors;
  BBSetVector Successors;
  for (auto P : ExitingBlocks) {
    Predecessors.insert(P);
    for (auto S : successors(P)) {
      auto SL = LI.getLoopFor(S);
      if (SL == L || L->contains(SL)) continue;
      Successors.insert(S);
    }
  }

  LLVM_DEBUG(dbgs() << "Found successor:";
             for (auto H : Successors) {
               dbgs() << " " << H->getName();
             }
             dbgs() << "\n");

  LLVM_DEBUG(dbgs() << "Found predecessors:";
             for (auto P : Predecessors) {
               dbgs() << " " << P->getName();
             }
             dbgs() << "\n");

  squeeze(Predecessors, Successors);
}

bool UnifyLoopExitsPass::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "===== Function: " << F.getName() << "\n");
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  auto Loops = LI.getLoopsInPreorder();
  for (auto L : Loops) {
    if (!L->getParentLoop()) continue;
    if (L->getExitingBlock()) continue;
    unifyLoopExits(LI, L);
  }
  return true;
}
