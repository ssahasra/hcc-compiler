#include "llvm/ADT/LoopWalker.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Scalar.h"

#define DEBUG_TYPE "eliminate-irreducibility"

using namespace llvm;
using namespace llvm::PatternMatch;

using BBPredicates = DenseMap<BasicBlock *, Value *>;
using PredMap = DenseMap<BasicBlock *, BBPredicates>;
using BBVector = SmallVector<BasicBlock *, 8>;
using BBSet = SmallPtrSet<BasicBlock *, 8>;
using BBSetVector = SetVector<BasicBlock *, BBVector, BBSet>;

namespace {
struct EliminateIrreducibilityPass : public FunctionPass {
  static char ID;
  EliminateIrreducibilityPass()
      : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &F);
};
} // end anonymous namepsace

char EliminateIrreducibilityPass::ID = 0;

static RegisterPass<EliminateIrreducibilityPass>
X("make-reducible", "Eliminate irreducible cycles",
  false /* Only looks at CFG */,
  false /* Analysis Pass */);

Pass *llvm::createEliminateIrreducibilityPass() {
  return new EliminateIrreducibilityPass();
}

/*
template <class LoopInfoT>
static void makeReducible(LoopWalker<LoopInfoT> &G,
                          std::vector<typename LoopWalker<LoopInfoT>::BlockT *> &Nodes) {
  using BlockT = typename LoopWalker<LoopInfoT>::BlockT;
  SmallPtrSet<BlockT*, 8> Blocks;
  SmallVector<BlockT*, 8> Headers;
  for (auto N : Nodes) {
    Blocks.insert(N);
  }

  for (auto I : Nodes) {
    for (const auto P : predecessors(I)) {
      if (!Blocks.count(P) && G.isIncomingPred(P)) {
        Headers.push_back(I);
        break;
      }
    }
  }

  dbgs() << "Irreducible loop: (";
  for (auto H : Headers) {
    dbgs() << " " << H->getName();
  }
  dbgs() << ")";
  for (auto I : Nodes) {
    dbgs() << " " << I->getName();
  }
  dbgs() << "\n";
}

*/

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

static void squeeze(Function &F, const BBSetVector &Headers) {
  auto &Context = F.getContext();
  auto BoolTrue = ConstantInt::getTrue(Context);
  auto BoolFalse = ConstantInt::getFalse(Context);
  PredMap Predicates;
  BBSetVector Predecessors;
  for (auto Header : Headers) {
    for (auto Predecessor : predecessors(Header)) {
      // if (!isIncoming(CI, Predecessor, Header)) {
      //   continue;
      // }
      Predecessors.insert(Predecessor);
      auto Branch = cast<BranchInst>(Predecessor->getTerminator());
      if (!Branch->isConditional()) {
        Predicates[Header][Predecessor] = BoolTrue;
        continue;
      }
      if (Branch->getSuccessor(0) == Header) {
        Predicates[Header][Predecessor] = Branch->getCondition();
        continue;
      }
      Predicates[Header][Predecessor] = invert(Branch->getCondition());
    }
  }

  dbgs() << "Found headers:";
  for (auto H : Headers) {
    dbgs() << " " << H->getName();
  }
  dbgs() << "\n";

  dbgs() << "Found predecessors:";
  for (auto P : Predecessors) {
    dbgs() << " " << P->getName();
  }
  dbgs() << "\n";

    for (auto Predecessor : Predecessors) {
    auto Branch = cast<BranchInst>(Predecessor->getTerminator());
    if (Branch->getNumSuccessors() != 1) {
      assert(Branch->getNumSuccessors() == 2);
      auto Succ0 = Branch->getSuccessor(0);
      auto Succ1 = Branch->getSuccessor(1);
      if (Headers.count(Succ0) && !Headers.count(Succ1)) {
        Predicates[Succ0][Predecessor] = BoolTrue;
      } else if (Headers.count(Succ1) && !Headers.count(Succ0)) {
        Predicates[Succ1][Predecessor] = BoolTrue;
      }
    }
    for (auto Header : Headers) {
      if (Predicates[Header].count(Predecessor) == 0) {
        Predicates[Header][Predecessor] = BoolFalse;
      }
    }
  }

  auto PreHeader = BasicBlock::Create(Context, "CycleHeader", &F);

  assert(Predecessors.size() > 1);

  for (auto Predecessor : Predecessors) {
    auto Branch = cast<BranchInst>(Predecessor->getTerminator());
    if (!Branch->isConditional()) {
      Branch->setSuccessor(0, PreHeader);
      continue;
    }
    auto Succ0 = Branch->getSuccessor(0);
    auto Succ1 = Branch->getSuccessor(1);
    if (Headers.count(Succ0)) {
      if (Headers.count(Succ1)) {
        Branch->eraseFromParent();
        BranchInst::Create(PreHeader, Predecessor);
        continue;
      }
      Branch->setSuccessor(0, PreHeader);
      continue;
    }
    assert(Headers.count(Succ1));
    Branch->setSuccessor(1, PreHeader);
  }

  for (auto Header : Headers) {
    while (!Header->empty() && isa<PHINode>(Header->begin())) {
      auto Phi = cast<PHINode>(Header->begin());
      Phi->moveBefore(*PreHeader, PreHeader->begin());

      for (auto Predecessor : Predecessors) {
        if (-1 == Phi->getBasicBlockIndex(Predecessor)) {
          Phi->addIncoming(UndefValue::get(Phi->getType()), Predecessor);
        }
      }
    }
  }

  BBPredicates Guards;
  for (int i = 0, e = Headers.size() - 1; i != e; ++i) {
    auto Header = Headers[i];
    auto Phi = PHINode::Create(BoolTrue->getType(), Predecessors.size(), "", PreHeader);
    Guards[Header] = Phi;
    dbgs() << "Guard for " << Header->getName() << ": ";
    auto Preds = Predicates[Header];
    for (auto P : Preds) {
      dbgs() << " " << P.first->getName();
    }
    dbgs() << "\n";
    Predicates.erase(Header);
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
  int i = 0;
  for (int e = Headers.size() - 2; i != e; ++i) {
    auto Header = Headers[i];
    auto Next = BasicBlock::Create(Context, "CycleHeader", &F);
    BranchInst::Create(Header, Next, Guards[Header], GuardBlock);
    GuardBlock = Next;
  }
  BranchInst::Create(Headers[i], Headers[i+1], Guards[Headers[i]], GuardBlock);
}

static void makeReducible(Function &F, LoopWalker<LoopInfo> &G,
                          std::vector<BasicBlock *> &Nodes) {
  BBSetVector Headers;
  BBSetVector Blocks;

  for (auto N : Nodes) {
    Blocks.insert(N);
  }

  for (auto I : Nodes) {
    for (const auto P : predecessors(I)) {
      if (!Blocks.count(P) && G.isIncomingPred(P)) {
        Headers.insert(I);
        break;
      }
    }
  }

  squeeze(F, Headers);
}

static void makeReducible(Function &F, LoopWalker<LoopInfo> &G) {
  for (auto Scc = scc_begin(G); !Scc.isAtEnd(); ++Scc) {
    if (Scc->size() < 2)
      continue;
    std::vector<BasicBlock*> Blocks;
    for (auto N : *Scc) {
      Blocks.push_back(N.BB);
    }
    makeReducible(F, G, Blocks);
  }
}

bool EliminateIrreducibilityPass::runOnFunction(Function &F) {
  dbgs() << "===== Function: " << F.getName() << "\n";
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  auto Loops = LI.getLoopsInPreorder();
  for (auto L : Loops) {
    LoopWalker<LoopInfo> G{&LI, L->getHeader()};
    makeReducible(F, G);
  }
  LoopWalker<LoopInfo> G{&LI, &F.getEntryBlock()};
  makeReducible(F, G);

  return true;
}
