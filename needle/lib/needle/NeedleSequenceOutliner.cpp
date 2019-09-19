//Author: Apala Guha

#define DEBUG_TYPE "needle"

#include "Common.h"
#include "NeedleSequenceOutliner.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CFLAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <cxxabi.h>

#include <algorithm>
#include <deque>

using namespace llvm;
using namespace needle;
using namespace std;

extern cl::list<std::string> FunctionList;

extern cl::opt<bool> EnableValueLogging;
extern cl::opt<bool> EnableMemoryLogging;
extern cl::opt<bool> EnableSimpleLogging;

extern bool isTargetFunction(const Function &f,
                             const cl::list<std::string> &FunctionList);
extern cl::opt<bool> SimulateDFG;
extern cl::opt<ExtractType> ExtractAs;
extern cl::opt<bool> DisableUndoLog;

void NeedleSequenceOutliner::readSequences() {
    ifstream SeqFile(SeqFilePath.c_str(), ios::in);
    assert(SeqFile.is_open() && "Could not open file");
    string Line;

    int64_t Count = 0;
    for (; getline(SeqFile, Line);) {
        Path P;

        SmallVector<StringRef, 16> Tokens;
        StringRef Temp(Line);
        Temp.split(Tokens, ' ');

        P.Id = Tokens[0];

        P.Freq  = stoull(Tokens[1]);
        P.PType = static_cast<PathType>(stoi(Tokens[2]));

        // Token[3] is the number of instructions in path.
        // Not needed here. It's there in the file for filtering
        // and finding out exactly how many paths we want to
        // analyse.

        move(Tokens.begin() + 4, Tokens.end() - 1, back_inserter(P.Seq));
        Sequences.push_back(P);
        // errs() << *P.Seq.begin() << " " << *P.Seq.rbegin() << "\n";
        Count++;

        if (ExtractAs == path)
            break;

    }
    SeqFile.close();
}

NeedleSequenceOutliner::NeedleSequenceOutliner(std::string S,
                               std::vector<std::unique_ptr<llvm::Module>> &EM)
    : llvm::ModulePass(ID), SeqFilePath(S), ExtractedModules(EM) {
    switch (ExtractAs) {
    case path:
        extractAsChop = false;
        break;
    case braid:
        extractAsChop = true;
        break;
    case sequence:
        extractAsChop = false;
        break;
    }
}

bool NeedleSequenceOutliner::doInitialization(Module &M) {
    readSequences();
    Data.clear();
    return false;
}

bool NeedleSequenceOutliner::doFinalization(Module &M) {
    ofstream Outfile("mwe.stats.txt", ios::out);
    for (auto KV : Data) {
        Outfile << KV.first << " " << KV.second << "\n";
    }
    return false;
}

static StructType *getStructType(SetVector<Value *> &LiveOut, Module *Mod) {
    SmallVector<Type *, 16> LiveOutTypes(LiveOut.size());
    transform(LiveOut.begin(), LiveOut.end(), LiveOutTypes.begin(),
              [](const Value *V) -> Type * { return V->getType(); });
    // Create a packed struct return type
    return StructType::get(Mod->getContext(), LiveOutTypes, true);
}

static bool isBlockInPath(const string &S, const Path &P) {
    return find(P.Seq.begin(), P.Seq.end(), S) != P.Seq.end();
}



static inline void fSliceDFSHelper(
    BasicBlock *BB, DenseSet<BasicBlock *> &FSlice,
    DenseSet<pair<const BasicBlock *, const BasicBlock *>> &BackEdges) {
    FSlice.insert(BB);
    for (auto SB = succ_begin(BB), SE = succ_end(BB); SB != SE; SB++) {
        if (!FSlice.count(*SB) && BackEdges.count(make_pair(BB, *SB)) == 0)
            fSliceDFSHelper(*SB, FSlice, BackEdges);
    }
}

static DenseSet<BasicBlock *>
fSliceDFS(BasicBlock *Begin,
          DenseSet<pair<const BasicBlock *, const BasicBlock *>> &BackEdges) {
    DenseSet<BasicBlock *> FSlice;
    fSliceDFSHelper(Begin, FSlice, BackEdges);
    return FSlice;
}


static void liveInHelper(OrgBlocks &RevTopoChop,
                         OrgBlocks::iterator &BB,
                         SetVector<Value *> &LiveIn,
                         SetVector<Value *> &Globals, Value *Val) {
    if (auto Ins = dyn_cast<Instruction>(Val)) {
        for (auto OI = Ins->op_begin(), EI = Ins->op_end(); OI != EI; OI++) {
            if (auto OIns = dyn_cast<Instruction>(OI)) {

                auto &nextBB = RevTopoChop.ProducerOnTrace(BB, OIns, Ins);
                if (nextBB == RevTopoChop.end()) {
                    LiveIn.insert(OIns);
                    BB.AddLiveIn(OIns);
                }
                
            }  else liveInHelper(RevTopoChop, BB, LiveIn, Globals, *OI);
        }
    } else if (auto CE = dyn_cast<ConstantExpr>(Val)) {
        for (auto OI = CE->op_begin(), EI = CE->op_end(); OI != EI; OI++) {
            assert(!isa<Instruction>(OI) &&
                   "Don't expect operand of ConstExpr to be an Instruction");

            liveInHelper(RevTopoChop, BB, LiveIn, Globals, *OI);
        }
    } else if (auto Arg = dyn_cast<Argument>(Val)) {
        LiveIn.insert(Val);
        BB.AddLiveIn(Val);
        

    } else if (auto GV = dyn_cast<GlobalVariable>(Val)) {
        Globals.insert(GV);

        
    }

    // Constants should just fall through and remain
    // in the trace.
}

static bool checkCall(const Instruction &I, string name) {
    if (isa<CallInst>(&I) && dyn_cast<CallInst>(&I)->getCalledFunction() &&
        dyn_cast<CallInst>(&I)->getCalledFunction()->getName().startswith(name))
        return true;
    return false;
}

static string getOpcodeStr(unsigned int N) {    
    switch (N) {
#define HANDLE_INST(N, OPCODE, CLASS)                                          \
    case N:                                                                    \
        return string(#OPCODE);
#include "llvm/IR/Instruction.def"
    default:
        return ("Unknown");
    }
}

void static generalStats(vector<BasicBlock*> &F, string fname) {



    std::map<std::string, uint64_t> OpcodeCount;
    std::map<std::string, uint64_t> OpcodeWt;
    int nBB = 0;


    SetVector<BasicBlock*> uniqBlks;
    auto nUniqBlks = 0;
    auto nUniqIns = 0;



    for(uint64_t i = 0; i < 100; i++){
        OpcodeCount[getOpcodeStr(i)] = 0;
    }

#define HANDLE_INST(N, OPCODE, CLASS) OpcodeCount[#OPCODE] = 0;
#include "llvm/IR/Instruction.def"
    OpcodeCount["CondBr"] = 0;
    OpcodeCount["Guard"]  = 0;

#define HANDLE_INST(N, OPCODE, CLASS) OpcodeWt[#OPCODE] = 1;
#include "llvm/IR/Instruction.def"


    for (auto &BB : F) {
        nBB ++;

        auto uniqFlag = false;
        if (uniqBlks.count(BB) == 0) { //block first found
            uniqBlks.insert(BB);
            nUniqBlks ++;
            uniqFlag = true;
        }

        for (auto &I : *BB) {

            

            // Also remove the GEPs required to index
            // into LO struct.
            if (auto *SI = dyn_cast<StoreInst>(&I)) {
                if (SI->getMetadata("LO") != nullptr) {
                    continue;
                }
            }

            if (uniqFlag) nUniqIns ++;

            switch (I.getOpcode()) {
#define HANDLE_INST(N, OPCODE, CLASS)                                          \
    case N:                                                                    \
        OpcodeCount[#OPCODE] += 1;                                             \
        break;
#include "llvm/IR/Instruction.def"
            }

            if (auto *BI = dyn_cast<BranchInst>(&I)) {
                if (BI->isConditional()) {
                    OpcodeCount["Br"] -= 1;
                    OpcodeCount["CondBr"] += 1;
                }
            }

            if (checkCall(I, "__guard_func")) {
                OpcodeCount["Call"] -= 1;
                OpcodeCount["Guard"] += 1;
            }
        }
    }

    ofstream Outfile(fname, ios::out);
    uint64_t TotalCount = 0;
    for (auto KV : OpcodeCount) {
        Outfile << KV.first << " " << KV.second << "\n";
        TotalCount += KV.second;
    }
    Outfile << "TotalOpsCount " << TotalCount << "\n";
    Outfile << "NUMBB" << " " << nBB << "\n";
    Outfile << "nUniqBlks" << " " << nUniqBlks << "\n";
    Outfile << "nUniqIns" << " " << nUniqIns << "\n";
    Outfile << "nLiveout 0\n";
    Outfile << "nLivein 0\n";


    Outfile.close();
    return;
}

void NeedleSequenceOutliner::extractHelper(Function *StaticFunc, Function *GuardFunc,
                                   SetVector<Value *> &LiveIn,
                                   SetVector<Value *> &LiveOut,
                                   SetVector<Value *> &Globals,
                                   OrgBlocks &RevTopoChop,
                                   LLVMContext &Context) {

    std::vector<ValMap*> VMap;
    VMap.push_back(new ValMap);

    auto BackEdges = common::getBackEdges(RevTopoChop.back());

    auto handleCallSites = [&VMap, &StaticFunc, &RevTopoChop](CallSite &OrigCS,
                                                CallSite &StaticCS) {
        assert(OrigCS.getCalledFunction() &&
               "We do not support indirect function calls in traces.");
        auto *FTy = OrigCS.getCalledFunction()->getFunctionType();
        auto *Val = OrigCS.getCalledValue();
        auto Name = OrigCS.getCalledFunction()->getName();
        if (RevTopoChop.count(Val, VMap) == 0) {
            Function *ExtFunc =
                Function::Create(FTy, GlobalValue::ExternalLinkage, Name,
                                 StaticFunc->getParent());
            assert(RevTopoChop.count(Val, VMap) == 0 && "Need new values"); 
            VMap.back()->insert(Val, static_cast<Value *>(ExtFunc));
        }
        StaticCS.setCalledFunction(RevTopoChop.find(Val, VMap));
    };

    // Add Globals
    for (auto Val : Globals) {
        auto OldGV = dyn_cast<GlobalVariable>(Val);
        assert(OldGV && "Could not reconvert to Global Variable");
        GlobalVariable *GV = new GlobalVariable(
            *StaticFunc->getParent(), OldGV->getType()->getElementType(),
            OldGV->isConstant(), GlobalValue::ExternalLinkage,
            (Constant *)nullptr, OldGV->getName(), (GlobalVariable *)nullptr,
            OldGV->getThreadLocalMode(), OldGV->getType()->getAddressSpace());
        GV->copyAttributesFrom(OldGV);
        assert(VMap.back()->count(OldGV) == 0 && "Need new values");
        VMap.back()->insert(OldGV, GV);
        // Just set the linkage for the original global variable in
        // case the it was private or something.
        OldGV->setLinkage(GlobalValue::ExternalLinkage);
    }

    //create the map stack first and then initialize iterators into it
    RevTopoChop.SetValueMaps(VMap);


    for (auto IT = RevTopoChop.rbegin(), IE = RevTopoChop.rend(); IT != IE;
         ++IT) {
        
        auto C      = *IT;
        auto *NewBB = BasicBlock::Create(
            Context, StringRef("my_") + C->getName() + StringRef("_") + IT.GetPathPos(), StaticFunc, nullptr);

        //the following assert is not needed anymore because 
        //there maybe multiple copies of the same basic block
        //assert(VMap.count(C) == 0 && "Need new values");
        //assert(IT.count(C, VMap) == 0 && "Need new values");

        IT.insert(C, NewBB);

        for (auto &I : *C) {
            auto phi = dyn_cast<PHINode>(&I);

            if ((IT.IsLiveIn(&I) && !IT.IsLiveOut(&I) && !RevTopoChop.AnyConsumerOnTrace(IT, &I))
                || (phi && (IT == RevTopoChop.rbegin()))) continue;
            auto *NewI = I.clone();
            NewBB->getInstList().push_back(NewI);

            //the following assert is not needed anymore
            //assert(VMap.count(&I) == 0 && "Need new values");
            //assert(IT.count(&I, VMap) == 0 && "Need new values");

            IT.insert(&I, NewI);

            CallSite CS(&I), TraceCS(NewI);
            if (CS.isCall() || CS.isInvoke())
                handleCallSites(CS, TraceCS);
        }
    }



    // Assign names, if you don't have a name,
    // a name will be assigned to you.
    Function::arg_iterator AI = StaticFunc->arg_begin();
    uint32_t VReg             = 0;
    for (auto Val : LiveIn) {
        auto Name = Val->getName();
        if (Name.empty())
            AI->setName(StringRef(string("vr.") + to_string(VReg++)));
        else
            AI->setName(Name + string(".in"));
        // VMap[Val] = AI;
        VMap.back()->insert(Val, &*AI);
        ++AI;
    }




    auto inChop = [&RevTopoChop](const BasicBlock *BB) -> bool {
        return (find(RevTopoChop.begin(), RevTopoChop.end(), BB) !=
                RevTopoChop.end());
    };




    ValueToValueMapTy GlobalPointer;
    for (auto G : Globals) {
        AI->setName(G->getName() + ".in");
        Value *RewriteVal = &*AI++;
        GlobalPointer[G]  = RewriteVal;
        VMap.back()->insert(G, RewriteVal);
    }

    function<void(Value *, Value *)> rewriteHelper;
    rewriteHelper = [&GlobalPointer, &rewriteHelper](Value *I, Value *P) {
        if (auto *CE = dyn_cast<ConstantExpr>(I)) {
            for (auto OI = CE->op_begin(), OE = CE->op_end(); OI != OE; OI++) {
                rewriteHelper(*OI, CE);
            }
        } else if (auto *GV = dyn_cast<GlobalVariable>(I)) {
            dyn_cast<User>(P)->replaceUsesOfWith(GV, GlobalPointer[GV]);
        }

    };

    for (auto &BB : *StaticFunc) {
        for (auto &I : BB) {
            for (auto OI = I.op_begin(), OE = I.op_end(); OI != OE; OI++) {
                rewriteHelper(*OI, &I);
            }
        }
    }



    //the original code maps uses of live-ins, globals, and other values to arguments, and, new values respectively.
    //however, for a sequence, different uses of the same value could be referring to different copies of it. therefore,
    //instead of updating uses of a value, we have search for the definition nearest to the use and see what this definition
    //has been mapped to. we use the value map stack for this.

    for (auto IT = RevTopoChop.begin(), IE = RevTopoChop.end(); IT != IE; ++IT) {

        //walk sequence in reverse topological order
        //for each basic block, find the block it has been mapped to
        auto newBB = IT.find(*IT, VMap);
        assert(newBB && "clone bbl not found\n");


        for (auto &I : **IT) {

            if (dyn_cast<PHINode>(&I)) continue;


            auto newIns = IT.find(&I, VMap);
            assert(newIns && "clone ins not found\n");

            for (auto OI = I.op_begin(), OE = I.op_end(); OI != OE; OI++) {
                Value* oldVal = *OI;
                if ((!dyn_cast<BasicBlock>(oldVal)) && (!dyn_cast<Instruction>(oldVal)) && (LiveIn.count(oldVal) == 0) 
                    && (!dyn_cast<ConstantExpr>(oldVal))) continue;

                if (dyn_cast<TerminatorInst>(&I)) {
                    Value* newVal;

                    auto T = dyn_cast<TerminatorInst>(&I);
                    auto succs = T->successors();
                    if (find(succs, oldVal) != succs.end()) {
                        //branch targets should be search forwards in circular manner
                        //unlike values that are searched backwards
                        newVal = IT.FindBranchTarget(oldVal, VMap);
                    } else {
                        newVal = IT.find(oldVal, VMap);
                    }


                    //some operands are off trace
                    if (!newVal) continue;
                    (dyn_cast<User>(newIns))->replaceUsesOfWith(oldVal, newVal);

                    
                } else if ((dyn_cast<BasicBlock>(oldVal)) || (dyn_cast<Instruction>(oldVal)) || (LiveIn.count(oldVal) > 0)) {
                    auto newVal = IT.find(oldVal, VMap);
                    assert(newVal && "clone operand not found\n");
                    (dyn_cast<User>(newIns))->replaceUsesOfWith(oldVal, newVal);
                } else if (auto CE = dyn_cast<ConstantExpr>(oldVal)) {
                    //map globals and constant expression operands to nearest definitions
                    //generate a new mapping for this constant expression
                    auto updatedCE = CE;
                    int nOperands = CE->getNumOperands();

                    for (int i = 0; i < nOperands; i++) {
                        auto opnd = updatedCE->getOperand(i);
                        if (auto GV = dyn_cast<GlobalVariable>(opnd)) {
                            auto mappedGV = VMap.back()->find(GV);
                            assert(mappedGV && "global variable not mapped\n");
                            auto newCE = cast<ConstantExpr>(updatedCE->getWithOperandReplaced(i, cast<Constant>(mappedGV)));
                            updatedCE = newCE;
                        } else if (auto ACE = dyn_cast<ConstantExpr>(opnd)) {
                            auto mappedCE = IT.find(ACE, VMap);
                            assert(mappedCE && "constant expression not mapped\n");
                            auto newCE = cast<ConstantExpr>(updatedCE->getWithOperandReplaced(i, cast<Constant>(mappedCE)));
                            updatedCE = newCE;
                        }
                    }//each operand of constant expression


                    IT.insert(CE, updatedCE);
                    (dyn_cast<User>(newIns))->replaceUsesOfWith(CE, updatedCE);
                }//constant expression  

                
            }
        }
    }







    // Add return true for last block
    auto *BB = cast<BasicBlock>(RevTopoChop.begin().find(RevTopoChop.front(), VMap));
    BB->getTerminator()->eraseFromParent();
    ReturnInst::Create(Context, ConstantInt::getTrue(Context), BB);

    // Patch branches
    auto insertGuardCall = [&GuardFunc, &Context](BranchInst *CBR,
                                                  bool FreqCondition) {
        auto *Blk  = CBR->getParent();
        Value *Arg = CBR->getCondition();
        Value *Dom = FreqCondition ? ConstantInt::getTrue(Context)
                                   : ConstantInt::getFalse(Context);
        vector<Value *> Params = {Arg, Dom};
        auto CI                = CallInst::Create(GuardFunc, Params, "", Blk);
        // Add a ReadNone+NoInline attribute to the CallSite, which
        // will hopefully help the optimizer.
        CI->setDoesNotAccessMemory();
        CI->setIsNoInline();
    };

    for (auto IT = next(RevTopoChop.begin()), IE = RevTopoChop.end(); IT != IE;
         ++IT) {
        auto *NewBB = cast<BasicBlock>(IT.find(*IT, VMap));
        auto T      = NewBB->getTerminator();

        assert(!isa<IndirectBrInst>(T) && "Not handled");
        assert(!isa<InvokeInst>(T) && "Not handled");
        assert(!isa<ResumeInst>(T) && "Not handled");
        assert(!isa<ReturnInst>(T) && "Should not occur");
        assert(!isa<UnreachableInst>(T) && "Should not occur");

        if (isa<SwitchInst>(T)) {
            assert(false &&
                   "Switch instruction not handled, "
                   "use LowerSwitchPass to convert switch to if-else.");
        } else if (auto *BrInst = dyn_cast<BranchInst>(T)) {
            if (extractAsChop) {
                auto NS = T->getNumSuccessors();
                if (NS == 1) {
                    // Unconditional branch target *must* exist in chop
                    // since otherwith it would not be reachable from the
                    // last block in the path.
                    auto BJ = T->getSuccessor(0);
                    assert(IT.find(BJ, VMap) && "Value not found in map");
                    T->setSuccessor(0, cast<BasicBlock>(IT.find(BJ, VMap)));
                } else {
                    SmallVector<BasicBlock *, 2> Targets;
                    for (unsigned I = 0; I < NS; I++) {
                        auto BL = T->getSuccessor(I);
                        if (inChop(BL) &&
                            BackEdges.count(make_pair(*IT, BL)) == 0) {
                            assert(IT.find(BL, VMap) && "Value not found in map");
                            Targets.push_back(cast<BasicBlock>(IT.find(BL, VMap)));
                        }
                    }

                    assert(Targets.size() &&
                           "At least one target should be in the chop");

                    if (Targets.size() == 2) {
                        BrInst->setSuccessor(0, cast<BasicBlock>(Targets[0]));
                        BrInst->setSuccessor(1, cast<BasicBlock>(Targets[1]));
                    } else {
                        if (inChop(T->getSuccessor(0))) {
                            insertGuardCall(BrInst, true);
                        } else {
                            insertGuardCall(BrInst, false);
                        }
                        T->eraseFromParent();
                        BranchInst::Create(cast<BasicBlock>(Targets[0]), NewBB);
                    }
                }
            } else {
                // Trace will replace the terminator inst with a direct branch
                // to the successor, the DCE pass will remove the comparison and
                // the simplification with merge the basic blocks later.
                if (T->getNumSuccessors() > 0) {
                    

                    auto *SuccBB = *prev(IT);
                    vector<BasicBlock *> Succs(succ_begin(*IT), succ_end(*IT));
                    assert(find(Succs.begin(), Succs.end(), SuccBB) !=
                               Succs.end() &&
                           "Could not find successor!");
                    assert(IT.FindBranchTarget(SuccBB, VMap) && "Successor not found in VMap");
                    if (T->getNumSuccessors() == 2) {
                        if (T->getSuccessor(0) == IT.FindBranchTarget(SuccBB, VMap))
                            insertGuardCall(BrInst, true);
                        else
                            insertGuardCall(BrInst, false);
                    }
                    T->eraseFromParent();
                    BranchInst::Create(cast<BasicBlock>(IT.FindBranchTarget(SuccBB, VMap)), NewBB);
                }
            }
        } else {
            assert(false && "Unknown TerminatorInst");
        }
    }






    auto handlePhis = [&VMap, &RevTopoChop, &BackEdges](PHINode *Phi,
                                                        bool extractAsChop,
                                                        OrgBlocks::iterator& BB) {

        auto nextBB = BB;
        ++nextBB;

        auto NV = Phi->getNumIncomingValues();
        vector<BasicBlock *> ToRemove;
        for (unsigned I = 0; I < NV; I++) {
            auto *Blk = Phi->getIncomingBlock(I);
            auto *Val = Phi->getIncomingValue(I);

            //off-trace
            if (!extractAsChop &&
                (*nextBB != Blk)) {
                ToRemove.push_back(Blk);
                continue;
            }

            // Is this a backedge? Remove the incoming value
            // Is this predicated on a block outside the chop? Remove




            
            
           assert(BB.find(Phi, VMap) &&
                   "New Phis should have been added by Instruction clone");

            auto *NewPhi = cast<PHINode>(BB.find(Phi, VMap));
            assert(nextBB.find(Blk, VMap) && "Value not found in ValMap");
            NewPhi->setIncomingBlock(I, cast<BasicBlock>(nextBB.find(Blk, VMap)));

            // Rewrite the value if it is available in the val map
            // Val may be constants or globals which are not present
            // in the map and don't need to be rewritten.
            if (nextBB.find(Val, VMap)) {
                NewPhi->setIncomingValue(I, nextBB.find(Val, VMap));
            }
        }
        for (auto R : ToRemove) {
            assert(BB.find(Phi, VMap) &&
                   "New Phis should have been added by Instruction clone");
            auto *NewPhi = cast<PHINode>(BB.find(Phi, VMap));
            NewPhi->removeIncomingValue(R, false);
        }
    };

    // Patch the Phis of all the blocks in Topo order
    // apart from the first block (those become inputs)
    for (auto BB = RevTopoChop.begin();
         BB != --RevTopoChop.end(); ++BB) {
        for (auto &Ins : **BB) {
            if (auto *Phi = dyn_cast<PHINode>(&Ins)) {
                handlePhis(Phi, extractAsChop, BB);
            }
        }
    }



    // Get the struct pointer from the argument list,
    // assume that output struct is always last arg
    auto StructPtr = --StaticFunc->arg_end();

    // Store the live-outs to the output struct
    int32_t OutIndex = 0;
    Value *Idx[2];
    Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
    for (auto &L : LiveOut) {

        Value *LO = RevTopoChop.begin().find(L, VMap);
        assert(LO && "Live Out not remapped");

        auto *Block = cast<Instruction>(LO)->getParent();
        Idx[1]      = ConstantInt::get(Type::getInt32Ty(Context), OutIndex);
        GetElementPtrInst *StructGEP = GetElementPtrInst::Create(
            cast<PointerType>(StructPtr->getType())->getElementType(),
            &*StructPtr, Idx, "", Block->getTerminator());

        auto *SI  = new StoreInst(LO, StructGEP, Block->getTerminator());
        MDNode *N = MDNode::get(Context, MDString::get(Context, "true"));
        SI->setMetadata("LO", N);
        OutIndex++;
    }




}

static pair<BasicBlock *, BasicBlock *> getReturnBlocks(Function *F) {
    auto &Ctx           = F->getContext();
    BasicBlock *RetTrue = nullptr, *RetFalse = nullptr;
    ConstantInt *True  = ConstantInt::getTrue(Ctx),
                *False = ConstantInt::getFalse(Ctx);
    /// RetFalse will remain nullptr where the outlined function does not
    /// have any guards which are converted to conditional branches to a
    /// block which returns false.
    for (auto &BB : *F) {
        if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
            if (True == RI->getReturnValue()) {
                RetTrue = &BB;
            } else if (False == RI->getReturnValue()) {
                RetFalse = &BB;
            } else {
                llvm_unreachable("Offload functions should "
                                 "only have true or false return blocks");
            }
        }
    }
    return {RetTrue, RetFalse};
}

/// Live Value logging : This should only be run on offloaded functions
/// it makes assumptions on what things are being returned.
/// 1. Live in logging happens immediately upon function entry
/// 2. Live out logging happens in 2 places,
///     a. success : actual values are dumped out
///     b. fail : partial
/// This approach is to ensure consistency per invocation
void NeedleSequenceOutliner::valueLogging(Function *F) {
    if (!EnableValueLogging)
        return;

    auto Mod     = F->getParent();
    auto DL      = Mod->getDataLayout();
    auto &Ctx    = F->getContext();
    auto *VoidTy = Type::getVoidTy(Ctx);

    BasicBlock *RetTrue = nullptr, *RetFalse = nullptr;
    tie(RetTrue, RetFalse)                   = getReturnBlocks(F);
    /// Live in value logging
    /// First gather all the live in values into a struct
    /// a. Create a struct inside the function
    auto InsertionPt = F->getEntryBlock().getFirstInsertionPt();
    SetVector<Value *> LiveInArgs;
    for (auto AB = F->arg_begin(), AE = F->arg_end(); AB != AE; AB++) {
        LiveInArgs.insert(&*AB);
    }
    auto *StructTy = getStructType(LiveInArgs, Mod);
    auto *LIS      = new AllocaInst(StructTy, "", &*InsertionPt);

    /// b. Stick the live in values into this struct
    int32_t Index = 0;
    Value *Idx[2];
    Idx[0]        = Constant::getNullValue(Type::getInt32Ty(Ctx));
    StoreInst *SI = nullptr;
    for (auto &L : LiveInArgs) {
        Idx[1] = ConstantInt::get(Type::getInt32Ty(Ctx), Index);
        GetElementPtrInst *StructGEP = GetElementPtrInst::Create(
            cast<PointerType>(LIS->getType())->getElementType(), &*LIS, Idx,
            "");
        if (!SI) {
            StructGEP->insertAfter(LIS);
        } else {
            StructGEP->insertAfter(SI);
        }
        SI = new StoreInst(L, StructGEP);
        SI->insertAfter(StructGEP);
        Index++;
    }

    /// c. Write out this struct
    auto *LiveInStructBI =
        new BitCastInst(LIS, PointerType::getInt8PtrTy(Ctx), "dump_cast");
    LiveInStructBI->insertAfter(SI);
    uint64_t LiveInSize = DL.getTypeStoreSize(
        cast<PointerType>(LIS->getType())->getElementType());
    auto *LiveInSz        = ConstantInt::get(Type::getInt64Ty(Ctx), LiveInSize);
    Value *LiveInParams[] = {LiveInStructBI, LiveInSz};
    auto *LiveInDumpFn    = Mod->getOrInsertFunction(
        "__log_in",
        FunctionType::get(
            VoidTy, {Type::getInt8PtrTy(Ctx), Type::getInt64Ty(Ctx)}, false));
    assert(LiveInDumpFn && "Could not insert dump function");

    CallInst::Create(LiveInDumpFn, LiveInParams, "")
        ->insertAfter(LiveInStructBI);

    /// Live out value logging
    /// Will have consistent values for successful invocations of the path
    /// for invocations that fail, struct fields may have garbage values.
    /// Assume last argument is live out struct for offloaded functions
    auto LiveOutStructPtr = --F->arg_end();
    auto *LiveOutStructTy =
        cast<PointerType>(LiveOutStructPtr->getType())->getElementType();
    uint64_t Size         = DL.getTypeStoreSize(LiveOutStructTy);
    auto *LiveOutStructBI = new BitCastInst(
        &*LiveOutStructPtr, PointerType::getInt8PtrTy(Ctx), "dump_cast",
        &*F->getEntryBlock().getFirstInsertionPt());
    auto *Sz = ConstantInt::get(Type::getInt64Ty(Ctx), Size);

    Value *Params[] = {LiveOutStructBI, Sz};
    auto *DumpFn    = Mod->getOrInsertFunction(
        "__log_out",
        FunctionType::get(
            VoidTy, {Type::getInt8PtrTy(Ctx), Type::getInt64Ty(Ctx)}, false));
    assert(DumpFn && "Could not insert dump function");
    CallInst::Create(DumpFn, Params, "", RetTrue->getTerminator());
    if (RetFalse)
        CallInst::Create(DumpFn, Params, "", RetFalse->getTerminator());

    /// Dump a value which indicates whether the iteration was successful

    auto *SuccDumpFn = Mod->getOrInsertFunction(
        "__log_succ", FunctionType::get(VoidTy, {Type::getInt1Ty(Ctx)}, false));
    assert(SuccDumpFn && "Could not insert dump function");
    CallInst::Create(SuccDumpFn, {ConstantInt::getTrue(Ctx)}, "",
                     RetTrue->getTerminator());
    if (RetFalse)
        CallInst::Create(SuccDumpFn, {ConstantInt::getFalse(Ctx)}, "",
                         RetFalse->getTerminator());

    /// Generate a C declaration for the struct being dumped
    auto getFormat = [](uint64_t Sz) -> string {
        switch (Sz) {
        case 64:
            return string("\"%llu\"");
        default:
            return string("\"%lu\"");
        }
    };
    auto structDefXMacro = [&DL, &getFormat](StructType *ST, StringRef Name,
                                             raw_ostream &Out) {
        stringstream Def;
        Def << ("\n\n#define X_FIELDS_" + Name.str() + " \\\n");
        for (uint32_t I = 0; I < ST->getNumElements(); I++) {
            if (I != 0)
                Def << " \\\n";
            auto Sz = DL.getTypeStoreSizeInBits(ST->getElementType(I));
            // TODO : Signed vs Unsigned ?
            Def << ("    X( uint" + to_string(Sz) + "_t, el" + to_string(I) +
                    ", " + getFormat(Sz) + ")");
        }
        Out << (Def.str() + "\n\n");
    };

    error_code EC;
    raw_fd_ostream Out("LogTypes.def", EC, sys::fs::F_None);
    structDefXMacro(StructTy, "LIVEIN", Out);
    structDefXMacro(cast<StructType>(LiveOutStructTy), "LIVEOUT", Out);
    Out.close();
}

/// This function records the value of each memory read from the
/// offloaded function in Topological order. To reduce the number
/// of values instrumented alias analysis is used to only grab the
/// first unique memory location. (address, value) tuples are written
/// out to a file.
void NeedleSequenceOutliner::memoryLogging(Function *F) {
    if (!EnableMemoryLogging)
        return;

    SetVector<LoadInst *> Loads;
    auto &AA            = getAnalysis<AAResultsWrapperPass>(*F).getAAResults();
    auto isAliasingLoad = [&AA, &Loads](LoadInst *LI) -> bool {
        for (auto &L : Loads) {
            if (AA.isMustAlias(MemoryLocation::get(LI), MemoryLocation::get(L)))
                return true;
        }
        return false;
    };

    ReversePostOrderTraversal<Function *> RPOT(F);
    for (auto BB = RPOT.begin(); BB != RPOT.end(); ++BB) {
        for (auto &I : **BB) {
            if (auto *LI = dyn_cast<LoadInst>(&I)) {
                if (!isAliasingLoad(LI)) {
                    Loads.insert(LI);
                }
            }
        }
    }

    /// Add the instrumentation for each non-aliasing load
    auto *Mod     = F->getParent();
    auto &Ctx     = Mod->getContext();
    auto &DL      = Mod->getDataLayout();
    auto *VoidTy  = Type::getVoidTy(Ctx);
    auto *Int64Ty = Type::getInt64Ty(Ctx);
    auto *MLogFn  = Mod->getOrInsertFunction(
        "__mlog",
        FunctionType::get(VoidTy, {Int64Ty, Int64Ty, Int64Ty}, false));

    for (auto &LI : Loads) {
        auto *Ptr      = LI->getPointerOperand();
        auto *AddrCast = new PtrToIntInst(Ptr, Int64Ty);
        errs() << *LI << "\n";

        // TODO : Needs testing for FP stuff > 64 bits?
        uint64_t Sz = DL.getTypeSizeInBits(LI->getType());
        if (Sz > 64) {
            report_fatal_error("Cannot convert larger than 64 bits");
        }

        auto CastOp   = CastInst::getCastOpcode(LI, false, Int64Ty, false);
        auto *ValCast = CastInst::Create(CastOp, LI, Int64Ty);

        ConstantInt *Size = ConstantInt::get(Int64Ty, Sz, false);
        Value *Params[]   = {AddrCast, ValCast, Size};
        AddrCast->insertAfter(LI);
        ValCast->insertAfter(AddrCast);
        CallInst::Create(MLogFn, Params)->insertAfter(ValCast);
    }

    /// Add final instrumentation to indicate end of iteration
    /// with success or fail

    BasicBlock *RetTrue = nullptr, *RetFalse = nullptr;
    tie(RetTrue, RetFalse)                   = getReturnBlocks(F);
    auto *TrueSentinel  = ConstantInt::get(Int64Ty, 0x1, false);
    auto *FalseSentinel = ConstantInt::get(Int64Ty, 0x0, false);
    Value *Params[]     = {TrueSentinel, FalseSentinel, FalseSentinel};
    CallInst::Create(MLogFn, Params, "", RetTrue->getTerminator());
    if (RetFalse) {
        Value *Params[] = {FalseSentinel, FalseSentinel, FalseSentinel};
        CallInst::Create(MLogFn, Params, "", RetFalse->getTerminator());
    }
}

static void getTopoChopHelper(
    BasicBlock *BB, DenseSet<BasicBlock *> &Chop, DenseSet<BasicBlock *> &Seen,
    SetVector<BasicBlock *> &Order,
    DenseSet<pair<const BasicBlock *, const BasicBlock *>> &BackEdges) {
    Seen.insert(BB);
    for (auto SB = succ_begin(BB), SE = succ_end(BB); SB != SE; SB++) {
        if (!Seen.count(*SB) && Chop.count(*SB)) {
            getTopoChopHelper(*SB, Chop, Seen, Order, BackEdges);
        }
    }
    Order.insert(BB);
}

/// TODO : Rewrite this to use FunctionRPOT instead
SetVector<BasicBlock *>
getTopoChop(DenseSet<BasicBlock *> &Chop, BasicBlock *StartBB,
            DenseSet<pair<const BasicBlock *, const BasicBlock *>> &BackEdges) {
    SetVector<BasicBlock *> Order;
    DenseSet<BasicBlock *> Seen;
    getTopoChopHelper(StartBB, Chop, Seen, Order, BackEdges);
    return Order;
}



Function *NeedleSequenceOutliner::extract(PostDominatorTree *PDT, Module *Mod,
                                  OrgBlocks &RevTopoChop,
                                  SetVector<Value *> &LiveIn,
                                  SetVector<Value *> &LiveOut,
                                  SetVector<Value *> &Globals,
                                  DominatorTree *DT, LoopInfo *LI, string Id) {



    auto StartBB = RevTopoChop.rbegin();
    auto LastBB  = RevTopoChop.begin();

    auto BackEdges         = common::getBackEdges(*StartBB);
    auto ReachableFromLast = fSliceDFS(*LastBB, BackEdges);


    auto handlePhiIn = [&LiveIn, &RevTopoChop, &Globals,
                        &StartBB](PHINode *Phi, OrgBlocks::iterator &BB) {

        if (BB == --RevTopoChop.end()) { //the same basic block could appear in multiple positions now
            LiveIn.insert(Phi);
            BB.AddLiveIn(Phi);
        } else {
            for (uint32_t I = 0; I < Phi->getNumIncomingValues(); I++) {

                auto *Blk = Phi->getIncomingBlock(I);
                auto *Val = Phi->getIncomingValue(I);
                
                if (BB != RevTopoChop.end()) {
                    
                    if (auto *VI = dyn_cast<Instruction>(Val)) {
                        if ((RevTopoChop.BlockOnTrace(BB, Blk) != RevTopoChop.end())
                            && (RevTopoChop.ProducerOnTrace(BB, VI, dyn_cast<Instruction> (Phi)) == RevTopoChop.end()))
                        {
                            LiveIn.insert(Val);
                            BB.AddLiveIn(Val);
                            
                        }
                    } else if (auto *AI = dyn_cast<Argument>(Val)) {
                        LiveIn.insert(AI);
                        BB.AddLiveIn(AI);
                        
                        
                    } else if (auto *GV = dyn_cast<GlobalVariable>(Val)) {
                        Globals.insert(GV);
                        
                        
                    }
                }
            }
        }
    };

    
    // Live In Loop
    //TODO: why are live-ins discovered in topological order
    //while live-outs are discovered in reverse order
    for (auto &BB = RevTopoChop.begin(); BB != RevTopoChop.end(); ++BB) {
        for (auto &I : **BB) {

            errs() << "processing for inputs: " << I << "\n";

            if (auto Phi = dyn_cast<PHINode>(&I)) {
                handlePhiIn(Phi, BB);
            } else {
                liveInHelper(RevTopoChop, BB, LiveIn, Globals, &I);
            }
        }
    }


    auto notInChop = [&RevTopoChop](const Instruction *I) -> bool {
        return find(RevTopoChop.begin(), RevTopoChop.end(), I->getParent()) ==
               RevTopoChop.end();
    };

    // Value is a live out only if it is used by an instruction
    // a. Reachable from the last block
    // b. As input itself (Induction Phis)
    // c. ??

    auto isLabelReachable = [&ReachableFromLast](
        const PHINode *Phi, const Instruction *Ins) -> bool {
        for (uint32_t I = 0; I < Phi->getNumIncomingValues(); I++) {
            if (Phi->getIncomingValue(I) == Ins &&
                ReachableFromLast.count(Phi->getIncomingBlock(I)))
                return true;
        }
        return false;
    };

    auto processLiveOut = [&LiveOut, &RevTopoChop, &StartBB, &LastBB, &LiveIn,
                           &notInChop, &DT, &LI, &ReachableFromLast
                           , &isLabelReachable
                            ](  
                                                OrgBlocks::iterator& BB,
                                                Instruction *Ins,
                                                Instruction *UIns) {
    if (RevTopoChop.UseIsReachable(Ins, UIns, BB)) {
            LiveOut.insert(Ins);
            BB.AddLiveOut(Ins);
        }
    };

    // Live Out Loop
    for (auto &BB = RevTopoChop.rbegin(); BB != RevTopoChop.rend(); ++BB) {
        for (auto &I : **BB) {
            // Don't consider Phi nodes in the first block
            // since we are not going to include them in
            // the extracted function anyway.
            if (isa<PHINode>(&I) && BB == StartBB)
                continue;
            if (auto Ins = dyn_cast<Instruction>(&I)) {
                for (auto UI = Ins->use_begin(), UE = Ins->use_end(); UI != UE;
                     UI++) {
                    if (auto UIns = dyn_cast<Instruction>(UI->getUser())) {
                        processLiveOut(BB, Ins, UIns);
                    }
                }
            }
        }
    }

    auto isDefInOutlineBlocks = [&StartBB, &notInChop, &RevTopoChop](Value *Val, TerminatorInst* LastT) -> bool {
        if (isa<Constant>(Val) ||
            (isa<Instruction>(Val) && notInChop(dyn_cast<Instruction>(Val))) ||
            ((isa<PHINode>(Val)) &&
                (RevTopoChop.ProducerOnTrace(RevTopoChop.begin(), dyn_cast<PHINode>(Val), LastT) == --RevTopoChop.end()))) {
            return false;

        }
        return true;
    };

    // If LastT has 2 successors then, evaluate condition inside
    // and create a condition inside the success block to do the
    // same and branch accordingly.

    // If LastT has 1 successor then, the successor is a target
    // of the backedge from LastT, then nothing to do.

    // If the LastT has 0 successor then, there may be a return
    // value to patch.

    auto *LastT = (*LastBB)->getTerminator();

    switch (LastT->getNumSuccessors()) {
    case 2: {
        auto *CBR = dyn_cast<BranchInst>(LastT);
        LiveOut.insert(CBR->getCondition());
        RevTopoChop.begin().AddLiveOut(CBR->getCondition());
    } break;
    case 1:
        break;
    case 0: {
        auto *RT = dyn_cast<ReturnInst>(LastT);
        assert(RT && "Path with 0 successor should have returninst");
        auto *Val = RT->getReturnValue();
        // This Val is added to the live out set only if it
        // is def'ed in the extracted region.
        if (Val != nullptr && isDefInOutlineBlocks(Val, LastT)) {
            LiveOut.insert(Val);
            RevTopoChop.begin().AddLiveOut(Val);
        }
    } break;
    default:
        assert(false && "Unexpected num successors -- lowerswitch?");
    }

    Data["num-live-in"]  = LiveIn.size();
    Data["num-live-out"] = LiveOut.size();
    Data["num-globals"]  = Globals.size();

    auto DataLayoutStr   = Mod->getDataLayout();
    auto TargetTripleStr = (*StartBB)->getParent()->getParent()->getTargetTriple();
    Mod->setDataLayout(DataLayoutStr);
    Mod->setTargetTriple(TargetTripleStr);


    // Bool return type for extracted function
    auto VoidTy = Type::getVoidTy(Mod->getContext());
    auto Int1Ty = IntegerType::getInt1Ty(Mod->getContext());

    std::vector<Type *> ParamTy;
    // Add the types of the input values
    // to the function's argument list
    for (auto Val : LiveIn)
        ParamTy.push_back(Val->getType());

    for (auto &G : Globals) {
        ParamTy.push_back(G->getType());
    }

    auto *BufPtrTy = Type::getInt8PtrTy(Mod->getContext());

    ParamTy.push_back(BufPtrTy);

    auto *StructTy    = getStructType(LiveOut, Mod);
    auto *StructPtrTy = PointerType::getUnqual(StructTy);
    ParamTy.push_back(StructPtrTy);

    FunctionType *StFuncType = FunctionType::get(Int1Ty, ParamTy, false);


    // Create the trace function
    Function *StaticFunc = Function::Create(
        StFuncType, GlobalValue::ExternalLinkage, "__offload_func_" + Id, Mod);

    // Create an external function which is used to
    // model all guard checks. First arg is the condition, second is whether
    // the condition is dominant as true or as false. This
    // guard func is later replaced by a branch and return statement.
    // we use this as placeholder to create a superblock and enable
    // optimizations.
    ParamTy.clear();
    ParamTy                  = {Int1Ty, Int1Ty};
    FunctionType *GuFuncType = FunctionType::get(VoidTy, ParamTy, false);

    // Create the guard function
    Function *GuardFunc = Function::Create(
        GuFuncType, GlobalValue::ExternalLinkage, "__guard_func", Mod);


    common::writeModule(Mod, string("offload_func_prototyped.ll"));

    extractHelper(StaticFunc, GuardFunc, LiveIn, LiveOut, Globals, RevTopoChop,
                  Mod->getContext());

    StripDebugInfo(*Mod);

    common::writeModule((*(RevTopoChop.begin()))->getParent()->getParent(), string("original.ll"));
    common::writeModule(StaticFunc->getParent(), string("offload_extract_clone.ll"));

    // Dumbass verifyModule function returns false if no
    // errors are found. Ref "llvm/IR/Verifier.h":46
    assert(!verifyModule(*Mod, &errs()) && "Module verification failed!");

    return StaticFunc;
}

static SetVector<BasicBlock *>
getTraceBlocks(Path &P, map<string, BasicBlock *> &BlockMap) {

    SetVector<BasicBlock *> RPath;
    for (auto RB = P.Seq.rbegin(), RE = P.Seq.rend(); RB != RE; RB++) {
        if (BlockMap.count(*RB) == 0)
            errs() << "Missing :" << *RB << "\n";
        assert(BlockMap.count(*RB) && "Path does not exist");
        RPath.insert(BlockMap[*RB]);
    }
    return RPath;
}

static uint32_t getMaxUndoSize(SmallVector<Module *, 4> Modules) {
    uint32_t R = 64;

    for (auto Mod : Modules) {
        for (auto GI = Mod->global_begin(), GE = Mod->global_end(); GI != GE;
             GI++) {
            if (GI->getName().startswith("__undo_num_stores_")) {
                auto A = cast<ConstantInt>(
                             cast<GlobalVariable>(GI)->getInitializer())
                             ->getValue();
                uint32_t Val = A.getLimitedValue();
                R            = Val > R ? Val : R;
            }
        }
    }

    return R;
}

static void FindPhiMapHelper(SetVector<BasicBlock*>& path, BasicBlock* merge, BasicBlock* avoid, SetVector<BasicBlock*>& visited) {

    if (!path.empty()) {
        BasicBlock* bbl = path.back();
        if (bbl == merge) return;
        if (bbl == avoid) {
            path.pop_back();
            return;
        }

        for (auto S = succ_begin(bbl), E = succ_end(bbl); S != E; S++) {

            auto next = *S;
            if (next == avoid) continue;

            //avoid loops
            if (visited.count(next) != 0) continue;
            path.insert(next);
            visited.insert(next);
            if (next == merge) return;
            

            //if we reached merge we are done
            FindPhiMapHelper(path, merge, avoid, visited);
            if (path.back() == merge) return;


        }//for each child

        path.pop_back();

    }//process stack of basic blocks

}//FindPhiMapHelper



static Value* FindPhiMap(Value* val, BasicBlock* merge, BasicBlock* avoid){ //avoid finding paths through success block
                                                                            //as live-out will be overwritten

    auto ins = dyn_cast<Instruction>(val);
    assert(ins && "live-out should be an instruction\n");

    SetVector<BasicBlock*> path;
    SetVector<BasicBlock*> visited;
    path.insert(ins->getParent());
    visited.insert(ins->getParent());

    FindPhiMapHelper(path, merge, avoid, visited);

    assert(!path.empty() && path.back() == merge && "path to merge block not found\n");

    //path should have the sequence leading upto merge now
    //walk it and map the original value
    auto latestMap = val;

    for (auto &BB: path) {
        //ignore phi nodes at beginning of first block
        if (BB == path[0]) continue;

        for (auto &I: *BB) {
          


            //check the phi nodes of each block
            //phi nodes are always at beginning of block
            auto Phi = dyn_cast<PHINode>(&I);
            if (Phi == NULL) break;



            //check if latest map is an operand in this phi node
            auto NV = Phi->getNumIncomingValues();
            for (unsigned i = 0; i < NV; i++) {
                auto *Val = Phi->getIncomingValue(i);

                if (Val == latestMap) {
                    latestMap = &I;
                    break;
                }

            }//find latest map
        }//for each instruction in a block
    }//for each block in path

    return latestMap;

}//FindPhiMap

static void AvailableAnalyze(Value* val, Value* opnd, queue<BasicBlock*>& q, SetVector<BasicBlock*>& available, SetVector<BasicBlock*>& visited, SetVector<BasicBlock*>& bottoms
                            , BasicBlock* Merge) {

    bool change = true;
    
    while(change) {

        q.push(Merge);
        change = false;

        while (!q.empty()) {

            auto bbl = q.front();
            q.pop();

            //value should be defined in this block or should be available at all predecessors' end
            if (dyn_cast<Instruction>(val)->getParent() != bbl) {
                auto PB = pred_begin(bbl);
                for (auto PE = pred_end(bbl); PB != PE; PB++) {

                    auto prev = *PB;
                    if (bottoms.count(prev) == 0) break;
                    
                }//for each predecessor

                if (PB != pred_end(bbl)) continue;

            }//check all predecessors

            auto I = bbl->begin();
            for (; I != bbl->end(); ++I) {

                //cut off by re-definition of operand
                if (&*I == opnd) break;

                auto phi = dyn_cast<PHINode>(I);
                if (!phi) continue;

                auto NV = phi->getNumIncomingValues();
                unsigned i = 0;
                for (; i < NV; i++) {
                    auto incoming = phi->getIncomingValue(i);
                    if (incoming == val) break;
                }//for each incoming value

                if (i != NV) break;

            }//for each instruction if it is a potential use
            if (I != bbl->end()) continue;

            if(bottoms.count(bbl) == 0) {
                bottoms.insert(bbl);
                change = true;
            }
            

            for (auto SB = succ_begin(bbl), SE = succ_end(bbl); SB != SE; SB++) {

                auto next = *SB;

                //if (visited.count(next) != 0) continue;
                available.insert(next);
                q.push(next);
                visited.insert(next);
                
            }//for each successor

        }//if there are more items to process

    }//while change


}//LivenessAnalyze

static void LivenessAnalyze(Value* val, queue<BasicBlock*>& q, SetVector<BasicBlock*>& liveBlocks, SetVector<BasicBlock*>& visited) {


    while (!q.empty()) {

        auto bbl = q.front();
        q.pop();

        auto I = bbl->rbegin();
        for (; I != bbl->rend(); ++I) {
            if (&*I == val) break;
        }//for each instruction in reverse order
        if (I != bbl->rend()) continue;

        liveBlocks.insert(bbl);

        for (auto PB = pred_begin(bbl), PE = pred_end(bbl); PB != PE; PB++) {
            auto prev = *PB;

            if (visited.count(prev) != 0)  continue;
            q.push(prev);
            visited.insert(prev);
            
        }//for each predecessor

    }//while there are more items to process


}//LivenessAnalyze

static PHINode* PropagatePhi(PHINode* Phi, BasicBlock* Merge, Value* Val, BasicBlock* lastBB, Instruction* UserInst, BasicBlock* UserBB) {
    //check if the phi is properly propagated
    //first check where the original value is live
    queue<BasicBlock*> q;
    SetVector<BasicBlock*> visited;
    SetVector<BasicBlock*> liveBlocks;

    q.push(UserBB);
    visited.insert(UserBB);

    LivenessAnalyze(Val, q, liveBlocks, visited);

    //now check where the new phi is available
    SetVector<BasicBlock*> available;
    SetVector<BasicBlock*> bottoms;
    SetVector<BasicBlock*> visited2;

    AvailableAnalyze(Phi, Val, q, available, visited2, bottoms, UserBB);


    //both values should not be available at the top of any block
    for (auto &bottom: bottoms) {
        

        //check each successor
        for (auto SB = succ_begin(bottom), SE = succ_end(bottom); SB != SE; SB++) { 

            auto tops = *SB;
            if (liveBlocks.count(tops) == 0) continue;

            //if both values are available, then one supercedes
            //by inserting a new phi. but the new phi should be inserted
            //in the top block where this occurs
            auto *newPhi   = PHINode::Create(Val->getType(), 0, "", &*(tops->begin()));
            bool dominated = true;
            for (auto PB = pred_begin(tops), PE = pred_end(tops); PB != PE; PB++) { 
                auto prev = *PB;
                if (bottoms.count(prev) != 0) {
                    newPhi->addIncoming(Phi, prev);
                } else {
                    newPhi->addIncoming(Val, prev);
                    dominated = false;
                }
            }

            //blocks has been processed
            liveBlocks.remove(tops);


            if (dominated) {
                newPhi->eraseFromParent();
                continue;
            }


            auto UserPhi = dyn_cast<PHINode>(UserInst);
            assert(UserPhi && "user has to be a phi node\n");

            auto NV = UserPhi->getNumIncomingValues();
            unsigned i = 0;
            for (; i < NV; i++) {
                auto incoming = UserPhi->getIncomingValue(i);
                if (incoming != Val) continue;
                UserPhi->setIncomingValue(i, newPhi);
            }//for each incoming value

            return newPhi;

        }//for each successor

                     

    }

    return NULL;
}//PropagatePhi

static void writeCFG(Function& F, string Name, OrgBlocks& Blocks, BasicBlock* SSplit, bool verbose=false) {
    error_code EC;
    raw_fd_ostream File(Name, EC, sys::fs::OpenFlags::F_RW);

    File << "digraph {\n";

    int counter = 0;
    int nameCtr = 0;
    map<BasicBlock*, int> bbId;

    for (auto &BB: F) {

        counter ++;
        bbId.insert(pair<BasicBlock*, int>(&BB, counter));

        File << counter << " [label=\"" << BB.getName() << ":\n";

        if (verbose) {
            for (auto &I:BB) {
                if (I.hasName()) File << I.getName();
                else {
                    nameCtr ++;
                    I.setName(string("needle.")+to_string(nameCtr));
                    File << I.getName();
                }

                auto phi = dyn_cast<PHINode>(&I);
                if (phi) File << " = phi(";
                else File << " = f(";

                

                for (Use &UI:(&I)->operands()) {

                    auto OI = UI.get();

                    if (OI->hasName()) File << OI->getName() << ", ";
                    else {
                        nameCtr ++;
                        OI->setName(string("needle.")+to_string(nameCtr));
                        File << OI->getName() << ", ";
                    }

                }//for each operand

                File << ")\n";
            }//for each inst
        }//instructions


        File << "\"]\n";

    }


    for (auto &BB: F) {
        for (auto S = succ_begin(&BB), E = succ_end(&BB); S != E; S++) {
            auto src = bbId.find(&BB)->second;
            auto dst = bbId.find(*S)->second;
            File << src << "->" << dst << "\n";
        }
    }

    //color code blocks
    for (auto &BB:Blocks) {
        int id = bbId.find(BB)->second;
        File << id << "[color=red]\n";
    }
    int id = bbId.find(SSplit)->second;
    File << id << "[color=red]\n";




    File << "}\n";

    File.close();
}

static void instrument(Function &F, OrgBlocks &Blocks,
                       FunctionType *OffloadTy, SetVector<Value *> &LiveIn,
                       SetVector<Value *> &LiveOut, SetVector<Value *> &Globals,
                       DominatorTree *DT, string &Id) {





    if (Blocks.size() == 1) {
        auto *B = Blocks.front();
        auto *R = B->splitBasicBlock(B->getTerminator(), "unit.split");
        Blocks.prepend(R);
    }

    // Setup Basic Control Flow
    auto StartBB = *Blocks.rbegin();
    auto LastBB = *Blocks.begin();
    auto BackEdges         = common::getBackEdges(StartBB);
    auto ReachableFromLast = fSliceDFS(LastBB, BackEdges);
    auto &Ctx              = F.getContext();
    auto *Mod              = F.getParent();
    auto *Int64Ty          = Type::getInt64Ty(Ctx);
    auto *Int32Ty          = Type::getInt32Ty(Ctx);
    auto *VoidTy           = Type::getVoidTy(Ctx);
    auto *Success          = BasicBlock::Create(Ctx, "offload.true", &F);

    if (EnableValueLogging || EnableMemoryLogging || EnableSimpleLogging) {
        CallInst::Create(Mod->getOrInsertFunction(
                             "__success", FunctionType::get(VoidTy, {}, false)),
                         {}, "", Success);
    }

    auto *Fail        = BasicBlock::Create(Ctx, "offload.false", &F);
    auto *Merge       = BasicBlock::Create(Ctx, "mergeblock", &F, nullptr);
    ConstantInt *Zero = ConstantInt::get(Int64Ty, 0);
    auto *Offload     = cast<Function>(
        Mod->getOrInsertFunction("__offload_func_" + Id, OffloadTy));

    // Split the start basic block so that we can insert a call to the offloaded
    // function while maintaining the rest of the original CFG.
    auto *SSplit = (StartBB)->splitBasicBlock((StartBB)->getFirstInsertionPt());
    SSplit->setName((StartBB)->getName());
    BranchInst::Create(Merge, Success);

    if (StartBB == LastBB) LastBB = SSplit;

    // Add a struct to the function entry block in order to
    // capture the live outs.
    auto InsertionPt = F.getEntryBlock().getFirstInsertionPt();
    auto *StructTy   = getStructType(LiveOut, Mod);
    auto *LOS        = new AllocaInst(StructTy, "", &*InsertionPt);
    GetElementPtrInst *StPtr =
        GetElementPtrInst::CreateInBounds(LOS, {Zero}, "", &*InsertionPt);

    // Erase the branch of the split start block (this is always UBR).
    // Replace with a call to the offloaded function and then branch to
    // success / fail based on retval.
    (StartBB)->getTerminator()->eraseFromParent();
    vector<Value *> Params;
    for (auto &V : LiveIn)
        Params.push_back(V);

    /// Fix for Legup issue :
    /// They build a block ram for each global present as a reference in the
    /// function being offloaded. So this creates functionally incorrect
    /// code for the ARM hybrid where the FPGA fabric only references its own
    /// global block ram instead of the one shared with the CPU.
    // if (ConvertGlobalsToPointers) {
    for (auto &G : Globals) {
        Params.push_back(G);
    }
    //}

    auto *BufTy = Type::getInt8PtrTy(Ctx);
    // ArrayType *LogArrTy   = ArrayType::get(IntegerType::get(Ctx, 8), 0);
    ArrayType *SizesArrTy = ArrayType::get(IntegerType::get(Ctx, 32), 0);

    // auto *ULog     = Mod->getOrInsertGlobal("__undo_log_" + Id, LogArrTy);
    auto *USizes   = Mod->getOrInsertGlobal("__undo_sizes_" + Id, SizesArrTy);
    auto *NumStore = Mod->getOrInsertGlobal("__undo_num_stores_" + Id,
                                            IntegerType::getInt32Ty(Ctx));

    // 1. Load the address from the undo log global pointer
    // 2. Pass the address as parameter
    vector<Value *> Idx = {Zero, Zero};

    GlobalVariable *ULogPtrGV =
        new GlobalVariable(*Mod, BufTy, false, GlobalValue::InternalLinkage,
                           ConstantInt::getNullValue(BufTy), "__undo_buffer");


    auto *ULog = new LoadInst(ULogPtrGV, "", (StartBB));

    Params.push_back(ULog);

    Params.push_back(StPtr);


    /// Create the call to the offloaded function
    auto *CI = CallInst::Create(Offload, Params, "", (StartBB));
    BranchInst::Create(Success, Fail, CI, (StartBB));

    // Divert control flow to pass through merge block from
    // original CFG.
    Merge->getInstList().push_back((LastBB)->getTerminator()->clone());
    (LastBB)->getTerminator()->eraseFromParent();
    BranchInst::Create(Merge, (LastBB));



    // Fail Path -- Begin
    Type *ParamTy[] = {Type::getInt8PtrTy(Ctx), Type::getInt32Ty(Ctx),
                       Type::getInt32PtrTy(Ctx)};
    auto *UndoTy = FunctionType::get(Type::getVoidTy(Ctx), ParamTy, false);
    auto *Undo   = Mod->getOrInsertFunction("__undo_mem", UndoTy);

    auto *USizesGEP = GetElementPtrInst::CreateInBounds(USizes, Idx, "", Fail);
    auto *UNS = GetElementPtrInst::CreateInBounds(NumStore, {Zero}, "", Fail);
    auto *NSLoad = new LoadInst(UNS, "", Fail);

    // Fail -- Undo memory
    vector<Value *> Args = {ULog, NSLoad, USizesGEP};

    CallInst::Create(Undo, Args, "", Fail);

    if (EnableValueLogging || EnableMemoryLogging || EnableSimpleLogging) {
        CallInst::Create(Mod->getOrInsertFunction(
                             "__fail", FunctionType::get(VoidTy, {}, false)),
                         {}, "", Fail);
    }

    BranchInst::Create(SSplit, Fail);
    // Fail Path -- End

    // Update the Phi's in the targets of the merge block to use the merge
    // block instead of the LastBB. This needs to run before rewriting
    // uses since the rewriter has to have correct information
    // about the Phi's predecessor blocks in order to update the incorrect
    // values.
    auto updatePhis = [](BasicBlock *Tgt, BasicBlock *Old, BasicBlock *New) {
        for (auto &I : *Tgt) {
            if (auto *Phi = dyn_cast<PHINode>(&I)) {
                // errs() << *Phi << "\n";
                Phi->setIncomingBlock(Phi->getBasicBlockIndex(Old), New);
            }
        }
    };

    for (auto S = succ_begin(Merge), E = succ_end(Merge); S != E; S++) {
        updatePhis(*S, (LastBB), Merge);
    }

    // Success Path - Begin
    // 1. Unpack the live out struct
    // 2. Merge live out values if required
    // 3. Rewrite Phi's in successor of LastBB
    //  a. Use merged values
    //  b. Use incoming block as Merge
    for (uint32_t Idx = 0; Idx < LiveOut.size(); Idx++) {
        auto *Val = LiveOut[Idx];
        // GEP Indices always need to i32 types.
        Value *StIdx     = ConstantInt::get(Int32Ty, Idx, false);
        Value *GEPIdx[2] = {Zero, StIdx};
        auto *EValTy = cast<PointerType>(StPtr->getType())->getElementType();
        auto *ValPtr = GetElementPtrInst::Create(
            EValTy, StPtr, GEPIdx, "st_gep", Success->getTerminator());
        auto *Load = new LoadInst(ValPtr, "live_out", Success->getTerminator());

        // Merge the values -- Use original LiveOut if you came from
        // the LastBB. Use new loaded value if you came from the
        // offloaded function.
        auto *ValTy = Val->getType();
        auto *Phi   = PHINode::Create(ValTy, 2, "", Merge->getTerminator());

        auto prod = dyn_cast<Instruction>(Val);
        assert(prod && "liveout should be instruction");
        auto mappedVal = Val; 

        Phi->addIncoming(mappedVal, (LastBB));
        Phi->addIncoming(Load, Success);

        auto *Orig = dyn_cast<Instruction>(mappedVal);
        assert(Orig && "Expect live outs to be instructions");

        SSAUpdater SSAU;
        SSAU.Initialize(prod->getType(), prod->getName());
        SSAU.AddAvailableValue(Merge, Phi);
        SSAU.AddAvailableValue(prod->getParent(), prod);



        
        for (Value::use_iterator UI = Val->use_begin(), UE = Val->use_end();
             UI != UE;) {
            Use &U = *UI;
            ++UI;
            Instruction *UserInst = cast<Instruction>(U.getUser());
            BasicBlock *UserBB    = UserInst->getParent();
            if ((UserBB != prod->getParent()) || (isa<PHINode>(UserInst))) {
                SSAU.RewriteUseAfterInsertions(U);
            } 
        }//for every use of original value*



 

    
    }
    // Success Path - End


    //propagate phis

    writeCFG(F, F.getName().str()+string(".dot"), Blocks, SSplit);
    common::writeModule(Mod,
                        string("single.") + F.getName().str() + string(".ll"));
    assert(!verifyModule(*Mod, &errs()) && "Module verification failed!");
}

static void runHelperPasses(Function *Offload, string Id) {
    legacy::PassManager PM;
    PM.add(createBasicAAWrapperPass());
    PM.add(llvm::createTypeBasedAAWrapperPass());
    PM.add(createGlobalsAAWrapperPass());
    PM.add(createSCEVAAWrapperPass());
    PM.add(createScopedNoAliasAAWrapperPass());
    PM.add(createCFLAAWrapperPass());
    PM.add(new NeedleHelper(Id));
    PM.run(*Offload->getParent());
}


static void addCtorAndDtor(Module *Mod,
                           SmallVector<Module *, 4> OffloadModules) {

    auto &Ctx    = Mod->getContext();
    auto *VoidTy = Type::getVoidTy(Ctx);

    auto *BufPtrTy        = PointerType::getUnqual(Type::getInt8PtrTy(Ctx));
    auto *PtrToUndoBuffer = Mod->getNamedGlobal("__undo_buffer");

    Type *CtorParamTy[] = {BufPtrTy, Type::getInt32Ty(Ctx),
                           Type::getInt1Ty(Ctx)};

    auto *MweCtor = Mod->getOrInsertFunction(
        "__mwe_ctor", FunctionType::get(VoidTy, CtorParamTy, false));

    auto *CtorWrap = cast<Function>(
        Mod->getOrInsertFunction("__ctor_wrap", VoidTy, nullptr));

    auto *CtorBB = BasicBlock::Create(Ctx, "entry", CtorWrap);
    IRBuilder<> Builder(CtorBB);

    auto maxUndoSize = getMaxUndoSize(OffloadModules);

    auto *Cond =
        EnableValueLogging || EnableMemoryLogging || EnableSimpleLogging
            ? ConstantInt::getTrue(Ctx)
            : ConstantInt::getFalse(Ctx);

    vector<Value *> Args = {
        PtrToUndoBuffer,
        ConstantInt::get(Type::getInt32Ty(Ctx), maxUndoSize * 2 * 8, false),
        Cond};
    Builder.CreateCall(MweCtor, Args);
    Builder.CreateRet(nullptr);

    appendToGlobalCtors(*Mod, llvm::cast<Function>(CtorWrap), 0);

    auto *MweDtor = Mod->getOrInsertFunction(
        "__mwe_dtor", FunctionType::get(VoidTy, {BufPtrTy}, false));
    auto *DtorWrap = cast<Function>(
        Mod->getOrInsertFunction("__dtor_wrap", VoidTy, nullptr));
    auto *DtorBB = BasicBlock::Create(Ctx, "entry", DtorWrap);
    IRBuilder<> Builder2(DtorBB);
    Builder2.CreateCall(MweDtor, {PtrToUndoBuffer});
    Builder2.CreateRet(nullptr);

    appendToGlobalDtors(*Mod, cast<Function>(DtorWrap), 0);
}

void NeedleSequenceOutliner::process(Function &F) {

    //common::runStatsPasses(F);
    vector<BasicBlock*> bbs;
    for (auto &BB:F) bbs.push_back(&BB);
    generalStats(bbs, (F.getName()+".stats.txt").str());


    PostDomTree = &getAnalysis<PostDominatorTree>(F);
    auto *DT    = &getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
    auto *LI    = &getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();

    map<string, BasicBlock *> BlockMap;
    for (auto &BB : F)
        BlockMap[BB.getName().str()] = &BB;

    //we need to allocate a data structure based on which type of code region we are handling
    OrgBlocks& Blocks = *(new OrgBlocks);
    std::string Id;
    if (ExtractAs == braid) {
        assert(0 && "NYI here\n");
        BasicBlock *Start = nullptr, *End = nullptr;
        DenseSet<BasicBlock *> MergeBlocks;
        for (auto &P : Sequences) {

            if (Start == nullptr) {
                Start = BlockMap[P.Seq.front()];
                End   = BlockMap[P.Seq.back()];
                Id    += "_" + P.Id;
            } 

            if (Start == BlockMap[P.Seq.front()] &&
                End == BlockMap[P.Seq.back()]) {
                for (auto BN : P.Seq)
                    MergeBlocks.insert(BlockMap[BN]);
            }
        }

        auto BackEdges = common::getBackEdges(Start);

    } else if (ExtractAs == sequence) { //sequence

        //get back edges
        //paths in sequential order
        //and the blocks within each path in topological order
        BasicBlock *Start = nullptr, *End = nullptr;
        for (auto &P : Sequences) {
            DenseSet<BasicBlock *> MergeBlocks;
            Id    += "_" + P.Id;
            if (Start == nullptr) {
                Start = BlockMap[P.Seq.front()];
                End   = BlockMap[P.Seq.back()];
            }

            if (Start == BlockMap[P.Seq.front()] &&
                End == BlockMap[P.Seq.back()]) {
                for (auto BN : P.Seq)
                    MergeBlocks.insert(BlockMap[BN]);
            }
            Blocks.append(&P, getTraceBlocks(P, BlockMap)); 
        }
        

        auto BackEdges = common::getBackEdges(Start); 

        Blocks.ConnectBlocksAndPaths();
        

    } else { //path
        assert(Sequences.size() == 1 && "Only 1 sequence for path");
        auto &P = Sequences.front();
        Id      = P.Id;
        Blocks.append(&P, getTraceBlocks(P, BlockMap));

    }

    Data["num-extract-blocks"] = Blocks.size();

    /// Print out the source lines :
    // common::printPathSrc(Blocks);
    // errs() << "Blocks:\n";
    // Get the number of phi nodes originally

    uint32_t PhiBefore = 0;
    
    for (auto &BB : Blocks) {
        for (auto &I : *BB) {
            if (isa<PHINode>(&I))
                PhiBefore++;
        }
    }

    ExtractedModules.push_back(
        llvm::make_unique<Module>("mwe", getGlobalContext()));
    Module *Mod = ExtractedModules.back().get();
    Mod->setDataLayout(F.getParent()->getDataLayout());

    OrgBlocks& BlockV = Blocks;

    // Extract the blocks and create a new function
    SetVector<Value *> LiveOut, LiveIn, Globals;
    Function *Offload =
        extract(PostDomTree, Mod, BlockV, LiveIn, LiveOut, Globals, DT, LI, Id);

    runHelperPasses(Offload, Id);

    vector<BasicBlock *> sequ(BlockV.rbegin(), BlockV.rend());
    generalStats(sequ, (F.getName() + Id + ".stats.txt").str());

    instrument(F, BlockV, Offload->getFunctionType(), LiveIn, LiveOut, Globals,
               DT, Id);

    addCtorAndDtor(F.getParent(), {Offload->getParent()});

    // Get the number of phi nodes after
    uint32_t PhiAfter = 0;
    for (auto &BB : *Offload) {
        for (auto &I : BB) {
            if (isa<PHINode>(&I))
                PhiAfter++;
        }
    }

    Data["phi-simplified"] = PhiBefore - PhiAfter;

    valueLogging(Offload);
    memoryLogging(Offload);

    // common::printCFG(F);
    common::writeModule(Mod, (Id) + string(".ll"));

    assert(!verifyModule(*Mod, &errs()) && "Module verification failed!");
}

void static PrintDebugValue(Value*& val) {

    val->dump();
  
}

void static PrintDebugInstr(Instruction*& val) {

    val->dump();
  
}

void static PrintDebugGlobal(GlobalVariable*& val) {

    val->dump();
  
}

void static PrintDebugBbl(BasicBlock*& val) {

    val->dump();
  
}

void static PrintDebugFunc(Function*& val) {

    val->dump();
  
}

void static PrintDebugPhi(PHINode*& val) {

    val->dump();
  
}

bool NeedleSequenceOutliner::runOnModule(Module &M) {

    for (auto &F : M)
        if (isTargetFunction(F, FunctionList))
            process(F);

    return false;
}



char NeedleSequenceOutliner::ID = 0;
