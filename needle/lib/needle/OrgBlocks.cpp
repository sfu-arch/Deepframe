//Author: Apala Guha

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

#include "NeedleSequenceOutliner.h"
#include "Common.h"


using namespace llvm;
using namespace needle;
using namespace std;
using namespace common;

extern cl::opt<ExtractType> ExtractAs;

OrgBlocks::OrgBlocks() {}
void OrgBlocks::operator=(SetVector<BasicBlock*> const BlockList) {
	assert(0 && "NYI\n");
	if (ExtractAs == ExtractType::sequence) {
		assert (0 && "NA for sequences\n");
	} 

}//operator=
int OrgBlocks::size () const {
		return seqBlocks.size();

}//size
OrgBlocks::iterator& OrgBlocks::begin() {
    
	iterator* iter = new iterator(seqBlocks.begin());
	return *iter;
	
}//begin
OrgBlocks::iterator& OrgBlocks::end() {

	iterator* iter = new iterator(seqBlocks.end());
	return *iter;
	
}//end

OrgBlocks::iterator& OrgBlocks::rbegin() {

	iterator* iter = new iterator(seqBlocks.rbegin());
	return *iter;
	
}//rend


OrgBlocks::iterator& OrgBlocks::rend() {

	iterator* iter = new iterator(seqBlocks.rend());
	return *iter;
	
}//rend


void OrgBlocks::append(Path* P, SetVector<BasicBlock*> blocks) {




		SetVector<Value*>* lin = new SetVector<Value*>;
		SetVector<Value*>* lout = new SetVector<Value*>;

		SeqPath thisPath;
		thisPath.path = P;
		thisPath.LiveIn = lin;
		thisPath.LiveOut = lout;

		seqPaths.push_back(thisPath);
		int pos = seqPaths.size();
		seqPaths.back().pos = to_string(pos);

		//append basic blocks
		bool isHeader = true;
		for (auto BB=blocks.rbegin(); BB != blocks.rend(); BB++) {
			seqBlocks.insert(seqBlocks.begin(), {*BB, isHeader});
			isHeader = false;
			
		}


		
}//append


void OrgBlocks::prepend(BasicBlock* block) {




		auto pathIter = seqBlocks.begin()->bbPath;
		auto mapIter = seqBlocks.begin()->bbMap;


		//append basic blocks
		bool isHeader = false;
		seqBlocks.insert(seqBlocks.begin(), {block, isHeader, pathIter, mapIter});
		
		
		
}//prepend


void OrgBlocks::insert(iterator& iter, llvm::BasicBlock* bb) {
	assert(0 && "NYI\n");
	assert(!iter.useReverse && "cannot insert basic block using reverse iterator position");


}//insert


OrgBlocks::iterator& OrgBlocks::ProducerOnTrace(iterator const& iter, Instruction* prod, Instruction* cons) {

	assert(size() != 0 && "list of blocks must not be empty\n");

	assert (iter != end() && "iterator must point to a valid element\n");

	auto &blockIter = *(new iterator(iter));
	
	//walk forwards in the consumer block and see if the producer is in that block
	for (auto &I: **blockIter) {
		if (&I == prod) return blockIter;
		if (&I == cons) break;
	}//for each instruction in this block

	//walk back through the preceding blocks to find the producer
	++blockIter;
	for (; blockIter != end(); ++ blockIter) {
		for (auto &I: (**blockIter)) {
			if (&I == prod) {
				return blockIter;
			} //check if it is the producer
		}//for each instruction in a block
	} //for each block in reverse order

	return end();

}//ProducerOnTrace

OrgBlocks::iterator& OrgBlocks::BlockOnTrace(iterator const& iter, BasicBlock* blk) {

	assert(size() != 0 && "list of blocks must not be empty\n");

	assert (iter != end() && "iterator must point to a valid element\n");

	auto &blockIter = *(new iterator(iter));
	auto start = *iter; // starting block
	


	//walk back through the preceding blocks to find the producer
	++blockIter;
	for (; blockIter != end(); ++ blockIter) {
		if (*blockIter == start) return end(); //starting block is not upwards exposed
		if (*blockIter == blk) return blockIter; //found phi operand on trace
	} //for each block in reverse order

	return end();

}//BlockOnTrace



bool OrgBlocks::ConsumerOnTrace(iterator const& iter, Value* val, Instruction* cons) {
	assert(size() != 0 && "list of blocks must not be empty\n");
	Instruction* prod = dyn_cast<Instruction> (val);

	//walk forward from producer to try to find consumer

	//skip instructions in current block, until producer is reached
	auto &blockIter = *(new iterator(iter));
	auto I = (**blockIter).begin();
	for (; I != (**blockIter).end(); ++I) {
		if (&*I != prod) continue;
		break;
	}//for each instruction in this block

	//the following assert is not needed because producer could be a
	//live-in or a global
	//assert ((I != (**blockIter).end()) && "producer must be on the block\n");

	//try to find consumer in this block
	I++;
	for (; I != (**blockIter).end(); ++I) {
		if (&*I == cons) return true;

	}//for each instruction in this block




	//walk forward through the succeeding blocks to find the consumer
	++blockIter;
	for (; blockIter != rend(); ++blockIter) {
		for (auto &I: (**blockIter)) {
			if (&I == prod) { //producer is not downwards exposed
				return false;
			}
			if (&I == cons) {
				return true;
			} //check if it is the producer
		}//for each instruction in a block
	} //for each block in reverse order

	return false;

}//ConsumerOnTrace

bool OrgBlocks::AnyConsumerOnTrace(iterator const& iter, Value* val) {
	auto Ins = dyn_cast<Instruction> (val);
	assert(Ins && "value has to be an instruction\n");

	for (auto UI = Ins->use_begin(), UE = Ins->use_end(); UI != UE;
         UI++) {
        if (auto UIns = dyn_cast<Instruction>(UI->getUser())) {
            if (ConsumerOnTrace(iter, Ins, UIns)) return true;
        }
    }

    return false;

}//AnyConsumerOnTrace



BasicBlock* OrgBlocks::back() {
	return seqBlocks.back().bb;
}//back

BasicBlock* OrgBlocks::front() {
	return seqBlocks.front().bb;
}//front

int OrgBlocks::count (Value* oldVal, vector<ValMap*>& mapStack) {
	
	for(auto &mapIter:mapStack) {
		int exists = mapIter->count(oldVal);
		if (exists) return exists;

	}

	return 0;
}

Value* OrgBlocks::find(Value* oldVal, vector<ValMap*>& mapStack) {

	for(auto &mapIter:mapStack) {
		Value* newVal = mapIter->find(oldVal);
		if (newVal != NULL) return newVal;

	}

	return NULL;
}

void OrgBlocks::SetValueMaps(vector<ValMap*>& mapStack) {
	//iterate over the path vector and set the value maps
	for (auto &blk: seqBlocks) {
		ValMap* newmap = new ValMap;
		mapStack.insert(mapStack.begin(), newmap);
	}


	auto mapIter = mapStack.begin();
	for (auto &blk: seqBlocks) {
		assert (mapIter != mapStack.end() && "map stack must have one more map than the number of paths in the sequence, for globals\n");
		blk.bbMap = mapIter;
		mapIter ++;
	}

	assert (mapIter == mapStack.end() - 1 && "map stack must have one more map than the number of paths in the sequence, for globals\n");
}//SetValueMaps

void OrgBlocks::ConnectBlocksAndPaths() {
	auto pathIter = seqPaths.end() - 1;
	auto blockIter = seqBlocks.end() - 1;

	assert (blockIter->isHeader ==  true && "first block must be a header\n");

	blockIter->bbPath = pathIter;
	if (blockIter == seqBlocks.begin()) return; //single-block path
	

	--blockIter;
	for (; ; --blockIter) {

		//check continuity
		auto nextBlock = blockIter;
		++nextBlock;
		if (nextBlock != seqBlocks.end()) {
	        vector<BasicBlock *> Succs(succ_begin(nextBlock->bb), succ_end(nextBlock->bb));
	        assert(std::find(Succs.begin(), Succs.end(), blockIter->bb) != Succs.end() && "Path is not continuous!\n");			
		}


		if (!blockIter->isHeader) {
			blockIter->bbPath = pathIter;
		} else {
			pathIter --;
			blockIter->bbPath = pathIter; 
		}

		if (blockIter == seqBlocks.begin()) break;

	}

	assert ((pathIter == seqPaths.begin()) && "path vector iterator must be pointing to globals frame\n");
}//ConnectBlocksAndPaths






bool OrgBlocks::UseIsReachable(Instruction* prod, Instruction* cons, OrgBlocks::iterator& iter) {
	//some uses are reachable only through side exits
	//these should not be live-outs
	//check whether the producer is live at the end of the last block	


	//initialize
	auto consBbl = cons->getParent();
	auto prodBbl = prod->getParent();
	auto Phi = dyn_cast<PHINode>(cons);
	if ((Phi == NULL) && (consBbl == prodBbl)) return false; //there is no path to this use from end of last block because the use is not upward exposed

	//breadth first search through predecessors of consumer
	queue <BasicBlock*> q;
	DenseSet <BasicBlock*> visited;

	//if consumer is a phi node, only some of its predecessors will be pushed
	//otherwise all predecessors will be pushed
	
	if (Phi != NULL) { //phi node
		auto NV = Phi->getNumIncomingValues();
		for (unsigned i = 0; i < NV; i++) {
            auto *Val = Phi->getIncomingValue(i);

            if (Val == prod) {
            	q.push(Phi->getIncomingBlock(i));
            	visited.insert(q.back());
            	break;
            }

        }//queue producer block
	} else { //not a phi node
		for (pred_iterator PI = pred_begin(consBbl), E = pred_end(consBbl); PI != E; ++PI) {
  			q.push(*PI);
  			visited.insert(q.back());
  		}
	}//not a phi node

	//bfs through blocks
	bool found = false;
	while (!q.empty() && !found) {
		auto bbl = q.front();
		q.pop();


		if (bbl == *begin()) { //found
			found = true;
			break;
		}
		if (bbl == prodBbl) continue; //cut off at definition


		for (pred_iterator PI = pred_begin(bbl), E = pred_end(bbl); PI != E; ++PI) {
			if (visited.count(*PI) != 0) continue;
  			q.push(*PI);
  			visited.insert(q.back());
  		}
	} //process queue

	if (!found) return false; //value not alive at end of last block

	//now check if the producer is downwards exposed
	return DownwardsExposed(prodBbl, iter);






}//UseIsReachable


bool OrgBlocks::DownwardsExposed(BasicBlock* prodBbl, iterator iter) {
	assert(iter.UseReverse() && "should be searching in topological order\n");
	assert(*iter == prodBbl && "starting iterator must point to defining block\n");

	++iter;
	for (; iter != rend(); ++iter) {
		if (*iter == prodBbl) return false;
	}

	return true;

}//DownwardsExposed









