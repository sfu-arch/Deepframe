#ifndef ORGBLOCKS_H
#define ORGBLOCKS_H


//Author: Apala Guha




namespace needle {


class ValMap {

	//this class maps values

private:
	std::map<llvm::Value*, llvm::Value*> valueMap;

public:
	int count (llvm::Value* oldVal) { return valueMap.count(oldVal); }
	void insert(llvm::Value* oldVal, llvm::Value* newVal) { valueMap.insert(std::pair<llvm::Value*, llvm::Value*> (oldVal, newVal)); }
	llvm::Value* find(llvm::Value* oldVal) { 
		std::map<llvm::Value*, llvm::Value*>::iterator iter = valueMap.find(oldVal);
		if (iter == valueMap.end()) return NULL;
		return iter->second;
	}

}; //class ValMap



class OrgBlocks {
private:

	typedef struct {

		needle::Path* path;
		

		//we need to remember which copies of a value
		//is a live-in and a live-out
		llvm::SetVector<llvm::Value*>* LiveIn; 
		llvm::SetVector<llvm::Value*>* LiveOut;

		std::string pos;


	} SeqPath;

	typedef std::vector <SeqPath> PathVec;
	PathVec seqPaths;

	typedef struct {
		llvm::BasicBlock* bb;
		bool isHeader; //starts a path
		PathVec::iterator bbPath;

		//each basic block points to a specific value map in the stack
		//it is maintained per bbl so that phi nodes get correct maps
		//of operands that have been defined after them
		std::vector<ValMap*>::iterator bbMap;
		


	} SeqBlk;
	

	//for sequences
	typedef std::vector <SeqBlk> Sequence;
	Sequence seqBlocks;




public:

	class iterator:public std::iterator<std::bidirectional_iterator_tag, llvm::BasicBlock*, int> {
	private:
	
		//for sequences
		Sequence::iterator seqIter;
		Sequence::reverse_iterator seqRIter;
		
		
		//flag to select braid vs sequence
		bool useReverse;


	public:
		iterator(Sequence::iterator const &i) { seqIter = i; /*useSet = false;*/  useReverse = false; }
		iterator(Sequence::reverse_iterator const &i) { seqRIter = i; /*useSet = false;*/  useReverse = true; }
		

		iterator& operator=(iterator const &i) {
			*this = i;
		}

		bool operator!=(iterator const &i) const{
			if (useReverse != i.useReverse) return true;
			
            if (!useReverse) {
				return (seqIter != i.seqIter);
			} else {
				return (seqRIter != i.seqRIter);
			}
		}

		bool operator==(iterator const &i) const{
			return (!(*this != i));
		}

		iterator& operator++() {
        if (!useReverse) seqIter ++;
			else seqRIter ++;
			return *this;
		}
		iterator& operator--() {
        if (!useReverse) seqIter --;
			else seqRIter --;
			return *this;
		}


		llvm::BasicBlock* const& operator*() const{
        if (!useReverse) return seqIter->bb;
			else return seqRIter->bb;
			
		}

		needle::Path* GetPath() {
			if (!useReverse) return seqIter->bbPath->path;
			else return seqRIter->bbPath->path;
		}

		llvm::SetVector<llvm::Value*>& GetLiveOutSet() {
			if (!useReverse) return *(seqIter->bbPath->LiveOut);
			else return *(seqRIter->bbPath->LiveOut);
		}



		void insert(llvm::Value* oldVal, llvm::Value* newVal) { 


			if (!useReverse) (*(seqIter->bbMap))->insert(oldVal, newVal);
			else (*(seqRIter->bbMap))->insert(oldVal, newVal);
		}

		int count (llvm::Value* oldVal, std::vector<ValMap*>& mapStack) {
			std::vector<ValMap*>::iterator mapIter;
			if (!useReverse) mapIter = seqIter->bbMap;
			else mapIter = seqRIter->bbMap;

			for(; mapIter != mapStack.end(); mapIter ++) {
				int exists = (*mapIter)->count(oldVal);
				if (exists) return exists;

			}

			return 0;
		}

		llvm::Value* find(llvm::Value* oldVal, std::vector<ValMap*>& mapStack) {
			std::vector<ValMap*>::iterator mapIter;
			if (!useReverse) mapIter = seqIter->bbMap;
			else mapIter = seqRIter->bbMap;

			for(; mapIter != mapStack.end(); mapIter ++) {
				llvm::Value* newVal = (*mapIter)->find(oldVal);
				if (newVal != NULL) return newVal;

			}

			return NULL;
		}

		llvm::Value* FindBranchTarget(llvm::Value* oldVal, std::vector<ValMap*>& mapStack) {



			std::vector<ValMap*>::iterator mapIter;
			if (!useReverse) mapIter = seqIter->bbMap;
			else mapIter = seqRIter->bbMap;
			assert(mapIter != mapStack.end() - 1 && "bbl cannot be pointing to globals map\n");


			if (mapIter != mapStack.begin()) {
				//search in reverse order until end
				for(auto mapIter2 = mapIter - 1; ; mapIter2--) {
					llvm::Value* newVal = (*mapIter2)->find(oldVal);
					if (newVal != NULL) return newVal;

					//check for end
					if (mapIter2 == mapStack.begin()) break;

				}
			}



			return NULL;
		}



		void AddLiveIn(llvm::Value* val) {
			if (IsLiveIn(val)) return;

			if (!useReverse) seqIter->bbPath->LiveIn->insert(val);
			else seqRIter->bbPath->LiveIn->insert(val);
		}

		void AddLiveOut(llvm::Value* val) {
			if (IsLiveOut(val)) return;

			if (!useReverse) seqIter->bbPath->LiveOut->insert(val);
			else seqRIter->bbPath->LiveOut->insert(val);
		}

		bool IsLiveIn (llvm::Value* val) {
			if (!useReverse) return (std::find(seqIter->bbPath->LiveIn->begin(), seqIter->bbPath->LiveIn->end(), val) != seqIter->bbPath->LiveIn->end());
			else return (std::find(seqRIter->bbPath->LiveIn->begin(), seqRIter->bbPath->LiveIn->end(), val) != seqRIter->bbPath->LiveIn->end());
		}

		bool IsLiveOut (llvm::Value* val) {
			if (!useReverse) return (std::find(seqIter->bbPath->LiveOut->begin(), seqIter->bbPath->LiveOut->end(), val) != seqIter->bbPath->LiveOut->end());
			else return (std::find(seqRIter->bbPath->LiveOut->begin(), seqRIter->bbPath->LiveOut->end(), val) != seqRIter->bbPath->LiveOut->end());
		}

		llvm::StringRef GetPathPos () {
			if (!useReverse) return llvm::StringRef(seqIter->bbPath->pos);
			else return llvm::StringRef(seqRIter->bbPath->pos);
		}

		bool IsHeader() {
			if (!useReverse) return seqIter->isHeader;
			else return seqRIter->isHeader;
		}

		bool UseReverse() {
			return useReverse;
		}



		friend class OrgBlocks;
		
	};	
	
	

	OrgBlocks();
	void operator=(llvm::SetVector<llvm::BasicBlock*> const BlockList);
	int size () const;
	iterator& begin() ;
	iterator& end() ;
	iterator& rend() ;
	iterator& rbegin() ;
	void append(needle::Path*, llvm::SetVector<llvm::BasicBlock*> blocks);
	void prepend(llvm::BasicBlock*);
	void insert(iterator& iter, llvm::BasicBlock* bb);
	iterator& ProducerOnTrace(iterator const& iter, llvm::Instruction* prod, llvm::Instruction* cons);
	iterator& BlockOnTrace(iterator const& iter, llvm::BasicBlock*); //for phi node operands
	bool ConsumerOnTrace(iterator const& iter, llvm::Value* val, llvm::Instruction* cons);
	bool AnyConsumerOnTrace(iterator const& iter, llvm::Value* val);
	llvm::BasicBlock* back();
	llvm::BasicBlock* front();
	int count (llvm::Value* oldVal, std::vector<ValMap*>& mapStack);
	llvm::Value* find(llvm::Value* oldVal, std::vector<ValMap*>& mapStack);
	void SetValueMaps(std::vector<ValMap*>& mapStack);
	void ConnectBlocksAndPaths();
	llvm::Value* FindPhiMap(llvm::Value*, llvm::BasicBlock*); //find the latest phi value a live out has been mapped to
	bool UseIsReachable(llvm::Instruction*, llvm::Instruction*, iterator& iter); //can the producer reach the consumer on successful execution?
	bool DownwardsExposed(llvm::BasicBlock* prodBbl, iterator iter);

  
}; //class OrgBlocks






} //end needle namespace

#endif //ORGBLOCKS_H