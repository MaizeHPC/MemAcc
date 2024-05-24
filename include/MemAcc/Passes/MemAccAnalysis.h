#ifndef DIALECT_MEMACC_ANALYSIS_H
#define DIALECT_MEMACC_ANALYSIS_H

#include "mlir/Pass/Pass.h"
#include "MemAcc/Ops.h"
#include "MemAcc/Passes/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <iostream>

namespace mlir {
    class DFS{
    public:

    ///GatherUseInfo
    // Record how the result of GatherPath is used by other ops
    // users: the ops that use the result of the deepest load ops
    // operandIdx: the operand index of the users that use the result of the deepest load ops
    // This structure is needed for pointing the users to load from the scratchpad
    struct GatherUseInfo{
        llvm::SmallVector<Operation *, 16> users;
        llvm::SmallVector<int, 16> operandIdx;
    };
    
    // Gather path stores the indirect chain from induction variable to several deepest load ops
    // It also records how the result of the deepest load ops are used by other ops for rewriting
    struct GatherPath{
        llvm::SmallVector<Operation *, 16> indirectChain;
        llvm::SmallPtrSet<Operation *, 16> indirectUseSet;
        llvm::DenseMap<Operation *, GatherUseInfo> deepestLoadToExternUsers;
        unsigned int indirectDepth = 0;
        void verification();
        void merge(const GatherPath& other);
        void print();
    };

    // Scatter path stores the indirect chain from induction variable to several deepest store ops
    // It also records the value of the store ops for rewriting
    struct ScatterPath{
        llvm::SmallVector<Operation *, 16> indirectChain;
        llvm::SmallPtrSet<Operation *, 16> indirectUseSet;
        llvm::DenseMap<Operation *, Value> storeOpVals;
        unsigned int indirectDepth = 0;
        void verification();
        void merge(const ScatterPath& other);
        void print();
    };

    // Start from a ScatterPath: 
    // Target: a store Op with value that is a binary arith.* op
    //         one of the operands is a load op
    //         the load op has the same value as the address offset of the store op
    struct RMWPath{
        // TODO: Implement this!
    };

    private:
    llvm::DenseMap<Operation *, GatherPath> gatherPaths_;
    llvm::DenseMap<Operation *, ScatterPath> scatterPaths_;

    // Give a load/store, return a load op/address calculation it address depends on
    // If it depends on an induction var, return the forOp that generates the induction var
    llvm::DenseMap<Operation *, Operation*> addressDependencyMap_;
    llvm::SmallVector<Operation *, 16> currIndChain_;
    llvm::SmallPtrSet<Operation *, 16> currIndMap_;
    GatherPath resultGatherPath_;
    ScatterPath resultScatterPath_;
    bool analysisDone = false;
    void solve(Value curr_val, Operation *op, unsigned int depth = 0, Operation* addressDependencyOp = nullptr);
    void filterGatherPath();
    public:
    void print_results();
    template <typename ForOpType>
    void analyzeLoadOps(ForOpType forOp){
        if (analysisDone) return;

        // Step1: DFS to find all gather traces and scatter traces
        // For all instructions in forOp's body, solve
        for (auto& op: forOp.getRegion().front()){
            currIndChain_.push_back(&op);
            currIndMap_.insert(&op);
            solve(forOp.getInductionVar(), &op, 0, forOp);
            currIndChain_.pop_back();
            currIndMap_.erase(&op);
        }

        // Step2: merge all scatter path
        auto scatterPathsIter = scatterPaths_.begin();
        while (scatterPathsIter != scatterPaths_.end()){
            resultScatterPath_.merge(scatterPathsIter->second);
            scatterPathsIter++;
        }

        // Step3: filter out gather paths that are only used for address of scatter paths
        filterGatherPath();

        // Step4: merge all gather paths from the beginning
        auto gatherPathsIter = gatherPaths_.begin();
        while (gatherPathsIter != gatherPaths_.end()){
            resultGatherPath_.merge(gatherPathsIter->second);
            gatherPathsIter++;
        }

        analysisDone = true;
    }

    GatherPath getGatherPath(){
        assert(analysisDone && "Analysis not done yet\n");
        return resultGatherPath_;
    }

    ScatterPath getScatterPath(){
        assert(analysisDone && "Analysis not done yet\n");
        return resultScatterPath_;
    }

    llvm::DenseMap<Operation *, Operation*> getAddressDependencyMap(){
        assert(analysisDone && "Analysis not done yet\n");
        return addressDependencyMap_;
    }
};

}

#endif
