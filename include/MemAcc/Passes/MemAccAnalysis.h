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
    struct DFS{
    
    // record how the result of GatherPath is used by other ops
    struct GatherPathOut{
        llvm::SmallVector<Operation *, 16> users;
        llvm::SmallPtrSet<Operation *, 16> userSet;
        llvm::SmallVector<int, 16> operandIdx;
    };
    
    struct GatherPath{
        llvm::SmallVector<Operation *, 16> indirectChain;
        llvm::SmallPtrSet<Operation *, 16> indirectUseSet;
        llvm::DenseMap<Operation *, GatherPathOut> externUsers;
        unsigned int indirectDepth = 0;

        void verification();
        void merge(const GatherPath& other);
        void print();
    };
    struct ScatterPath{
        llvm::SmallVector<Operation *, 16> indirectChain;
        llvm::SmallPtrSet<Operation *, 16> indirectUseSet;
        llvm::DenseMap<Operation *, Value> storeOpVals;
        unsigned int indirectDepth = 0;
        void verification();
        void merge(const ScatterPath& other);
        void print();
    };
    struct RMWTrace{
        // TODO: Implement this!
    };
    llvm::DenseMap<Operation *, GatherPath> gatherPaths_;
    llvm::DenseMap<Operation *, ScatterPath> scatterPaths_;
    llvm::SmallVector<Operation *, 16> currIndChain_;
    llvm::SmallPtrSet<Operation *, 16> currIndMap_;

    void print_results();

    private:
    void solve(Value curr_val, Operation *op, unsigned int depth = 0);

    public:
    template <typename ForOpType>
    void analyzeLoadOps(ForOpType forOp, 
                             GatherPath& gatherPath,
                             ScatterPath& scatterPaths){
        // Step1: DFS to find all gather traces and scatter traces
        // For all instructions in forOp's body, solve
        for (auto& op: *forOp.getBody()){
            currIndChain_.push_back(&op);
            currIndMap_.insert(&op);
            solve(forOp.getInductionVar(), &op);
            currIndChain_.pop_back();
            currIndMap_.erase(&op);
        }


        // print_results();
        // Step2: merge all gather paths from the beginning
        auto gatherPathsIter = gatherPaths_.begin();
        while (gatherPathsIter != gatherPaths_.end()){
            gatherPath.merge(gatherPathsIter->second);
            gatherPathsIter++;
        }
        // gatherPath.print();

        // Step3: merge all scatter path
        auto scatterPathsIter = scatterPaths_.begin();
        while (scatterPathsIter != scatterPaths_.end()){
            scatterPaths.merge(scatterPathsIter->second);
            scatterPathsIter++;
        }
        // scatterPaths.print();
    }
};

}

#endif
