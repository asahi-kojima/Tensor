#include "TensorManager.h"


void TensorManager::createTensorLike(u32& instanceNo, const u32 targetTensorNo)
{
	instanceNo = CreatedTensorNum;
	CreatedTensorNum++;

	TensorInformation info;
	info.mInstanceNo = instanceNo;
	info.mTensorCore = std::make_shared<TensorCore>(*(mTensorInfoTbl[targetTensorNo].mTensorCore));

	mTensorInfoTbl.push_back(info);
}

u32 TensorManager::getComponentNum(const u32 instanceNo) const
{
	return mTensorInfoTbl[instanceNo].mTensorCore->getComponentNum();
}



void TensorManager::forward(u32 instanceNo)
{
	TensorInformation& targetInfo = mTensorInfoTbl[instanceNo];
	targetInfo.mForwardRule(targetInfo.mTensorCore, targetInfo.mParentTensorCoreTbl);
}

void TensorManager::backward(u32 instanceNo)
{
	TensorInformation& targetInfo = mTensorInfoTbl[instanceNo];

	targetInfo.mTensorCore->zeroGrad();
	for (u32 i = 0; i < targetInfo.mChildTensorCoreTbl.size(); i++)
	{
		targetInfo.mBackwardRuleTbl[i](targetInfo.mTensorCore, targetInfo.mChildTensorCoreTbl[i]);
	}
}


void TensorManager::constructCalculationGraph2(u32 targetTensorNo, u32 tensor0No, u32 tensor1No)
{
	TensorInformation& targetInfo = mTensorInfoTbl[targetTensorNo];
	TensorInformation& tensor0Info = mTensorInfoTbl[tensor0No];
	TensorInformation& tensor1Info = mTensorInfoTbl[tensor1No];

	std::shared_ptr<TensorGraph>& targetGraph = targetInfo.mTensorGraph;
	std::shared_ptr<TensorGraph>& tensor0Graph = tensor0Info.mTensorGraph;
	std::shared_ptr<TensorGraph>& tensor1Graph = tensor1Info.mTensorGraph;


	bool hasGraph0 = (tensor0Graph ? true : false);
	bool hasGraph1 = (tensor1Graph ? true : false);


	if (hasGraph0 && hasGraph1)
	{
		if (tensor0Graph == tensor1Graph)
		{
			//何もしなくていい
		}
		else
		{
			//マージの必要がある。
			mergeGraph(tensor0Graph, tensor1Graph);
		}
	}
	else if (hasGraph0 && !hasGraph1)
	{
		tensor1Graph = tensor0Graph;
	}
	else if (!hasGraph0 && hasGraph1)
	{
		tensor0Graph = tensor1Graph;
	}
	else
	{
		std::shared_ptr<TensorGraph> newGraph = std::make_shared<TensorGraph>();
		TensorManager::getInstance().mTensorGraphTbl.push_back(newGraph);
		tensor0Graph = newGraph;
		tensor1Graph = newGraph;
	}

	targetGraph = tensor0Graph;

	TensorGraph& graph = *tensor0Graph;

	graph.mGraph[tensor0No].push_back(targetTensorNo);
	graph.mGraph[tensor1No].push_back(targetTensorNo);
}

void TensorManager::mergeGraph(std::shared_ptr<TensorGraph> targetGraphPtr, std::shared_ptr<TensorGraph> sourceGraph)
{
	sourceGraph->sortGraph();
	for (auto iter = (*sourceGraph).mSortedList.begin(), end = (*sourceGraph).mSortedList.end(); iter != end; iter++)
	{
		u32 instanceNo = *iter;

		mTensorInfoTbl[instanceNo].mTensorGraph = targetGraphPtr;

		if (sourceGraph->mGraph.find(instanceNo) != sourceGraph->mGraph.end())
		{
			std::vector<u32> targetInstanceNoTbl = sourceGraph->mGraph[instanceNo];
			(*targetGraphPtr).mGraph[instanceNo] = targetInstanceNoTbl;
		}
	}

	for (auto iter = mTensorGraphTbl.begin(), end = mTensorGraphTbl.end(); iter != end; iter++)
	{
		std::shared_ptr<TensorGraph>& graph = *iter;
		if (graph == sourceGraph)
		{
			graph.reset();
			mTensorGraphTbl.erase(iter);
			break;
		}
	}

	sourceGraph.reset();
}