#pragma once
#include <memory>
#include <vector>
#include <typeinfo>
#include <functional>
#include <map>

#include "typeinfo.h"
#include "TensorCore.h"


//using TensorGraph = std::map<u32, std::vector<u32>>;

struct TensorGraph
{
	using Graph = std::map<u32, std::vector<u32>>;

	Graph mGraph;
	std::vector<u32> mSortedList;
	std::vector<u32> mReverseSortedList;

	void sortGraph()
	{
		std::map<u32, bool> seen;
		for (auto tensorPtr : mGraph)
		{
			seen[tensorPtr.first] = false;
		}

		auto rec = [](auto self, const Graph& graph, u32 originIndex, std::map<u32, bool>& isSeen, std::vector<u32>& sortedList)->void
		{
			isSeen[originIndex] = true;

			bool isExist = graph.count(originIndex);

			if (isExist)
				for (const u32& targetIndex : graph.at(originIndex))
				{
					if (isSeen[targetIndex])
						continue;

					self(self, graph, targetIndex, isSeen, sortedList);
				}

			sortedList.push_back(originIndex);
		};

		mSortedList.clear();
		mReverseSortedList.clear();

		for (auto tensorPtr : mGraph)
		{
			if (seen[tensorPtr.first])
				continue;
			rec(rec, mGraph, tensorPtr.first, seen, mReverseSortedList);
		}

		for (auto riter = mReverseSortedList.rbegin(), rend = mReverseSortedList.rend(); riter != rend; riter++)
		{
			mSortedList.push_back(*riter);
		}
	}
};

class TensorManager
{
public:
	static TensorManager& getInstance()
	{
		static TensorManager manager;
		return manager;
	}

	template <typename ... Args>
	void createNewTensor(u32& instanceNo, Args ... args)
	{
		instanceNo = CreatedTensorNum;
		CreatedTensorNum++;

		TensorInformation info;
		info.mInstanceNo = instanceNo;
		info.mTensorCore = std::make_shared<TensorCore>(args...);

		mTensorInfoTbl.push_back(info);
	}

	void createTensorLike(u32& instanceNo, const u32 targetTensorNo);

	u32 getComponentNum(const u32) const;

	void registForwardInfo(u32 instanceNo, u32 rootInstanceNoTbl[], u32 rootNum,
	std::function<void (std::shared_ptr<TensorCore>, std::vector<std::shared_ptr<TensorCore>>)> forwardRule)
	{
		TensorInformation& targetInfo = mTensorInfoTbl[instanceNo];

		targetInfo.mParentTensorCoreTbl.resize(rootNum);
		for (u32 i = 0; i < rootNum; i++)
		{
			targetInfo.mParentTensorCoreTbl[i] = mTensorInfoTbl[rootInstanceNoTbl[i]].mTensorCore;
		}

		targetInfo.mForwardRule = forwardRule;
	}

	void registBackwardInfo(u32 instanceNo, u32 childInstanceNoTbl[], u32 childNum,
		std::function<void(std::shared_ptr<TensorCore>, std::vector<std::shared_ptr<TensorCore>>)> backwardRule)
	{
		TensorInformation& targetInfo = mTensorInfoTbl[instanceNo];

		std::vector<std::shared_ptr<TensorCore> > tmpChildTensorCoreInfo;

		tmpChildTensorCoreInfo.resize(childNum);
		for (u32 i = 0; i < childNum; i++)
		{
			tmpChildTensorCoreInfo[i] = mTensorInfoTbl[childInstanceNoTbl[i]].mTensorCore;
		}

		targetInfo.mBackwardRuleTbl.push_back(backwardRule);

		targetInfo.mChildTensorCoreTbl.push_back(tmpChildTensorCoreInfo);
	}



	void forward(u32);
	void backward(u32);
	void constructCalculationGraph2(u32, u32, u32);

	const std::vector<u32>& getSortedGraph(u32 instanceNo)
	{
		mTensorInfoTbl[instanceNo].mTensorGraph->sortGraph();
		return mTensorInfoTbl[instanceNo].mTensorGraph->mSortedList;
	}

	f32& getComp(u32 no, u32 i)
	{
		return mTensorInfoTbl[no].mTensorCore->getComponent(i);
	}
	f32& getDeltaComp(u32 no, u32 i)
	{
		return mTensorInfoTbl[no].mTensorCore->getDeltaComponent(i);
	}

private:
	TensorManager()=default;
	~TensorManager() = default;

	inline static u32 CreatedTensorNum = 0;

	//テンソルの情報を全て格納したクラス
	struct TensorInformation
	{
		TensorInformation()
		{
			mForwardRule = [](std::shared_ptr<TensorCore>, std::vector<std::shared_ptr<TensorCore>>) {};
		}

		//インスタンス番号
		u32 mInstanceNo;

		//テンソルグラフ
		std::shared_ptr<TensorGraph> mTensorGraph;

		//テンソル本体
		std::shared_ptr<TensorCore> mTensorCore;

		//自分の生成元
		std::vector<std::shared_ptr<TensorCore> > mParentTensorCoreTbl;
		std::function<void(std::shared_ptr<TensorCore>, std::vector<std::shared_ptr<TensorCore>>)> mForwardRule;

		//自分が生成したテンソル
		std::vector<std::vector<std::shared_ptr<TensorCore> > > mChildTensorCoreTbl;
		std::vector<std::function<void(std::shared_ptr<TensorCore>, std::vector<std::shared_ptr<TensorCore>>)> > mBackwardRuleTbl;
	};



	std::vector<TensorInformation> mTensorInfoTbl;

	std::vector<std::shared_ptr<TensorGraph> > mTensorGraphTbl;
	
	void mergeGraph(std::shared_ptr<TensorGraph>, std::shared_ptr<TensorGraph>);
};


