#include "Tensor.h"




Tensor::Tensor(const Tensor& tensor)
{
	TensorManager::getInstance().createTensorLike(mInstanceNo, tensor.mInstanceNo);
}

Tensor::Tensor(Tensor&& tensor)
{
	//インスタンス番号を共有する
	mInstanceNo = tensor.mInstanceNo;
}

u32 Tensor::getComponentNum() const
{
	return TensorManager::getInstance().getComponentNum(mInstanceNo);
}


void Tensor::forward()
{
	const std::vector<u32>& sortedGraph = TensorManager::getInstance().getSortedGraph(mInstanceNo);
	auto targetIter = std::find(sortedGraph.begin(), sortedGraph.end(), mInstanceNo);
	if (targetIter == sortedGraph.end())
	{
		assert(0);
	}

	for (auto iter = targetIter, end = sortedGraph.end(); iter != end; iter++)
	{
		u32 instanceNo = *iter;
		TensorManager::getInstance().forward(instanceNo);
	}
#if _DEBUG
	for (auto iter = targetIter, end = sortedGraph.end(); iter != end; iter++)
	{
		u32 instanceNo = *iter;
		std::cout << instanceNo << "->";
	}
	std::cout << "/" << std::endl;
#endif

	TensorManager::getInstance().forward(mInstanceNo);
}

void Tensor::backward()
{
	//ここで計算グラフにアクセスして
	//ループで後続全部のforwardを回す。
	/*for (;;)
	{
	}*/
	TensorManager::getInstance().backward(mInstanceNo);
}