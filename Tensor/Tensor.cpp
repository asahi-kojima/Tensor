#include "Tensor.h"




Tensor::Tensor(const Tensor& tensor)
{
	TensorManager::getInstance().createTensorLike(mInstanceNo, tensor.mInstanceNo);
}

Tensor::Tensor(Tensor&& tensor)
{
	//�C���X�^���X�ԍ������L����
	mInstanceNo = tensor.mInstanceNo;
}

u32 Tensor::getComponentNum() const
{
	return TensorManager::getInstance().getComponentNum(mInstanceNo);
}


void Tensor::forward()
{
	//�����Ōv�Z�O���t�ɃA�N�Z�X����
	//���[�v�Ō㑱�S����forward���񂷁B
	//for (;;)
	//{
	//}
	const std::vector<u32>& sortedGraph = TensorManager::getInstance().getSortedGraph(mInstanceNo);
	auto targetIter = std::find(sortedGraph.begin(), sortedGraph.end(), mInstanceNo);
	if (targetIter == sortedGraph.end())
	{
		assert(0);
	}

	for (auto iter = targetIter, end = sortedGraph.end(); iter != end; iter++)
	{
		u32 instanceNo = *iter;
		std::cout << instanceNo;
		TensorManager::getInstance().forward(instanceNo);
	}

	TensorManager::getInstance().forward(mInstanceNo);
}

void Tensor::backward()
{
	//�����Ōv�Z�O���t�ɃA�N�Z�X����
	//���[�v�Ō㑱�S����forward���񂷁B
	/*for (;;)
	{
	}*/
	TensorManager::getInstance().backward(mInstanceNo);
}