#include "TensorCore.h"
#include "Tensor.h"

namespace
{
	auto forwardRule_Add = [](std::shared_ptr<TensorCore> target, std::vector<std::shared_ptr<TensorCore>> roots) -> void
	{
		for (u32 i = 0, end = target->getComponentNum(); i < end; i++)
		{
			target->getComponent(i) = roots[0]->getComponent(i) + roots[1]->getComponent(i);
		}
	};
	auto backwardRule_Add = [](std::shared_ptr<TensorCore> target, std::vector<std::shared_ptr<TensorCore>> roots) -> void
	{
		for (u32 i = 0, end = target->getComponentNum(); i < end; i++)
		{
			target->getComponent(i) += roots[0]->getDeltaComponent(i);
		}
	};
}

Tensor Tensor::operator + (const Tensor& tensorR) const&
{
	const Tensor& tensorL = *this;

	assert(tensorL.getComponentNum() == tensorR.getComponentNum());

	Tensor targetTensor(tensorL);

	TensorManager& manager = TensorManager::getInstance();

	//生成したターゲットテンソルへの情報登録
	u32 rootTensorTbl[2] = { tensorL.mInstanceNo, tensorR.mInstanceNo };
	u32 rootTensorNum = sizeof(rootTensorTbl) / sizeof(rootTensorTbl[0]);
	manager.registForwardInfo(targetTensor.mInstanceNo, rootTensorTbl, rootTensorNum, forwardRule_Add);


	//左辺用
	{
		u32 childTensorTbl[1] = { targetTensor.mInstanceNo };
		u32 childTensorNum = sizeof(childTensorTbl) / sizeof(childTensorTbl[0]);
		manager.registBackwardInfo(tensorL.mInstanceNo, childTensorTbl, childTensorNum, backwardRule_Add);
	}

	//右辺用
	{
		u32 childTensorTbl[1] = { targetTensor.mInstanceNo };
		u32 childTensorNum = sizeof(childTensorTbl) / sizeof(childTensorTbl[0]);
		manager.registBackwardInfo(tensorR.mInstanceNo, childTensorTbl, childTensorNum, backwardRule_Add);
	}


	//グラフの構築
	TensorManager::getInstance().constructCalculationGraph2(targetTensor.mInstanceNo, tensorL.mInstanceNo, tensorR.mInstanceNo);


	TensorManager::getInstance().forward(targetTensor.mInstanceNo);
	return targetTensor;
}

Tensor Tensor::operator + (const Tensor&)&&
{
	return Tensor(1);
}


Tensor Tensor::operator + (Tensor&&) const&
{
	return Tensor(1);
}

Tensor Tensor::operator+(Tensor&&)&&
{
	return Tensor(1);

}