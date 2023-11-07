#pragma once
#include <cassert>
#include <iostream>
#include<vector>

#include "typeinfo.h"


class TensorCore
{
	friend class TensorManager;

public:
	u32 getComponentNum() const
	{
		return mComponentNum;
	}

	f32 getComponent(u32 i) const
	{
		assert(i < mComponentNum);
		return mTensorData[i];
	}

	f32& getComponent(u32 i)
	{
		assert(i < mComponentNum);
		return mTensorData[i];
	}

	f32 getDeltaComponent(u32 i) const
	{
		assert(i < mComponentNum);
		return mDeltaTensorData[i];
	}

	f32& getDeltaComponent(u32 i)
	{
		assert(i < mComponentNum);
		return mDeltaTensorData[i];
	}

	//�ϒ��R���X�g���N�^
	template <typename ...Args>
	TensorCore(Args ... args)
		: mTensorDataSize(0)
	{
		mTensorDimension = sizeof...(args);
		if (mTensorDimension == 0)
		{
			assert(0);
		}

		//������Ԃ̊e�����̃T�C�Y�����肷��B
		mEachAxisSize.resize(mTensorDimension);
		constructorArgsDevider(0, args...);

		//�e���\���̃f�[�^�T�C�Y���߂�B
		u64 size = 1;
		for (const auto& eachSize : mEachAxisSize)
		{
			size *= eachSize;
		}

		//�f�[�^���i�[���郁�����̃T�C�Y��ύX
		mComponentNum = size;
		mTensorData.resize(mComponentNum);
		mDeltaTensorData.resize(mComponentNum);

		//������菉�������邽�߂ɋ����I��const���O���Ă���B
		u32* p2mTensorDataSize = const_cast<u32*>(&mTensorDataSize);
		*p2mTensorDataSize = size;
	}

	TensorCore(const TensorCore& tensorCore)
		: mTensorDataSize(0)
	{
		mTensorDimension = tensorCore.mTensorDimension;

		//������Ԃ̊e�����̃T�C�Y�����肷��B
		mEachAxisSize.resize(mTensorDimension);
		for (u32 i = 0; i < mTensorDimension; i++)
		{
			mEachAxisSize[i] = tensorCore.mEachAxisSize[i];
		}

		//�e���\���̃f�[�^�T�C�Y���߂�B
		u64 size = 1;
		for (const auto& eachSize : mEachAxisSize)
		{
			size *= eachSize;
		}

		//�f�[�^���i�[���郁�����̃T�C�Y��ύX
		mComponentNum = size;
		mTensorData.resize(mComponentNum);
		mDeltaTensorData.resize(mComponentNum);

		//������菉�������邽�߂ɋ����I��const���O���Ă���B
		u32* p2mTensorDataSize = const_cast<u32*>(&mTensorDataSize);
		*p2mTensorDataSize = size;
	}


private:
	u32 mComponentNum;

	//���݂̃e���\���̎���
	u32 mTensorDimension;
	//���݂̃e���\���̊e�������̃T�C�Y
	std::vector<u32> mEachAxisSize;

	//�e���\���̃f�[�^�ʁi����͕s�ρj
	const u32 mTensorDataSize;
	//�e���\���̃f�[�^�i���̃T�C�Y���s�� =====> resize�͈�񂵂��Ă΂Ȃ�����!!!�j
	std::vector<f32> mTensorData;
	std::vector<f32> mDeltaTensorData;

	void zeroGrad()
	{
		for (auto& comp : mDeltaTensorData)
		{
			comp = 0;
		}
	}

	//�R���X�g���N�^�̉ϒ��������������邽�߂̊֐�
	//�e�������̃T�C�Y�������Ŋi�[����B
	template <typename Head, typename ... Tail>
	void constructorArgsDevider(u32 id, Head&& head, Tail&& ... tail)
	{
		const auto& headID = typeid(Head);
		const auto& s32ID = typeid(s32);
		const auto& u32ID = typeid(u32);

		if (!(headID == u32ID || headID == s32ID))
		{
			std::cout << "TensorCore " << id << "th component type error" << std::endl;
			assert(0);
		}

		if (head <= 0)
		{
			std::cout << "TensorCore " << id << "th component size error" << std::endl;
			assert(0);
		}

		mEachAxisSize[id] = head;
		constructorArgsDevider(id + 1, tail...);
	}
	void constructorArgsDevider(u32 id) {}
};