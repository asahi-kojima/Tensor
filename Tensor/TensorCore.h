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

	//可変長コンストラクタ
	template <typename ...Args>
	TensorCore(Args ... args)
		: mTensorDataSize(0)
	{
		mTensorDimension = sizeof...(args);
		if (mTensorDimension == 0)
		{
			assert(0);
		}

		//初期状態の各次元のサイズを決定する。
		mEachAxisSize.resize(mTensorDimension);
		constructorArgsDevider(0, args...);

		//テンソルのデータサイズを定める。
		u64 size = 1;
		for (const auto& eachSize : mEachAxisSize)
		{
			size *= eachSize;
		}

		//データを格納するメモリのサイズを変更
		mComponentNum = size;
		mTensorData.resize(mComponentNum);
		mDeltaTensorData.resize(mComponentNum);

		//無理やり初期化するために強制的にconstを外している。
		u32* p2mTensorDataSize = const_cast<u32*>(&mTensorDataSize);
		*p2mTensorDataSize = size;
	}

	TensorCore(const TensorCore& tensorCore)
		: mTensorDataSize(0)
	{
		mTensorDimension = tensorCore.mTensorDimension;

		//初期状態の各次元のサイズを決定する。
		mEachAxisSize.resize(mTensorDimension);
		for (u32 i = 0; i < mTensorDimension; i++)
		{
			mEachAxisSize[i] = tensorCore.mEachAxisSize[i];
		}

		//テンソルのデータサイズを定める。
		u64 size = 1;
		for (const auto& eachSize : mEachAxisSize)
		{
			size *= eachSize;
		}

		//データを格納するメモリのサイズを変更
		mComponentNum = size;
		mTensorData.resize(mComponentNum);
		mDeltaTensorData.resize(mComponentNum);

		//無理やり初期化するために強制的にconstを外している。
		u32* p2mTensorDataSize = const_cast<u32*>(&mTensorDataSize);
		*p2mTensorDataSize = size;
	}


private:
	u32 mComponentNum;

	//現在のテンソルの次元
	u32 mTensorDimension;
	//現在のテンソルの各次元毎のサイズ
	std::vector<u32> mEachAxisSize;

	//テンソルのデータ量（これは不変）
	const u32 mTensorDataSize;
	//テンソルのデータ（このサイズも不変 =====> resizeは一回しか呼ばないこと!!!）
	std::vector<f32> mTensorData;
	std::vector<f32> mDeltaTensorData;

	void zeroGrad()
	{
		for (auto& comp : mDeltaTensorData)
		{
			comp = 0;
		}
	}

	//コンストラクタの可変長引数を処理するための関数
	//各軸方向のサイズもここで格納する。
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