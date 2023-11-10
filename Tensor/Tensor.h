#pragma once


#include "TensorManager.h"


class Tensor
{
public:
	/// <summary>
	/// �T�C�Y���Ȃ��e���\��������Ă��Ӗ����Ȃ��ׁA�폜
	/// </summary>
	Tensor() = delete;

	/// <summary>
	/// 
	/// </summary>
	template <typename ... Args>
	Tensor(Args ... args)
	{
		TensorManager::getInstance().createNewTensor(mInstanceNo, args...);
	}

	//template <typename ... Args>
	//Tensor(const std::vector<f32>& components, Args ... args)
	//{
	//	TensorManager::getInstance().createNewTensor(mInstanceNo, components,  args...);
	//}

	/// <summary>
	/// �R�s�[�R���X�g���N�^
	/// </summary>
	/// <param name=""></param>
	Tensor(const Tensor&);
	Tensor& operator=(const Tensor&);


	Tensor(Tensor&&);
	Tensor& operator=(Tensor&&);

	~Tensor() = default;

	f32& operator[](const u32);
	f32 operator[](const u32) const;
	u32 getComponentNum() const;

	//operator�֌W
	Tensor operator+(const Tensor&) const &;
	Tensor operator+(const Tensor&) &&;
	Tensor operator+(Tensor&&) const &;
	Tensor operator+(Tensor&&) &&;

	Tensor operator*(const Tensor&) const &;
	//Tensor operator*(const Tensor&) &&;

	Tensor MatMul(const Tensor&, const Tensor&);

	void forward();
	void backward();

	//Debug
	f32& getComp(u32 i)
	{
		return TensorManager::getInstance().getComp(mInstanceNo, i);
	}
	f32& getDeltaComp(u32 i)
	{
		return TensorManager::getInstance().getDeltaComp(mInstanceNo, i);
	}


private:
	u32 mInstanceNo;

};