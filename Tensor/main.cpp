// Tensor.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>



#include "Tensor.h"


void setValue(Tensor& tensor, f32 value)
{
	for (u32 i = 0; i < tensor.getComponentNum(); i++)
	{
		tensor.getComp(i) = value * i;
	}
}

Tensor func(Tensor& x, Tensor& y)
{
	Tensor z = x + y;
	return z;
}

int main()
{

	TensorManager& manager = TensorManager::getInstance();
	Tensor t0 = Tensor(2, 2, 4); setValue(t0, 1);
	Tensor t1 = Tensor(2, 2, 4); setValue(t1, 2);
	Tensor t2 = t0 + t1;


	Tensor t3 = Tensor(2, 2, 4); setValue(t3, 2);
	Tensor t4 = Tensor(2, 2, 4); setValue(t4, 3);
	Tensor t5 = t3 + t4;

	Tensor t6 = t2 + t5;

	Tensor t8 = t6 + Tensor(2, 2, 2);
	//t8.forward();
	{
		Tensor t9 = t2 + t5;
	}

	Tensor t10 = t3 + (t0 + t6);
	Tensor t11 = func(t2, t5);

	t0.forward();
	t5.forward();
	t2.backward();

	//Tensor x0 = Tensor(1);
	//Tensor x1 = Tensor(1);
	//Tensor x2 = x0 + x1;
	//Tensor x3 = x0 + x2;
	//Tensor x4 = x0 + x3;
	//Tensor x5 = x0 + x3;
	//Tensor x6 = x4 + x5;
	//x0.forward();
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
