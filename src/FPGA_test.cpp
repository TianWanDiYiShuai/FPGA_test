	// FPGA_test.cpp : 定义控制台应用程序的入口点。
//
/*******************************************************
@王洪
@2018/8/30
@成都臻识科技
@项目FPGA深度学习加速器
*******************************************************/

#include "stdafx.h"
#include <iostream>
#include<caffe/caffe.hpp>
#include <fstream>
#include "FPGA_class.hpp"
#include "weights_to_txt.hpp"

using namespace caffe;
using namespace std;



int main()
{
	//用于加载网络结构，和网络训练的模型文件，进行FPGA cmd命令字文件的生成。
	ofstream fp_cmd;
	fp_cmd.open("./cnn_cmd.txt", ios::out);
	Get_original_cmd test("deploy_VGG16.prototxt", "VGG_ILSVRC_16_layers.caffemodel", 2000);

	for (int i = 0; i < test.cmd_array_temp.size(); i++)
	{
	cout << "第"<<i<<"个卷积层："<< endl;
	Set_cmd_tile w = Set_cmd_tile(test.cmd_array_temp[i], fp_cmd);
	}

	fp_cmd.close();

	//用于生成提取的权值参数和偏置参数的函数
	save_FPGA_w_b("deploy_VGG16.prototxt", "VGG_ILSVRC_16_layers.caffemodel", "w_b.txt");

	return 0;
}

