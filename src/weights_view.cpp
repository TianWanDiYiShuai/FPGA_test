#define USE_OPENCV 1
#define CPU_ONLY 1

#include "stdafx.h"
#include "weights_view.hpp"
#include <caffe/caffe.hpp>    //为了能正常编译，需要引入caffe的头文件
#include "caffe.pb.h"
#include <algorithm>
#include <iosfwd>
#include <memory>                  //使用c++智能指针，必须包含该目录
#include <utility>
#include <math.h>
#include "head.h"




using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
//using namespace std;

/******************************************************************************
归一化函数，用于对数据的归一化处理
*******************************************************************************/
inline int map_func(const float value)
{

	int result;
	result = int(value * 128);
	return result;
}


/******************************************************************************
将数据转化为16进制的字符串，字符串长度为16位，在字符串头，会加上标识符“ox”
*******************************************************************************/
inline string dec2hex(u64 i, int width)
{
	std::stringstream ioss; //定义字符串流
	std::string s_temp; //存放转化后字符
	ioss << std::hex << i; //以十六制形式输出
	ioss >> s_temp;
	std::string s(width - s_temp.size(), '0'); //补0
	s = "ox"+s+ s_temp; //合并
	return s;
}

/***********************************************************************************************
用于对没成中的feature map的可视化。
**************************************************************************************************/
cv::Mat visualize_weights(string prototxt, string caffemodel, int weights_layer_num)
{

	::google::InitGoogleLogging("0");
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	//初始化一个网络，网络结构从caffenet_deploy.prototxt文件中读取，TEST表示是测试阶段
	Net<float> net(prototxt, TEST);
	net.CopyTrainedLayersFrom(caffemodel);   //读取已经训练好的model的参数
	vector<boost::shared_ptr<Blob<float> > > params = net.params();    //获取网络的各个层学习到的参数(权值+偏置)

	//打印出该model学到的各层的参数的维度信息
	std::cout << "各层参数的维度信息为：\n";
	for (int i = 0; i<params.size(); ++i)
		std::cout << params[i]->shape_string() << std::endl;


	int width = params[weights_layer_num]->shape(3);     //宽度,第一个卷积层为11 
	int height = params[weights_layer_num]->shape(2);    //高度，第一个卷积层为11
	int channel = params[weights_layer_num]->shape(1);		//通道数
	int num = params[weights_layer_num]->shape(0);       //卷积核的个数


	//我们将num个图，放在同一张大图上进行显示，此时用OpenCV进行可视化，声明一个大尺寸的图片，使之能容纳所有的卷积核图	
	int imgHeight = (int)(1 + sqrt(num))*height;  //大图的尺寸	
	int imgWidth = (int)(1 + sqrt(num))*width;
	Mat img(imgHeight, imgWidth, CV_8UC3, Scalar(0, 0, 0));

	//各层的权值，是一个可正可负的实数，而在OpenCV里的一般图片，每个像素的值在0~255之间	
	//对权值进行归一化到0~255才能正常显示
	float maxValue = -1000, minValue = 10000;
	const float* tmpValue = params[weights_layer_num]->cpu_data();   //获取该层的参数，实际上是一个一维数组	
	for (int i = 0; i<params[weights_layer_num]->count(); i++){        //求出最大最小值		
		maxValue = std::max(maxValue, tmpValue[i]);
		minValue = std::min(minValue, tmpValue[i]);
	}
	//对最终显示的大尺寸图片，进行逐个像素赋值
	int kk = 0;                         //此时在画第kk个卷积核
	for (int y = 0; y<imgHeight; y += height){
		for (int x = 0; x<imgWidth; x += width){
			if (kk >= num)
				continue;
			Mat roi = img(Rect(x, y, width, height));
			for (int i = 0; i<height; i++){
				for (int j = 0; j<width; j++){
					for (int k = 0; k<channel; k++){
						float value = params[weights_layer_num]->data_at(kk, k, i, j);

						roi.at<Vec3b>(i, j)[k] = (value - minValue) / (maxValue - minValue) * 255;   //归一化到0~255
					}
				}
			}
			++kk;
		}
	}
	resize(img, img, Size(500, 500));   //将显示的大图，调整为500*500尺寸	
	imshow("conv1", img);              //显示	
	waitKey(0);
	return img;
}


/******************************************************************************
用于提取模型中每层的卷积核的维度信息PW
用于提取每层feature map的输入输出信息PX
*******************************************************************************/
void get_params_shape(std::string prototxt, std::string caffemodel, vector<temp> &pw, vector<temp> &px)
{
	//初始化一个网络，网络结构从caffenet_deploy.prototxt文件中读取，TEST表示是测试阶段
	Net<float> net(prototxt, TEST);
	net.CopyTrainedLayersFrom(caffemodel);   //读取已经训练好的model的参数


	vector<boost::shared_ptr<Blob<float> > > params = net.params();    //获取网络的各个层学习到的参数(权值+偏置)
	string para_names = "conv";

	vector<boost::shared_ptr<Blob<float>>> blobs = net.blobs();//得到各层的输出特征向量
	vector<string> blob_names = net.blob_names();

	cout << "卷积核层数为：" << params.size()/2 << endl;
	for (int i = 0; i < params.size(); ++i)
	{
		cout << "每层维度：" << params[i]->num_axes()<< endl;
		temp temp_weights_shape;
		int index = params[i]->num_axes();
		if (i % 2 == 0)
		{
			if (index==4)
			{
				temp_weights_shape.channce = params[i]->shape(0);	                     //图片数量
				temp_weights_shape.num = params[i]->shape(1);                            //通道数
				temp_weights_shape.x_size = params[i]->shape(2);                         //宽度
				temp_weights_shape.y_size = params[i]->shape(3);                         //高度
			}
			else if (index == 2)
			{
				temp_weights_shape.channce = params[i]->shape(0);	                     //图片数量
				temp_weights_shape.num = params[i]->shape(1);                            //通道数
				temp_weights_shape.x_size = 1;                                           //宽度
				temp_weights_shape.y_size = 1;                                           //高度
			}
			else cout << "出现错误1" << endl;

			temp_weights_shape.conv = para_names;
			pw.push_back(temp_weights_shape);
		}
	}
	//提取feature map的维度信息，
	cout << "feature map层数为：" << blobs.size() << endl;
	for (int i = 0; i < blobs.size(); ++i)
	{
		temp temp_feature_shape; 
		int index = blobs[i]->num_axes(); //获取每层的shape大小，每层的shape大小不定，有4维，和2维
		if (index==4)//在四维情况下提取数据
		{
			temp_feature_shape.num = blobs[i]->shape(0);
			temp_feature_shape.channce = blobs[i]->shape(1);
			temp_feature_shape.x_size = blobs[i]->shape(2);
			temp_feature_shape.y_size = blobs[i]->shape(3);
		}
		else if (index == 2)//在二维其情况下提取数据
		{
			temp_feature_shape.num = blobs[i]->shape(0);
			temp_feature_shape.channce = blobs[i]->shape(1);
			temp_feature_shape.x_size = 1;
			temp_feature_shape.y_size = 1;
		}
		else cout << "出现错误2" << endl;//在维度不符合的情况下报错
		temp_feature_shape.conv = blob_names[i];
		px.push_back(temp_feature_shape);
	}
}



/******************************************************************************
按照FPGA工程师要求的参数格式设计的，参数提取，归一化，转化算法
*******************************************************************************/


void save_FPGA_w_b(std::string prototxt, std::string caffemodel, std::string txtpath)
{
	::google::InitGoogleLogging("0");
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	
	//初始化一个网络，网络结构从caffenet_deploy.prototxt文件中读取，TEST表示是测试阶段
	Net<float> net(prototxt, TEST);
	net.CopyTrainedLayersFrom(caffemodel);   //读取已经训练好的model的参数
	vector<boost::shared_ptr<Blob<float> > > params = net.params();    //获取网络的各个层学习到的参数(权值+偏置)

	//打印出该model学到的各层的参数的维度信息
	std::cout << "各层参数的维度信息为：\n";
	cout << "wangluode cehngshu " << params.size() << endl;
	for (int i = 0; i<params.size(); ++i)
		std::cout << params[i]->shape_string() << std::endl;

	int size = params.size() / 2;
	
	bool zero_fill;
	int cyc_one,cyc_two,cyc_three,cyc_four,cyc_b;
	int one_index, two_index, three_index, four_index,b_index;
	int value = 0,value_b=0;
	ofstream fp_w_b;
	fp_w_b.open("w_b.txt", ios::out);
	/********************************************************
	1：每个输出帧对应的8个输入帧的卷积参数掺合在一起，占用1个地址。
	2：然后按输出帧0~15的顺序，摆放(0,0)点的参数，共占用16个地址。
	3：按1),2)的定义，如果做3x3卷积，依次摆放(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)点的卷积参数，共占用144个地址；如果做1x1的卷积，只摆放(0,0)点的参数。
	4：按1)~3)的定义，摆放剩余输入帧（8 ~ C_MAX）对于0～15个输出帧的卷积参数。
	5：按1)~4)的定义，摆放16 ~ N_MAX的卷积参数。
	6：在把每一层的权值参数提取完后，在提取偏置参数
	7：按1)~6)的定义，摆放网络每层的卷积参数
	********************************************************/

	for (int layer_index = 0; layer_index < size; ++layer_index)
	{
		int conv_index = layer_index * 2;

		int w_shanpe = params[conv_index]->num_axes();
		cout << "这一层的shape" << w_shanpe << endl;

		if (w_shanpe != 4) break;
		if (params[conv_index]->shape(1) < 8)
		{
			zero_fill = true;
			cyc_one = 1;
		}
		else
		{
			cyc_one = params[conv_index]->shape(1) / 8;
			zero_fill = false;
		}
		cyc_two = params[conv_index]->shape(0) / 16;
		cyc_b = params[conv_index]->shape(0) / 8;
		cyc_three = params[conv_index]->shape(2);
		cyc_four = params[conv_index]->shape(3);
		for (int k1 = 0; k1 < cyc_one; k1++)
		{
			for (int k2 = 0; k2 < cyc_two; k2++)
			{
				for (int k3 = 0; k3 < cyc_three; k3++)
				{
					for (int k4 = 0; k4 < cyc_four; k4++)
					{
						for (int k5 = 0; k5 < 16; k5++)
						{
							u64 kay_value = 0;
							if (zero_fill == false)
							{
								//提取权重参数，并做归一化处理，转化为16进制长为16 位的字符串
								//提取feature map输入通道为8 的倍数的权重
								for (int k6 = 0; k6 < 8; k6++)
								{
									one_index = k2 * 16 + k5;
									two_index = k1 * 8 + k6;
									three_index = k3;
									four_index = k4;
									value = map_func(params[conv_index]->data_at(one_index, two_index, three_index, four_index));
									if (value < 0)value = value + 256;
									u64 temp = value*pow(256, k6);
									kay_value = kay_value + temp;
								}
							}
							else
							{							
								//提取feature map输入通道不为8的倍数的权重，当不满足8通道时其余通道补零
								for (int k6 = 0; k6 < 3; k6++)
								{
									one_index = k2 * 16 + k5;
									two_index = k1 * 8 + k6;
									three_index = k3;
									four_index = k4;
									value = map_func(params[conv_index]->data_at(one_index, two_index, three_index, four_index));
									if (value < 0)value = value + 256;
									u64 temp = value*pow(256, k6);
									kay_value = kay_value + temp;
								}
							}
							string temp_string = dec2hex(kay_value, 16);
							fp_w_b << "*((int *)(cfg_addr + " << k5 << "+ * 8)) = " << temp_string << ";" << endl;
						}
						fp_w_b << "cfg_addr += 16*8;" << endl;
					}
				}
			}
		}
		//提取偏置参数
		for (int kb = 0; kb < cyc_b; kb++)
		{
			u64 kay_value = 0;
			for (int kb1 = 0; kb1 < 8; kb1++)
			{
				
				b_index = kb * 8 + kb1;
				value_b = map_func(params[conv_index + 1]->data_at(b_index,0,0,0));//做归一化处理
				if (value_b < 0)value_b = value_b + 256;
				u64 temp = value_b*pow(256, kb1);
				kay_value = kay_value + temp;
			}
			string temp_string = dec2hex(kay_value, 16);//转化为16进制的长为10位字符串
			
			fp_w_b << "*((int *)(cfg_addr + " << kb << "+ * 8)) = " << temp_string << ";" << endl;
		}
		//fp_w_b << "这里是偏置参数" << endl;
		fp_w_b << "cfg_addr += " << cyc_b << "*8;" << endl;
	}
	fp_w_b.close();
}
