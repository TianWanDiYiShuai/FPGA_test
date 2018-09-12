/***********************************************************
这个头文件中的内容可以参考caffe源码中
..\caffe-master\examples\cpp_classification\cpp_classification.cpp文件
***********************************************************/
#define USE_OPENCV 1
#define CPU_ONLY 1

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <caffe/caffe.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>   //使用智能指针
#include <utility>
#include <vector>



using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
/***********************************************************
Classifier这个类的主要功能是初始化了一个CNN分类器，读入已经迭代训练好的模型，
根据前向传播算法，对要进行预测的数据集或者单个输入图片进行预测。
***********************************************************/
class Classifier {
public:
	/*************************************************
	这个构造函数用于读入要预测的数据集，用于预测
	多张图片，一般为LeveDB LMDB数据库文件
	***********************************************/
	Classifier(const string& model_file, const string& trained_file);

	/****这个构造函数用于对单个的一张OpenCV对象的图片进行预测*******/
	int Classify(const cv::Mat& img);

	/*************************************************
	用于对输入图片，在某一层的feature map的可视化
	***********************************************/
	cv::Mat visualize_featuremap(const cv::Mat& img, string layer_name);
private:
	//void SetMean(const string& mean_file);

	/***********************************************
	这个函数，给它一张测试图片，然后它就把这张图片，丢进网络里，进行一次前向传播，
	然后再把网络的输出返回，也就是将一张图片，映射成一个特征向量
	*************************************************/
	std::vector<int> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);



private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	//cv::Mat mean_;
	std::vector<string> labels_;
};




