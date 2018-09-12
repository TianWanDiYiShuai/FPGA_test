#ifndef  _WEIGHTS_VIEW_HPP_
#define	 _WEIGHTS_VIEW_HPP_


#include <opencv2/core/core.hpp>            //这三行是为了引用opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "net2cmd.hpp"
  //为了能正常编译，需要引入caffe的头文件

inline int map_func(const float value);


inline string dec2hex(u64 i, int width);



/***********************************************************************************************
用于对没成中的feature map的可视化。
**************************************************************************************************/
cv::Mat visualize_weights(std::string prototxt, std::string caffemodel, int weights_layer_num);



/******************************************************************************
用于提取模型中每层的卷积核的维度信息PW
用于提取每层feature map的输入输出信息PX
*******************************************************************************/
 void get_params_shape(std::string prototxt, std::string caffemodel, vector<temp> &pw, vector<temp> &px);


 void save_FPGA_w_b(std::string prototxt, std::string caffemodel, std::string txtpath);
 


#endif