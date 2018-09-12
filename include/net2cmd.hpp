#ifndef  _NET2CMD_H_
#define	 _NET2CMD_H_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "base_type.hpp"

#define ifrm_xmax	127
#define	ifrm_ymax	63

#define conv_ymax	32
#define conv_xmax	64

using namespace std;



enum convsize {
	conv1 = 0,
	conv3
};

typedef struct  {
	//-- first conv stage control
	u16		ifrm_width;	//count from 1图片宽度
	u16		ifrm_height;//图片长度
	u8		conv_size;//卷积核大小最大3*3
	bool	conv_pad;//是否有零填充
	u8		conv_std;	//卷积步长
	u16		ifrm_num;//计算一个输出帧（feature map）需要多少个输入帧。从1计数。需要是8的整数倍。
	u32		ifrm_bsptr;//存放所有输入帧基地址的DDR地址指针。(*iframe_base_ptr)指向的DDR空间依次存放各个输入帧。


	bool	relu_en;//当为为TRUE第一层卷积输出为【0~127】：当为Flash第一层卷积输出为【-】
	bool	res_en;//当为TRUE feature map层累加功能，累加后再做Relu：当为Flashfeature map层累加功能
	u32		convp_bsptr;//存放卷积计算的滤波系数的地址指针。W
	u32		convk_bsptr;//存放卷积计算的K参数（每个输出帧的bias）的地址指针。b
	u32		res_bsptr;//存放Feature map累加层的数据的地址指针。
	bool	pool_en;//第一次2*2最大池化标准位

	//-- second depth wise conv  / pool stage control
	bool	dw_en;//Depth wise 卷积使能。
	bool	dw_pad;//True 零填充，Flash不需要填充
	u8		dw_std;	//步长
	u32		dwp_bsptr;//存放Depth wise卷积计算的滤波系数的地址指针。包含K参数（每个输出帧的bias）。
	bool	dw_relu_en;//1'b1:使能Relu，第二层卷积输出为[0~127]的整数；1'b0:不使能，第二层卷积输出为[-128~127]的整数。


	//-- frame output ctrl
	u16		ofrm_width;//输出帧的宽
	u16		ofrm_height;//输出帧的长
	u16		ofrm_num;//输出帧的通道数
	u32		ofrm_bsptr;//输出帧地址指针
	bool	conv_end;	//0：继续去读conv common ctrl info队列，进行卷积计算。1：卷积计算结束，发出中断。

}cal_ctrl;

typedef struct  {
	//-- first conv stage control
	u8		ifrm_xlen;	//count from 1
	u8		ifrm_ylen;
	u16		ifrm_xoff;	//line offset, pixel based
	u8		conv_size;
	bool	conv_tp;
	bool	conv_bp;
	bool	conv_lp;
	bool	conv_rp;
	u8		conv_std;	//stride: 0, stride_1; 1: stride_2
	u16		ifrm_num;
	u32		ifrm_bsptr;
	u32		ifrm_ioff;	//iframe initial pixel offset related to top-left [0,0] pixel, used for tile
	u32		ifrm_psize;	//iframe pixel size of a frame(not tiled), org_ifrm_xlen*org_ifrm_ylen
	bool	pool_en;
	bool	relu_en;
	bool	res_en;
	u32		convp_bsptr;
	u32		convk_bsptr;
	u32		res_bsptr;
	u16		res_xoff;
	u32		res_ioff;
	u32		res_psize;

	//-- second depth wise conv  / pool stage control
	bool	dw_en;
	u8		dw_ifrm_xlen;
	u8		dw_ifrm_ylen;
	bool	dw_tp;
	bool	dw_bp;
	bool	dw_lp;
	bool	dw_rp;
	u8		dw_std;	//stride: 0, stride_1; 1: stride_2
	u32		dwp_bsptr;
	bool	dw_relu_en;

	//-- frame output ctrl
	u8		ofrm_xlen;
	u8		ofrm_ylen;
	u16		ofrm_xoff;	//line offset, pixel based
	u16		ofrm_num;
	u32		ofrm_bsptr;
	u32		ofrm_ioff;	//oframe initial pixel offset related to top-left [0,0] pixel, used for tile
	u32		ofrm_psize;	//oframe pixel size of a frame(not tiled), org_ofrm_xlen*org_ofrm_ylen
	bool	firstile_layer;
	bool	lastile_layer;
	bool	conv_end;
}layer_ctrl;

typedef struct
{
	string         conv;
	u16            num;
	u16            channce;
	u16            x_size;
	u16            y_size;
}temp;



#endif

