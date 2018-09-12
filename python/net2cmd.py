# -*- coding: UTF-8 -*-
# date:2018/6/28
# User:WangHong
import numpy as np

#struct cal_ctrl从原始数据中获取到的结构体数据
cal_ctrl = np.dtype({'names':['ifrm_width',#图片宽度  y
                              'ifrm_height',#图片长度  y
                              'conv_size',#卷积核大小最大3*3   y
                              'conv_pad',#是否有零填充  y
                              'conv_std',#卷积步长   y
                              'ifrm_num',#计算一个输出帧（feature map）需要多少个输入帧。从1计数。需要是8的整数倍。 y
                              'ifrm_bsptr',#存放所有输入帧基地址的DDR地址指针。(*iframe_base_ptr)指向的DDR空间依次存放各个输入帧。   t


                              'relu_en',#当为为TRUE第一层卷积输出为【0~127】：当为Flash第一层卷积输出为【-】
                              'res_en',#当为TRUE feature map层累加功能，累加后再做Relu：当为Flashfeature map层累加功能
                              'pool_en',#第一次2*2最大池化标准位


                              'convp_bsptr',#存放卷积计算的滤波系数的地址指针。W     t
                              'convk_bsptr',#存放卷积计算的K参数（每个输出帧的bias）的地址指针。b  t
                              'res_bsptr',#存放Feature map累加层的数据的地址指针。  t

                                #----------------------second depth wise conv----------------------
                              'dw_en',#Depth wise 卷积使能。
                              'dw_pad',#True 零填充，Flash不需要填充
                              'dw_std',#步长
                              'dwp_bsptr',#存放Depth wise卷积计算的滤波系数的地址指针。包含K参数（每个输出帧的bias）。
                              'dw_relu_en',#1'b1:使能Relu，第二层卷积输出为[0~127]的整数；1'b0:不使能，第二层卷积输出为[-128~127]的整数。

                                #-------------------------- frame output ctrl-------------------------
                              'ofrm_width',#输出帧的宽
                              'ofrm_height',#输出帧的长
                              'ofrm_num',#输出帧的通道数
                              'ofrm_bsptr',#输出帧地址指针
                              'conv_end'],#继续去读conv common ctrl info队列，进行卷积计算。1：卷积计算结束，发出中断。
                   'formats':[np.uint16,np.uint16,np.uint8 ,bool,np.uint8,np.uint16,np.uint32,bool,bool,bool,np.uint32,np.uint32,np.uint32,
                                bool,bool,np.uint8,np.uint32,bool,np.uint16,np.uint16,np.uint16,np.uint32,bool]},align=True)#结构体中数据类型


layer_ctrl = np.dtype({'names':['ifrm_xlen',
                               'ifrm_ylen',
                               'ifrm_xoff',
                               'conv_size',
                               'conv_tp',
                               'conv_bp',
                               'conv_lp',
                               'conv_rp',
                               'conv_std',
                               'ifrm_num',
                               'ifrm_bsptr',
                               'ifrm_ioff',
                               'ifrm_psize',
                               'pool_en',
                               'relu_en',
                               'res_en',
                               'convp_bsptr',
                               'convk_bsptr',
                               'res_bsptr',
                               'res_xoff',
                               'res_ioff',
                               'res_psize',
                               'dw_en',
                               'dw_ifrm_xlen',
                               'dw_ifrm_ylen',
                               'dw_tp',
                               'dw_bp',
                               'dw_lp',
                               'dw_rp',
                               'dw_std',
                               'dwp_bsptr',
                               'dw_relu_en',
                               'ofrm_xlen',
                               'ofrm_ylen',
                               'ofrm_xoff',
                               'ofrm_num',
                               'ofrm_bsptr',
                               'ofrm_ioff',
                               'ofrm_psize',
                               'firstile_layer',
                               'lastile_layer',
                               'conv_end'],
                       'formats':[np.uint8,np.uint8,np.uint16,np.uint8,
                                  bool,bool,bool,bool,np.uint8,np.uint16,
                                  np.uint32,np.uint32,np.uint32,bool,bool,
                                  bool,np.uint32,np.uint32,np.uint32,
                                  np.uint16,np.uint32,np.uint32,bool,
                                  np.uint8,np.uint8,bool,bool,bool,bool,
                                  np.uint8,np.uint32,bool,np.uint8,np.uint8,
                                  np.uint16,np.uint16,np.uint32,np.uint32,
                                  np.uint32,bool,bool,bool]},align=True)