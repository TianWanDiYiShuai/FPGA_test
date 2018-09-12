# -*- coding: UTF-8 -*-
# date:2018/6/28
# User:WangHong
import pandas as pd
import re


from  net2cmd  import*


class set_cmd_cal():
    '''
    原始图片cmd设置类
    '''
    define_cmd_size = 64#cmd控制字的大小，固定大小，单位B
    __w_b_address = 0

    def __init__(self,path_x,path_w,start_memory = 0):
        '''
        初始化类的参数，用于在创建set_cmd_cal类的实例时，初始化对象
        :param path_x:存放feature map参数文件的地址
        :param path_w:存放卷积核参数文件的地址
        :param start_memory:FPGA提供的内存的起始地址
        '''
        self.path_x = path_x
        self.path_w = path_w
        self.start_memory = start_memory
        pf_w = pd.read_csv(self.path_w, header=None, sep=',')
        self.conv_size = pf_w.shape[0]-1
        self.array = self.read_csv(self.path_x)
        self.feature_map_max = self.get_feature_map_max()
        self.cmd_array = np.empty(self.conv_size, dtype=cal_ctrl)
        self.__w_b_address = self.feature_map_max * 2 + self.start_memory
        self.get_conv_cmd()

    def read_csv(self,path):
        '''
        :param path: 文件路径
        :return: 返回存储文件信息的列表
        '''
        pf = pd.read_csv(path,header=None,sep=',')
        temp=[]
        for i in range(1,pf.shape[0]):
            W = []
            W.append(pf[0][i])
            for j in range(1,pf.shape[1]):
                W.append(int(pf[j][i]))
            temp.append(W)
        return temp

    def get_conv_cmd(self):
        '''
        给原始结构体赋值，为类set_cmd_cal顶层逻辑函数
        :return:
        '''
        array_index = 0
        for i in range(self.conv_size):
            # 判断第i层的类型
            #print(i)
            self.conv_layer(i, array_index)
            if re.match(r'pool(.*)',self.array[array_index+1][0]) !=None:#是池化层
                self.cmd_array[i]['pool_en'] = True
                array_index = array_index+2
            elif re.match(r'fc(.*)',self.array[i][0]) !=None:#是全连接层
                #break
                array_index = array_index + 1
            else:#是卷积层
                self.cmd_array[i]['pool_en'] = False
                array_index = array_index + 1

    def conv_layer(self,i,array_index):
        self.cmd_array[i]['ifrm_width'] = self.array[array_index][3]  # 图片宽度
        self.cmd_array[i]['ifrm_height'] = self.array[array_index][4]  # 图片长度
        self.cmd_array[i]['conv_size'] = 3  # 卷积核大小最大3*3
        self.cmd_array[i]['conv_pad'] = True  # 是否有零填充
        self.cmd_array[i]['conv_std'] = 1  # 卷积步长
        if self.array[array_index][2] < 8:  # input->output
            self.cmd_array[i]['ifrm_num'] = 8
        else:
            self.cmd_array[i]['ifrm_num'] = self.array[array_index][2]
        # self.cmd_array[i]['ifrm_bsptr'] = self.array[i][3]#存放所有输入帧基地址的DDR地址指针   yes
        self.cmd_array[i]['relu_en'] = True  # relu使能
        self.cmd_array[i]['res_en'] = False  # 当为TRUE feature map层累加功能使能
        self.set_w_b_address(i)
        #self.cmd_array[i]['convp_bsptr'] = 0#存放卷积计算的滤波系数的地址指针   no
        #self.cmd_array[i]['convk_bsptr'] = 0#存放卷积计算的b参数  no
        self.cmd_array[i]['res_bsptr'] = 0#存放Feature map累加层的数据的地址指针 no
        #self.cmd_array[i]['pool_en'] = self.array[array_index][3]#第一次2*2最大池化标准位
        # Depth wise
        self.cmd_array[i]['dw_en'] = False  # Depth wise 卷积使能。
        self.cmd_array[i]['dw_pad'] = False  # Depth wise True 零填充，Flash不需要填充
        self.cmd_array[i]['dw_std'] = 0  # 步长
        self.cmd_array[i]['dwp_bsptr'] = 0  # 存放Depth wise卷积计算的滤波系数的地址指针W+b
        self.cmd_array[i]['dw_relu_en'] = False  # 使能Relu，第二层卷积输出为[0~127]的整数
        # frame output ctrl
        self.cmd_array[i]['ofrm_width'] = self.array[array_index+1][3]#输出帧的宽
        self.cmd_array[i]['ofrm_height'] = self.array[array_index+1][4]#输出帧的长
        self.cmd_array[i]['ofrm_num'] = self.array[array_index+1][2]#输出帧的通道数
        # self.cmd_array[i]['ofrm_bsptr'] = self.array[array_index][3]#输出帧地址指针   yes
        # self.cmd_array[i]['conv_end'] = self.array[array_index][3]#继续去读conv common ctrl info队列    yes
        self.set_feature_map_address(i)
        self.set_conv_end(i)

    def set_w_b_address(self,i):
        '''
        设置w和b的地址
        :return:
        '''
        pf_w = self.read_csv(self.path_w)
        self.cmd_array[i]['convp_bsptr'] = self.__w_b_address
        b_address = self.__w_b_address+pf_w[i][1]*pf_w[i][2]*pf_w[i][3]*pf_w[i][4]/8
        self.cmd_array[i]['convk_bsptr'] = b_address
        self.__w_b_address = b_address+pf_w[i][1]/8

    def set_feature_map_address(self,i):
        '''
        设置feature_map的地址
        :return:
        '''
        if i%2==0:
            self.cmd_array[i]['ifrm_bsptr'] = self.start_memory
            self.cmd_array[i]['ofrm_bsptr'] = self.start_memory+self.feature_map_max
        else:
            self.cmd_array[i]['ifrm_bsptr'] = self.start_memory+self.feature_map_max
            self.cmd_array[i]['ofrm_bsptr'] = self.start_memory

    def get_feature_map_max(self):
        '''
        计算feature_map_max
        :return:
        '''
        feature_map_max = 0
        for j in range(len(self.array)):#计算出feature_map最大值
            temp =self.array[j][2]*self.array[j][3]*self.array[j][4]
            if temp >=feature_map_max:
                feature_map_max = temp
        return feature_map_max/8

    def set_conv_end(self,i):
        if i == self.conv_size-1:
            self.cmd_array[i]['conv_end'] = False
        else:
            self.cmd_array[i]['conv_end'] = True

    def get_memory_use(self):
        '''
        获得现在使用的内存大小
        :return:
        '''
        # print((self.__w_b_address-self.start_memory)*8,'B')
        # print((self.__w_b_address-self.start_memory)*8/1024**2,'M')
        return (self.__w_b_address-self.start_memory)*8


class set_cmd_tile():
    '''
    分tile的类
    '''
    tile_number = 0
    def __init__(self,org_array,file):
        self.file = file
        self.org_array = org_array
        self.get_layer_tile()
        self.set_cmd_array()

    def get_layer_tile(self):
        '''给每层分tile
        :param i: 层的索引
        :return:
        '''
        if self.org_array['conv_std'] == 1:
            self.x_number = int(self.org_array['ifrm_width'] / 64)+1 # 计算X方向的分tile个数
            self.y_number = int(self.org_array['ifrm_height'] / 32)+1 # 计算Y方向的分tile个数
        elif self.org_array['conv_std'] == 2:
            self.x_number = int(self.org_array['ifrm_width'] / 128) + 1  # 计算X方向的分tile个数
            self.y_number = int(self.org_array['ifrm_height'] / 64) + 1  # 计算Y方向的分tile个数
        self.tile_number = self.x_number*self.y_number
        self.tile_cmd_array = np.empty(self.tile_number, dtype=layer_ctrl)
        for y_index in range(0,self.y_number):
            for x_index in range(0,self.x_number):
                index = y_index*self.x_number + x_index
                self.set_black_param(index)
                self.set_blue_param(index,x_index,y_index)
                if x_index == 0:
                    self.tile_cmd_array[index]['conv_lp'] = True
                else:
                    self.tile_cmd_array[index]['conv_lp'] = False
                if x_index == self.x_number-1:
                    self.tile_cmd_array[index]['conv_rp'] = True
                else:
                    self.tile_cmd_array[index]['conv_rp'] = False
                if y_index ==0:
                    self.tile_cmd_array[index]['conv_tp'] = True
                else:
                    self.tile_cmd_array[index]['conv_tp'] = False
                if y_index == self.y_number-1:
                    self.tile_cmd_array[index]['conv_bp'] = True
                else:
                    self.tile_cmd_array[index]['conv_bp'] = False
                self.get_error(index)
        #print(self.tile_cmd_array)

    def get_error(self,i):
        '''
        检查分tile的逻辑错误
        :param i:
        :return:
        '''
        if self.tile_cmd_array[i]['dw_en'] == True and self.tile_cmd_array[i]['pool_en']==True:
            print("Command Error, both dw_en and pool_en are enabled./n")
            exit()
        if self.tile_cmd_array[i]['dw_std'] == True and self.tile_cmd_array[i]['pool_en']==True:
            print("Command Error, both dw_std and pool_en are enabled./n")
            exit()
        if self.tile_cmd_array[i]['dw_std'] == True and not(self.tile_cmd_array[i]['pool_en']==True or self.tile_cmd_array[i]['dw_en']==True):
            print("Command Error, dw_pad enabled in 2nd stage bypass mode./n")
            exit()

    def set_black_param(self,i):
        '''
        设置结构体的蓝色参数
        :param i: 结构体的索引
        :return:
        '''
        self.tile_cmd_array[i]['ifrm_xoff'] = self.org_array['ifrm_width']
        self.tile_cmd_array[i]['conv_size']=self.org_array['conv_size']
        self.tile_cmd_array[i]['conv_std'] =self.org_array['conv_std']
        self.tile_cmd_array[i]['ifrm_num'] =self.org_array['ifrm_num']
        self.tile_cmd_array[i]['ifrm_bsptr'] =self.org_array['ifrm_bsptr']
        self.tile_cmd_array[i]['ifrm_psize'] =self.org_array['ifrm_width']*self.org_array['ifrm_height']
        self.tile_cmd_array[i]['pool_en'] =self.org_array['pool_en']
        self.tile_cmd_array[i]['relu_en'] =self.org_array['relu_en']
        self.tile_cmd_array[i]['res_en'] =self.org_array['res_en']
        self.tile_cmd_array[i]['convp_bsptr'] =self.org_array['convp_bsptr']
        self.tile_cmd_array[i]['convk_bsptr'] =self.org_array['convk_bsptr']
        self.tile_cmd_array[i]['res_bsptr'] =0# 没有累加层都置为零
        self.tile_cmd_array[i]['res_xoff'] =0# 没有累加层都置为零
        self.tile_cmd_array[i]['res_psize'] = 0  # 没有累加层都置为零
        self.tile_cmd_array[i]['dw_en'] =self.org_array['dw_en']
        self.tile_cmd_array[i]['dw_std'] =self.org_array['dw_std']
        self.tile_cmd_array[i]['dwp_bsptr'] =self.org_array['dwp_bsptr']
        self.tile_cmd_array[i]['dw_relu_en'] = self.org_array['dw_relu_en']
        self.tile_cmd_array[i]['ofrm_num'] = self.org_array['ofrm_num']
        self.tile_cmd_array[i]['ofrm_bsptr'] = self.org_array['ofrm_bsptr']
        self.tile_cmd_array[i]['ofrm_psize'] = self.org_array['ofrm_width']*self.org_array['ofrm_height']
        if i == 0:
            self.tile_cmd_array[i]['firstile_layer'] = True#没有完成
            self.tile_cmd_array[i]['lastile_layer'] = False#没有完成
        elif i == self.tile_number-1:
            self.tile_cmd_array[i]['firstile_layer'] = False#没有完成
            self.tile_cmd_array[i]['lastile_layer'] = True#没有完成
        else:
            self.tile_cmd_array[i]['firstile_layer'] = False#没有完成
            self.tile_cmd_array[i]['lastile_layer'] = False#没有完成
        self.tile_cmd_array[i]['conv_end'] = self.org_array['conv_end']

    def set_blue_param(self,i,x_index,y_index):
        '''
        设置蓝色参数的值
        :param i:
        :return:
        '''
        if self.org_array['conv_std']==1:
            if x_index+1 == self.x_number and y_index+1 != self.y_number:
                self.tile_cmd_array[i]['ifrm_xlen'] = self.org_array['ifrm_width']-(self.x_number-1)*63
                self.tile_cmd_array[i]['ifrm_ylen'] = 32
            elif x_index+1 != self.x_number and y_index+1 == self.y_number:
                self.tile_cmd_array[i]['ifrm_xlen'] = 64
                self.tile_cmd_array[i]['ifrm_ylen'] = self.org_array['ifrm_height']-(self.y_number-1)*31
            elif x_index+1 == self.x_number and y_index+1 == self.y_number:
                self.tile_cmd_array[i]['ifrm_xlen'] = self.org_array['ifrm_width'] - (self.x_number - 1) * 63
                self.tile_cmd_array[i]['ifrm_ylen'] = self.org_array['ifrm_height'] - (self.y_number - 1) * 31
            else:
                self.tile_cmd_array[i]['ifrm_xlen'] = 64
                self.tile_cmd_array[i]['ifrm_ylen'] = 32
        elif self.org_array['conv_std']==2:
            if x_index+1 == self.x_number and y_index+1 != self.y_number:
                self.tile_cmd_array[i]['ifrm_xlen'] = self.org_array['ifrm_width']-(self.x_number-1)*127
                self.tile_cmd_array[i]['ifrm_ylen'] = 63
            elif x_index+1 != self.x_number and y_index+1 == self.y_number:
                self.tile_cmd_array[i]['ifrm_xlen'] = 127
                self.tile_cmd_array[i]['ifrm_ylen'] = self.org_array['ifrm_height']-(self.y_number-1)*63
            elif x_index+1 == self.x_number and y_index+1 == self.y_number:
                self.tile_cmd_array[i]['ifrm_xlen'] = self.org_array['ifrm_width'] - (self.x_number - 1) * 127
                self.tile_cmd_array[i]['ifrm_ylen'] = self.org_array['ifrm_height'] - (self.y_number - 1) * 63
            else:
                self.tile_cmd_array[i]['ifrm_xlen'] = 127
                self.tile_cmd_array[i]['ifrm_ylen'] = 63
        self.tile_cmd_array[i]['res_ioff'] = 0# 没有累加层都置为零
        self.dw_param_set(i)
        self.tile_cmd_array[i]['ofrm_xoff'] = self.org_array['ofrm_width']
        self.set_tile_address(i,x_index,y_index)

    def set_tile_address(self,i,x_index,y_index):
        '''
        计算每个tile经过卷积后对应的输出tile的长，宽和下一个tile的偏移量
        :param i:
        :param x_index:x方向的偏移量
        :param y_index:y方向的偏移量
        :return:
        '''
        if x_index==0 and y_index==0:
            self.tile_cmd_array[i]['ifrm_ioff'] = 0
        else:
            self.tile_cmd_array[i]['ifrm_ioff'] = (y_index*self.org_array['ifrm_width']*(self.tile_cmd_array[i]['ifrm_ylen']-2)+ \
                                               x_index*(self.tile_cmd_array[i]['ifrm_xlen']-1)-1)#未完成
        if self.tile_cmd_array[i]['pool_en'] == False:
            self.tile_cmd_array[i]['ofrm_xlen'] = 64
            self.tile_cmd_array[i]['ofrm_ylen'] = 32
            self.tile_cmd_array[i]['ofrm_ioff'] = self.tile_cmd_array[i]['ifrm_ioff']
        elif self.tile_cmd_array[i]['pool_en'] == True:
            self.tile_cmd_array[i]['ofrm_xlen'] = 32
            self.tile_cmd_array[i]['ofrm_ylen'] = 16
            if y_index == 0 and x_index==0:
                self.tile_cmd_array[i]['ofrm_ioff']=0
            elif y_index ==0 and x_index != 0:
                self.tile_cmd_array[i]['ofrm_ioff'] = x_index*(self.tile_cmd_array[i]['ofrm_xlen'])-1
            elif y_index!=0 and x_index == 0:
                self.tile_cmd_array[i]['ofrm_ioff'] = (y_index * self.tile_cmd_array[i]['ofrm_ylen'] - 1) * self.org_array['ofrm_width']-1
            else:
                self.tile_cmd_array[i]['ofrm_ioff'] = (y_index * self.tile_cmd_array[i]['ofrm_ylen'] - 1) * self.org_array['ofrm_width'] + \
                                                      x_index * (self.tile_cmd_array[i]['ofrm_xlen']) - 1

    def dw_param_set(self,i):
        '''
        设置深度卷积参数
        :return:
        '''
        if self.tile_cmd_array[i]['dw_en']==0:#没有深度卷积操作时，参数的设置
            self.tile_cmd_array[i]['dw_ifrm_xlen'] = 0
            self.tile_cmd_array[i]['dw_ifrm_ylen'] = 0
            self.tile_cmd_array[i]['dw_tp'] = 0
            self.tile_cmd_array[i]['dw_bp'] = 0
            self.tile_cmd_array[i]['dw_lp'] = 0
            self.tile_cmd_array[i]['dw_rp'] = 0
            self.tile_cmd_array[i]['dw_std'] = 0
            self.tile_cmd_array[i]['dwp_bsptr'] = 0
            self.tile_cmd_array[i]['dw_relu_en'] = 0
        else:#具有深度卷积操作时，参数的设置
            pass

    def set_cmd_array(self):
        '''
        根据分tile的数量创建cmd命令字数据结构
        :return:
        '''
        self.cmd = np.empty((self.tile_number,8), dtype=np.uint64)
        for i in range(self.tile_number):
            #print("第"+repr(i)+"个tile")
            #self.file.write(u"第"+repr(i)+"个tile")
            self.cmd[i] = self.ctrl_pack(self.tile_cmd_array[i])
            self.file.write('\n')
            self.file.write('\n')

    def ctrl_pack(self,cmd_list):
        '''
        cmd命令生成函数
        :param tile_cmd_array:
        :return:
        '''
        cmd = np.empty(8,dtype=np.uint64)
        cmd[0] = ((int(cmd_list['ifrm_xlen']) << 0)|(int(cmd_list['ifrm_ylen'])<<7)
                 |(int(cmd_list['ifrm_xoff']) <<14)|(int(cmd_list['conv_size'])<<25)
                 |(int(cmd_list['conv_tp']) << 27 )|(int(cmd_list['conv_bp']) << 28)
                 |(int(cmd_list['conv_lp']) << 29 )|(int(cmd_list['conv_rp']) << 30)
                 |(int(cmd_list['conv_std']) << 31)|(int(cmd_list['ifrm_num'])<< 32)
                 |((int(cmd_list['ifrm_psize']) & 0x1ff)<< 45)
                  )

        cmd[1] = ((int(cmd_list['ifrm_bsptr']) << 0)|(int(cmd_list['ifrm_ioff']) << 32)
                 |(((int(cmd_list['ifrm_psize']) >> 9)& 0x1fff) << 54)
                 |(int(cmd_list['pool_en']) << 57 )|(int(cmd_list['relu_en']) << 58)
                 |(int(cmd_list['res_en']) << 59 )
                  )

        cmd[2] = ((int(cmd_list['convp_bsptr']) << 0 )|(int(cmd_list['convk_bsptr']) << 32 ))

        cmd[3] = ((int(cmd_list['dw_en'])<<0)|(int(cmd_list['dw_ifrm_xlen'])<<1)
                 |(int(cmd_list['dw_ifrm_ylen'])<<8)|(int(cmd_list['dw_tp'])<<15)
                 |(int(cmd_list['dw_bp']) << 16 )|(int(cmd_list['dw_lp'])<<17)
                 |(int(cmd_list['dw_rp']) << 18 )|(int(cmd_list['dw_std'])<<19)
                 |(int(cmd_list['dwp_bsptr']) << 20)|(int(cmd_list['dw_relu_en'])<< 52)
                 )

        cmd[4] = ((int(cmd_list['ofrm_xlen']) << 0)|(int(cmd_list['ofrm_ylen']) << 7)
                 |(int(cmd_list['ofrm_xoff']) <<14)|(int(cmd_list['ofrm_num']) <<25)
                 |(int(cmd_list['ofrm_psize']) << 38 )
                  )

        cmd[5] = ((int(cmd_list['ofrm_bsptr']) << 0)|(int(cmd_list['ofrm_ioff']) << 32)
                 |(int(cmd_list['firstile_layer']) <<54)|(int(cmd_list['lastile_layer']) <<55)
                 |(int(cmd_list['conv_end']) << 63 )
                  )

        cmd[6] = ((int(cmd_list['res_bsptr']) << 0 )|(int(cmd_list['res_xoff']) << 32 ))

        cmd[7] = ((int(cmd_list['res_ioff']) << 0 )|(int(cmd_list['res_psize']) << 32 ))

        for i in range(8):
            #print(cmd[i])
            array = (str(hex(cmd[i])))
            array = array[2:]
            leng = len(array)
            if leng<16:
                long = 16 - leng
                array = '0'*long+array
            self.file.write('*((int *)(cfg_addr + '+repr(i)+' * 8)) = ')
            self.file.write('0x'+array+';')
            self.file.write('\n')
        self.file.write('cfg_addr += 8*8；')
        return cmd



def run_test():
    path_x = 'resource/VGG16_x.csv'
    path_w = 'resource/VGG16_w.csv'
    w = set_cmd_cal(path_x,path_w,2000)
    print(w.cmd_array)
    print(w.array)
    use_memory_size = w.get_memory_use()
    with open("cmd.txt", 'w') as fo:
        cmd_number_count=0
        for i in range(len(w.cmd_array)):
            tile = set_cmd_tile(w.cmd_array[i],fo)
            cmd_number_count = tile.tile_number+cmd_number_count
            print(tile.tile_cmd_array)
        all_memory = use_memory_size + cmd_number_count*64
        print('总共使用的内存单位B',all_memory,'B')
        print('总共使用的内存单位M',all_memory / 1024 ** 2, 'M')

if __name__ =="__main__":
    run_test()


