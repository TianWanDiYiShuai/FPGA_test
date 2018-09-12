# # -*- coding: UTF-8 -*-
# # date:2018/6/14
# # User:WangHong
import caffe
import numpy as np
import sys
import time
np.set_printoptions(threshold='nan')

caffe.set_mode_cpu()

model_def = 'deploy_VGG16.prototxt'
model_weights = 'VGG_ILSVRC_16_layers.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

params_txt = 'w_test1.txt'
pf = open(params_txt, 'wb')


def map_func(x):
    result = int(x*128)
    return result



def get_conv_params(pf):
    w = 0
    for k, v in net.params.items():
        w = w +1
        w_shape = v[0].data.shape
        if len(w_shape) != 4:
            break
        else:
            if w_shape[1] < 8:
                zero_fill = True
                cyc_one = 1
            else:
                cyc_one = w_shape[1] / 8
                zero_fill = False
            cyc_two = w_shape[0] / 16
            cyc_b = w_shape[0]/8
            cyc_three = w_shape[2]
            cyc_four = w_shape[3]
            for k1 in range(0, cyc_one):
                for k2 in range(0, cyc_two):
                    for k3 in range(0, cyc_three):
                        for k4 in range(0, cyc_four):
                            for k5 in range(0, 16):
                                kay_value = 0
                                if zero_fill == False:
                                    for k6 in range(0, 8):  
                                        one_index = k2 * 16 + k5
                                        two_index = k1 * 8 + k6
                                        three_index = k3
                                        four_index = k4
                                        value = map_func(v[0].data[one_index][two_index][three_index][four_index])
                                        value = value % 256
                                        kay_value = kay_value + value*(256**k6)
                                        
                                        #pf.write(str(value)+',')
                                        #print(one_index, two_index, three_index, four_index)
                                        #pf.write(str((one_index, two_index, three_index, four_index)))
                                        #pf.write('\n')
                                else:
                                    for k6 in range(0, 3):
                                        one_index = k2 * 16 + k5
                                        two_index = k1 * 8 + k6
                                        three_index = k3
                                        four_index = k4
                                        value = map_func(v[0].data[one_index][two_index][three_index][four_index])
                                        value = value % 256
                                        kay_value = kay_value + value*(256**k6)
                                       # pf.write(str(value)+',')
#                                    for k7 in range(0,5):
#                                        pf.write("0"+",")
                                array = str(hex(kay_value))
                                array = array[2:]
                                leng = len(array)
                                long = 16-leng
                                array = '0x'+'0'*long +array
                                if len(array)>18:
                                    array = array[:-1]
                                pf.write('*((int *)(cfg_addr + '+repr(k5)+' * 8)) = ')
                                pf.write(array+';')
                                pf.write('\n')
                            pf.write('cfg_addr += 16*8；')
                            pf.write('\n')                  
            for kb in range(cyc_b):
                kay_value=0
                for kb1 in range(8):
                    b_index = kb*8 + kb1
                    value_b = map_func(v[1].data[b_index])
                    value_b = value_b % 256
                    kay_value = kay_value + value_b*(256**kb1)
                    print("kb1= ",kb1)
                    print(v[1].data[b_index])
                    print(kay_value)
                    
                time.sleep(100)
                array = str(hex(kay_value))
                array = array[2:]
                leng = len(array)
                long = 16-leng
                array = '0x'+'0'*long +array
                if len(array)>18:
                    array = array[:-1]
                pf.write('*((int *)(cfg_addr + '+repr(kb)+' * 8)) = ')
                pf.write(array+';')
                pf.write('\n')
            pf.write('cfg_addr +='+ repr(cyc_b)+'*8；')
            pf.write("\n")               
    pf.close()

#                               # pf.write(v[0].data[one_index][two_index][three_index][four_index],'\n')

get_conv_params(pf)

# layer_names = net.params.keys()
# for name in layer_names:
#    print name
#    length = len(net.params[name])
#    pf.write('layer name: %s, layer dims: %d\n' % (name, length))
#    for i in range(length):
#        pf.write('------------------------------------------------------------------------------------\n')
#        param = net.params[name][i].data
#        print param.shape
#        pf.write('layer %s, param %d shape: ' % (name, i))
#        for j in range(len(param.shape)):
#            pf.write('%d ' % param.shape[j])
#        pf.write('\n')
#
#
#        param.shape = (-1, 1)
#        for p in param:
#            pf.write('%.3f, ' % p)
#        pf.write('\n')
#    pf.write('\n=======================================================================================\n')
#
# pf.close()

