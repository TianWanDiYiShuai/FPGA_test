#ifndef  _FPGA_CLASS_HPP_
#define	 _FPGA_CLASS_HPP_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "base_type.hpp"
#include "net2cmd.hpp"
#include "weights_view.hpp"


#define ifrm_xmax	127
#define	ifrm_ymax	63

#define conv_ymax	32
#define conv_xmax	64

using namespace std;

class Get_original_cmd
{
public:
	Get_original_cmd(string prottxt, string caffemodel, const int _start_memory);
	~Get_original_cmd();
	inline string Trim(string& str);
	vector<temp> Read_csv(const char * str);
	vector<cal_ctrl> cmd_array_temp;
	void set_org_cmd();
	int conv_layer_number;
	int start_memory;
	int feature_map_max = 0;

private:
	cal_ctrl org_array;
	vector<temp> px;
	vector<temp> pw;
	int __w_b_address;
	void set_conv_param(const int index_w, const int index_x);
	inline void set_conv_end(const int i);
	inline int get_feature_map_max();
	inline void set_featuer_map_address(const int i);
	inline void set_w_b_address(const int i);
};


class Set_cmd_tile
{
public:
	Set_cmd_tile(cal_ctrl array, ofstream &fp_cmd);
	~Set_cmd_tile();
	cal_ctrl org_array;
	layer_ctrl temp_array;
	void get_layer_tile(ofstream &fp_cmd);
	int x_number, y_number, tile_number;

private:
	void set_black_param(const int index);
	void set_blue_param(const int index, const int x_index, const int y_index);
	void set_dw_param(const int index);
	void set_tile_address(const int index, const int x_index, const int y_index);
	void ctrl_pack(layer_ctrl *ctrl, u64 *cmd);
	void ctrl_dump(u64 *cmd, ofstream &fp_cmd);
};


#endif

