#include "stdafx.h"
#include "FPGA_class.hpp"
#include "weights_view.hpp"


Get_original_cmd::Get_original_cmd(string prottxt,string caffemodel,const int _start_memory)
{
	get_params_shape(prottxt, caffemodel, pw, px);
	start_memory = _start_memory;
	feature_map_max = get_feature_map_max();
	__w_b_address = feature_map_max * 2 + start_memory;
	set_org_cmd();
}


Get_original_cmd::~Get_original_cmd()
{
}


inline string Get_original_cmd::Trim(string& str)
{
	str.erase(0, str.find_first_not_of(" \t\r\n"));
	str.erase(str.find_last_not_of(" \t\r\n") + 1);
	return str;
};

/*********************************************************************
程序提供两个接口，当caffe环境配置好的情况下，可以不使用，用python提取的CSV
数据文件，当caffe环境没有配置好的情况下，可以使用python提取的CSV文件，
使用Read_CSV函数来读取函数中的内容。
*********************************************************************/
vector<temp> Get_original_cmd::Read_csv(const char* str)
{
	vector<temp> temp_csv;
	ifstream fin(str); //打开文件流操作
	if (!fin)
	{
		cout << "dfadadas" << endl;
		cout << "打开文件失败" << endl;
		exit(0);
	}
	string line;
	int index = 0;
	while (getline(fin, line))   //整行读取，换行符“\n”区分，遇到文件尾标志eof终止读取
	{
		istringstream sin(line); //将整行字符串line读入到字符串流istringstream中
		vector<string> fields; //声明一个字符串向量
		temp temp_test;
		string field;
		while (getline(sin, field, ',')) //将字符串流sin中的字符读入到field字符串中，以逗号为分隔符
		{
			fields.push_back(field); //将刚刚读取的字符串添加到向量fields中
		}
		if (index != 0)
		{
			temp_test.conv = Trim(fields[0]);
			temp_test.num = stoi(Trim(fields[1]));
			temp_test.channce = stoi(Trim(fields[2]));
			temp_test.x_size = stoi(Trim(fields[3]));
			temp_test.y_size = stoi(Trim(fields[4]));
			temp_csv.push_back(temp_test);
		}
		index++;
	}
	return temp_csv;
}

/*****************************************************************
用于生成cmd命令的逻辑层函数，用于判断，没成属于什么类型
****************************************************************/
void Get_original_cmd::set_org_cmd()
{
	int index_x = 0;
	int __w_b_address = feature_map_max * 2 + start_memory;
	for (int i = 0; i<pw.size(); i++)
	{
		size_t point = px[index_x + 1].conv.find("pool", 0);
		set_conv_param(i, index_x);
		if (point != string::npos)
		{
			index_x = index_x + 2;
			org_array.pool_en = true;
		}
		else if (point == string::npos)
		{
			index_x = index_x + 1;
			org_array.pool_en = false;
		}
		cmd_array_temp.push_back(org_array);
	}
}


void Get_original_cmd::set_conv_param(const int index_w, const int index_x)
{
	org_array.ifrm_width = px[index_x].x_size;
	org_array.ifrm_height = px[index_x].y_size;
	org_array.conv_size = 3;
	org_array.conv_pad = true;
	org_array.conv_std = 1;

	if (px[index_x].channce < 8)org_array.ifrm_num = 8;
	else org_array.ifrm_num = px[index_x].channce;

	org_array.relu_en = true;
	org_array.res_en = false;
	org_array.res_bsptr = 0;

	/*############################################################################

	这里设置卷积参数，w和b的地址，wei完成
	##############################################################################
	*/
	set_w_b_address(index_w);
	org_array.dw_en = false;
	org_array.dw_pad = false;
	org_array.dw_std = 0;
	org_array.dwp_bsptr = 0;
	org_array.dw_relu_en = false;

	org_array.ofrm_width = px[index_x + 1].x_size;
	org_array.ofrm_height = px[index_x + 1].y_size;
	org_array.ofrm_num = px[index_x + 1].channce;

	/*###########################################################################

	这里设置feature map的地址，完成
	#############################################################################
	*/
	set_featuer_map_address(index_w);
	/*###########################################################################

	这里设置卷积结束标志位。完成
	#############################################################################
	*/
	set_conv_end(index_w);
}

/***************设置结束标志位函数******************************/
inline void Get_original_cmd::set_conv_end(const int i)
{
	if (i == pw.size() - 1)org_array.conv_end = false;
	else org_array.conv_end = true;
}

/**********************获取整个网络中所有feature map 需要占用的最大内存的值***********/
inline int Get_original_cmd::get_feature_map_max()
{
	int  map_max = 0;
	for (int i = 0; i < pw.size(); i++)
	{
		int temp = px[i].channce *px[i].x_size*px[i].y_size;
		if (temp >= map_max)
		{
			map_max = temp;
		}
	}
	return map_max / 8;
}

/***用于设置每层feature map的内存地址的函数*************************/
inline void Get_original_cmd::set_featuer_map_address(const int i)
{
	if (i % 2 == 0)
	{
		org_array.ifrm_bsptr = start_memory;
		org_array.ofrm_bsptr = start_memory + feature_map_max;
	}
	else
	{
		org_array.ifrm_bsptr = start_memory + feature_map_max;
		org_array.ofrm_bsptr = start_memory;
	}
}

/*****用于设置每层的参数w和b 的内存地址的函数***********************/
inline void Get_original_cmd::set_w_b_address(const int i)
{
	org_array.convp_bsptr = __w_b_address;
	int b_address = __w_b_address + pw[i].channce*pw[i].num*pw[i].x_size*pw[i].y_size / 8;
	org_array.convk_bsptr = b_address;
	__w_b_address = b_address + pw[i].num / 8;
}





/*#############################################################################################
设置tile的cmd类。
###############################################################################################
*/

Set_cmd_tile::Set_cmd_tile(cal_ctrl array, ofstream &fp_cmd)
{
	org_array = array;
	get_layer_tile(fp_cmd);

}

Set_cmd_tile::~Set_cmd_tile()
{

}

void Set_cmd_tile::get_layer_tile(ofstream &fp_cmd)
{

	u64 cmd[8];
	if (org_array.conv_std == 1)
	{
		x_number = org_array.ifrm_width / 64 + 1;
		y_number = org_array.ifrm_height / 32 + 1;
	}
	else if (org_array.conv_std == 2)
	{
		x_number = org_array.ifrm_width / 128 + 1;
		y_number = org_array.ifrm_height / 64 + 1;
	}
	tile_number = x_number * y_number;
	for (int y_index = 0; y_index < y_number; y_index++)
	{
		for (int x_index = 0; x_index < x_number; x_index++)
		{

			int index = y_index * x_number + x_index;
			cout << "第"<<index<<"tile" << endl;
			//设置黑色的参数
			set_black_param(index);
			//设置蓝色的参数
			set_blue_param(index, x_index, y_index);
			//**************************************设置边界零填充的参数*********************************
			if (x_index == 0) temp_array.conv_lp = true;
			else temp_array.conv_lp = false;
			if (x_index == x_number - 1)temp_array.conv_rp = true;
			else temp_array.conv_rp = false;
			if (y_index == 0)temp_array.conv_tp = true;
			else temp_array.conv_tp = false;
			if (y_index == y_number - 1)temp_array.conv_bp = true;
			else temp_array.conv_bp = false;
			ctrl_pack(&temp_array, cmd);
			ctrl_dump(cmd, fp_cmd);
			fp_cmd << endl;
		}
	}
}

void Set_cmd_tile::set_black_param(const int index)
{
	temp_array.ifrm_xoff = org_array.ifrm_width;
	temp_array.conv_size = org_array.conv_size;
	temp_array.conv_std = org_array.conv_std;
	temp_array.ifrm_num = org_array.ifrm_num;
	temp_array.ifrm_bsptr = org_array.ifrm_bsptr;
	temp_array.ifrm_psize = org_array.ifrm_width * org_array.ifrm_height;
	temp_array.pool_en = org_array.pool_en;
	temp_array.relu_en = org_array.relu_en;
	temp_array.res_en = org_array.res_en;
	temp_array.convp_bsptr = org_array.convp_bsptr;
	temp_array.convk_bsptr = org_array.convk_bsptr;
	temp_array.res_bsptr = 0;
	temp_array.res_xoff = 0;
	temp_array.res_psize = 0;
	temp_array.dw_en = org_array.dw_en;
	temp_array.dw_std = org_array.dw_std;
	temp_array.dwp_bsptr = org_array.dwp_bsptr;
	temp_array.dw_relu_en = org_array.dw_relu_en;
	temp_array.ofrm_num = org_array.ofrm_num;
	temp_array.ofrm_bsptr = org_array.ofrm_bsptr;
	temp_array.ofrm_psize = org_array.ofrm_width*org_array.ofrm_height;
	if (index == 0)
	{
		temp_array.firstile_layer = true;
		temp_array.lastile_layer = false;
	}
	else if (index == tile_number - 1)
	{
		temp_array.firstile_layer = false;
		temp_array.lastile_layer = true;
	}
	else
	{
		temp_array.firstile_layer = false;
		temp_array.lastile_layer = false;
	}
	temp_array.conv_end = org_array.conv_end;
}

void Set_cmd_tile::set_blue_param(const int index, const int x_index, const int y_index)
{
	if (org_array.conv_std == 1)
	{
		if (x_index + 1 == x_number && y_index != y_number)
		{
			temp_array.ifrm_xlen = org_array.ifrm_width - (x_number - 1) * 63;
			temp_array.ifrm_ylen = 32;
		}
		else if (x_index + 1 != x_number && y_index + 1 == y_number)
		{
			temp_array.ifrm_xlen = 64;
			temp_array.ifrm_ylen = org_array.ifrm_height - (y_number - 1) * 31;
		}
		else if (x_index + 1 == x_number && y_index + 1 == y_number)
		{
			temp_array.ifrm_xlen = org_array.ifrm_width - (x_number - 1) * 63;
			temp_array.ifrm_ylen = org_array.ifrm_height - (y_number - 1) * 31;
		}
		else
		{
			temp_array.ifrm_xlen = 64;
			temp_array.ifrm_ylen = 32;
		}
	}
	else if (org_array.conv_std == 2)
	{
		if (x_index + 1 == x_number && y_index != y_number)
		{
			temp_array.ifrm_xlen = org_array.ifrm_width - (x_number - 1) * 127;
			temp_array.ifrm_ylen = 63;
		}
		else if (x_index + 1 != x_number && y_index + 1 == y_number)
		{
			temp_array.ifrm_xlen = 127;
			temp_array.ifrm_ylen = org_array.ifrm_height - (y_number - 1) * 63;
		}
		else if (x_index + 1 == x_number && y_index + 1 == y_number)
		{
			temp_array.ifrm_xlen = org_array.ifrm_width - (x_number - 1) * 127;
			temp_array.ifrm_ylen = org_array.ifrm_height - (y_number - 1) * 63;
		}
		else
		{
			temp_array.ifrm_xlen = 127;
			temp_array.ifrm_ylen = 63;
		}
	}
	temp_array.res_ioff = 0;//没有累加层都置为0
	temp_array.ofrm_xoff = org_array.ofrm_width;
	//###############################设置dw参数#############################
	set_dw_param(index);
	//######################################################################


	//###############################设置tile地址#############################
	set_tile_address(index, x_index, y_index);
	//######################################################################
}

void Set_cmd_tile::set_dw_param(const int index)
{
	if (org_array.dw_en == 0)
	{
		temp_array.dw_en = 0;
		temp_array.dw_ifrm_xlen = 0;
		temp_array.dw_ifrm_ylen = 0;
		temp_array.dw_tp = 0;
		temp_array.dw_bp = 0;
		temp_array.dw_lp = 0;
		temp_array.dw_rp = 0;
		temp_array.dw_std = 0;
		temp_array.dwp_bsptr = 0;
		temp_array.dw_relu_en = 0;
	}
	else
	{
		//后面的深度卷积预留位
	}
}

//计算每个tile经过卷积后对应的输出tile的长，宽和下一个tile的偏移量
void Set_cmd_tile::set_tile_address(const int index, const int x_index, const int y_index)
{
	if (x_index == 0 && y_index == 0) temp_array.ifrm_ioff = 0;
	else temp_array.ifrm_ioff = (y_index*org_array.ifrm_width*(temp_array.ifrm_ylen - 2)) + x_index * (temp_array.ifrm_xlen - 1) - 1;

	if (temp_array.pool_en == false)
	{
		temp_array.ofrm_xlen = 64;
		temp_array.ofrm_ylen = 32;
		temp_array.ofrm_ioff = temp_array.ifrm_ioff;
	}
	else if (temp_array.pool_en == true)
	{
		temp_array.ofrm_xlen = 32;
		temp_array.ofrm_ylen = 16;
		if (y_index == 0 && x_index == 0)temp_array.ofrm_ioff = 0;
		else if (y_index == 0 && x_index != 0)temp_array.ofrm_ioff = x_index*temp_array.ofrm_xlen - 1;
		else if (y_index != 0 && x_index == 0)temp_array.ofrm_ioff = (y_index * temp_array.ofrm_ylen - 1)*org_array.ofrm_width - 1;
		else temp_array.ofrm_ioff = (y_index * temp_array.ofrm_ylen - 1)*org_array.ofrm_width + x_index*temp_array.ofrm_xlen - 1;
	}
}

void Set_cmd_tile::ctrl_pack(layer_ctrl *ctrl, u64 *cmd)
{
	cmd[0] = ((u64)(ctrl->ifrm_xlen) << 0) | ((u64)(ctrl->ifrm_ylen) << 7)
		| ((u64)(ctrl->ifrm_xoff) << 14) | ((u64)(ctrl->conv_size) << 25)
		| ((u64)(ctrl->conv_tp) << 27) | ((u64)(ctrl->conv_bp) << 28)
		| ((u64)(ctrl->conv_lp) << 29) | ((u64)(ctrl->conv_rp) << 30)
		| ((u64)(ctrl->conv_std) << 31) | ((u64)(ctrl->ifrm_num) << 32)
		| (((u64)(ctrl->ifrm_psize) & 0x1ff) << 45);

	cmd[1] = ((u64)(ctrl->ifrm_bsptr) << 0) | ((u64)(ctrl->ifrm_ioff) << 32)
		| ((((u64)(ctrl->ifrm_psize) >> 9) & 0x1fff) << 54)
		| ((u64)(ctrl->pool_en) << 57) | ((u64)(ctrl->relu_en) << 58)
		| ((u64)(ctrl->res_en) << 59);

	cmd[2] = ((u64)(ctrl->convp_bsptr) << 0) | ((u64)(ctrl->convk_bsptr) << 32);

	cmd[3] = ((u64)(ctrl->dw_en) << 0) | ((u64)(ctrl->dw_ifrm_xlen) << 1)
		| ((u64)(ctrl->dw_ifrm_ylen) << 8) | ((u64)(ctrl->dw_tp) << 15)
		| ((u64)(ctrl->dw_bp) << 16) | ((u64)(ctrl->dw_lp) << 17)
		| ((u64)(ctrl->dw_rp) << 18) | ((u64)(ctrl->dw_std) << 19)
		| ((u64)(ctrl->dwp_bsptr) << 20) | ((u64)(ctrl->dw_relu_en) << 52);

	cmd[4] = ((u64)(ctrl->ofrm_xlen) << 0) | ((u64)(ctrl->ofrm_ylen) << 7)
		| ((u64)(ctrl->ofrm_xoff) << 14) | ((u64)(ctrl->ofrm_num) << 25)
		| ((u64)(ctrl->ofrm_psize) << 38);

	cmd[5] = ((u64)(ctrl->ofrm_bsptr) << 0) | ((u64)(ctrl->ofrm_ioff) << 32)
		| ((u64)(ctrl->firstile_layer) << 54) | ((u64)(ctrl->lastile_layer) << 55)
		| ((u64)(ctrl->conv_end) << 63);

	cmd[6] = ((u64)(ctrl->res_bsptr) << 0) | ((u64)(ctrl->res_xoff) << 32);

	cmd[7] = ((u64)(ctrl->res_ioff) << 0) | ((u64)(ctrl->res_psize) << 22);
}

void Set_cmd_tile::ctrl_dump(u64 *cmd, ofstream &fp_cmd)
{
	if (fp_cmd.is_open())
	{
		cout << "正在写入文件" << endl;

		for (int i = 0; i < 8; i++)
		{
			fp_cmd << "*((int *)(cfg_addr + "<<i<<" * 8)) = " << dec2hex(cmd[i],16) << endl;
		}
		fp_cmd << "cfg_addr += 8*8;" << endl;
		cout << "文件写入完成" << endl;
	}
	else
	{
		cout << "文件写入失败" << endl;
	}
}


