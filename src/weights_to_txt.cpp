#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include <caffe/proto/caffe.pb.h>
#include <fstream>  
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <iosfwd>
#include <stdio.h>
#include <iostream>
#include <map>
#include "weights_to_txt.hpp"


using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::io::ArrayInputStream;
using google::protobuf::Message;

using namespace std;



void writeData(FILE* f, const float* data, int len){
	for (int i = 0; i < len; ++i)//, ++pos)
	{
		if (i == len - 1)
			fprintf(f, "%f\n", data[i]);
		else
			fprintf(f, "%f ", data[i]);


	}
}



bool loadCaffemodel(const char* file, Message* net){
	int fd = _open(file, O_RDONLY | O_BINARY);
	if (fd == -1)
	{
		printf("代开文件失败");
		return false;

	}
	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(1024 * 1024 * 1024, 64 * 1024 * 1024);
	cout << "这里是测试1" << endl;
	bool success = net->ParseFromCodedStream(coded_input);
	delete coded_input;
	delete raw_input;
	_close(fd);
	cout << "这里是测试2" << endl;
	return success;
}


void parse_caffemodel(string caffemodel, string outtxt)
{
	
	printf("%s\n", caffemodel.c_str());
	NetParameter net;
	bool success = loadCaffemodel(caffemodel.c_str(), &net);
	if (!success){
		cout << "读取错误啦:%s可能错误的原因为最大能读取64M的caffemodel文件\n" << caffemodel.c_str()<< endl;
		return;
	}
	cout << "这里是测试6" << endl;
	cout << "一共有：" << net.layer_size() << endl;
	FILE* fmodel = fopen(outtxt.c_str(), "wb");

	for (int i = 0; i < net.layer_size(); ++i){
		cout << "一共有："<<net.layer_size() << endl;
		LayerParameter& param = *net.mutable_layer(i);
		int n = param.mutable_blobs()->size();
		if (n){
			const BlobProto& blob = param.blobs(0);
			cout << "layer:  " << param.name().c_str() << "weight: " << blob.data_size() << endl;
			cout << "这里是测试3" << endl;
			fprintf(fmodel, "\nlayer: %s weight(%d)\n", param.name().c_str(), blob.data_size());
			writeData(fmodel, blob.data().data(), blob.data_size());

			if (n > 1){
				const BlobProto& bais = param.blobs(1);
				cout << " bais: " << bais.data_size() << endl;
				cout << "这里是测试4" << endl;
				fprintf(fmodel, "\nlayer: %s bais(%d)\n", param.name().c_str(), bais.data_size());
				writeData(fmodel, bais.data().data(), bais.data_size());
			}
			cout << endl;
		}
	}
	fclose(fmodel);
	cout << "这里是测试5" << endl;
}

