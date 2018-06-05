import torchvision.models as models
import torch
import torch.nn as nn
from torch.autograd import Variable
torch.set_printoptions(threshold=80001) 
import torch.onnx
import caffe 
import numpy as np
np.set_printoptions(threshold=np.inf)

m = nn.Conv2d(16, 33, 3, stride=2)
model_input = Variable(torch.ones([8, 16, 25, 25])*0.01)
torch_output = m(model_input)
#print(model_input)
#print(torch_output)
#exit()
input_names = [ "actual_input_1"]
output_names = [ "output1" ]

#torch.onnx.export(m, model_input, "conv2d.onnx", verbose=True)
#exit()
#load caffe model 
net = caffe.Net('conv2d.prototxt', 1, weights='conv2d.caffemodel')
net.forward()
#layers = list(net._blob_names)
#for layer in layers:
#    print(layer)
#    print(net.blobs[layer].data)

print(type(torch_output.data.numpy()))
print(type(net.blobs['6'].data))
#print(model_input.data.numpy() - net.blobs['1'].data)
print(torch_output.data.numpy() - net.blobs['6'].data)
