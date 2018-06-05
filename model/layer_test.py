import caffe
from collections import OrderedDict
import numpy as np
np.set_printoptions(threshold=np.inf)

# load caffe model

# Create the net is TEST mode
caffe.set_mode_gpu()
#caffe.set_device(0)

net = caffe.Net('layertest.prototxt',      # the structure of the model
                'MobileNetV2.caffemodel', # parameters
                caffe.TEST)
print("Successfully loaded caffe model")

net.forward()

layers = list(net._blob_names)
for layer in layers:
    print(layer)
    print(net.blobs['443'].data)
    break

# pytorch model

