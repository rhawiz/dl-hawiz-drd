from time import sleep

from pylearn2.utils import serial
import net_configs
from data import load_file
import theano
import pylearn2
import os
def test():
    #The below line will tell you where to put .theanoarc for configuration
    #print os.path.expanduser('~/.theanorc.txt')
    params = serial.load('../networks/1457038649_26229_9000_NET0_PARAMS')
    net = net_configs.createNet0(epoch=0)
    net.load_params_from(params)
    img = load_file('../data/diabetic_ret/dataset_256_norm/test/30307_left.jpeg')
    prediction = net.predict(img)
    print prediction


if __name__ == '__main__':
    test()
