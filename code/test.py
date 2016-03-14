from pylearn2.utils import serial
import net_configs

def test():
    params = serial.load('../networks/1457038649_26229_9000_NET0_PARAMS')
    net = net_configs.createNet0(epoch=0)
    net.load_params_from(params)
    print net
    #img, val = load_file('../data/dataset_256_norm/test/30307_left.jpeg', '../data/trainLabels.csv')
    #value = clf.predict(img)
    #print value


if __name__ == '__main__':
    test()
