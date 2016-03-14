
from pylearn2.utils import serial
from os import walk


def convert_net():
    for (dirpath, dirnames, filenames) in walk('../networks/'):
        for file in filenames:
            model = serial.load(dirpath+file)
            try:
                print "Saving {} parameters".format(file)
                pvals = model.get_all_params_values()
                serial.save("{}/{}_PARAMS".format(dirpath,file), pvals)
            except Exception, e:
                print e


if __name__ == '__main__':
    convert_net()

