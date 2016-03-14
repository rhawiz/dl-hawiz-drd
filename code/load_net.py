import cPickle
import convert_net

def load_net(fname):
    try:
        f = open(fname, 'rb')
    except Exception, e:
        print e

    net = cPickle.load(f)

    return net

