import os
import click
import time
import datetime
import net_configs
import data
import cPickle
import numpy as np
from load_net import load_net
from net_configs import get_net

DEFAULT_TRAINING_DATA = os.path.relpath('../data/diabetic_ret/dataset_256/')
DEFAULT_LABELS = os.path.relpath('../data/diabetic_ret/trainLabels.csv')


def now():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def train(limit=-1, net_id=0, train_dir=DEFAULT_TRAINING_DATA, labels_dir=DEFAULT_LABELS, size=256):
    if limit == -1:
        limit_str = "All"
    else:
        limit_str = limit
    print "{}\tLoading {} images in {}".format(now(), limit, train_dir)
    files, labels = data.load_data(train_dir, labels_dir, verbose=0, limit=limit, size=size)

    net = get_net(net_id)
    if not net:
        print '{}\tNo network with id {}'.format(now(), net_id)
        return

    start_timestamp = int(time.time())
    print "{}\tInitialising network...".format(now())
    net.fit(files, labels)
    end_timestamp = int(time.time())
    duration = end_timestamp - start_timestamp
    print "{}\tTraining complete, total duration {} seconds".format(now(), duration)
    file_name = '{}_{}_{}_NET{}'.format(end_timestamp, duration, limit, net_id)

    f = open(file_name, 'w+')

    print "{}\tSaving file to {}".format(now(), file_name)
    cPickle.dump(net, f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()


@click.command()
@click.option('--limit', '-l', default=-1, show_default=True, help="Set limit to how many images to load")
@click.option('--train_dir', '-td', default=DEFAULT_TRAINING_DATA, show_default=True,
              help="Path to directory containing training data")
@click.option('--labels_dir', '-ld', default=DEFAULT_LABELS, show_default=True,
              help="Path to directory containing training labels")
@click.option('--net_id', '-n', default=0, show_default=True,
              help="Network ID defined in net_configs.py in NETWORK variable")
@click.option('--size', '-s', default=0, show_default=True,
              help="Size of image. Must align to network.")
def main(limit=-1, train_dir=DEFAULT_TRAINING_DATA, labels_dir=DEFAULT_LABELS, net_id=0, size=256):
    train(limit=limit, train_dir=train_dir, labels_dir=labels_dir, net_id=net_id, size=size)


if __name__ == '__main__':
    main()
