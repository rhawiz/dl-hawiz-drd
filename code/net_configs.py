from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, get_all_params, \
    FeaturePoolLayer, Pool2DLayer
from lasagne import init
from lasagne.nonlinearities import softmax, leaky_rectify
from lasagne.updates import adam, nesterov_momentum
from nolearn.lasagne import NeuralNet, objective, TrainSplit
import numpy as np
import theano
from nolearn.lasagne import BatchIterator


def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses


def get_conv_args(num_filters=32, filter_size=(3, 3), border_mode='same', nonlinearity=leaky_rectify,
                  W=init.Orthogonal(gain=1.0), b=init.Constant(0.05), untie_biases=True, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': filter_size,
        # 'border_mode': border_mode,
        'nonlinearity': nonlinearity,
        'W': W,
        'b': b,
        'untie_biases': untie_biases,
    }
    args.update(kwargs)
    return args


def get_pool_args(pool_size=3, stride=(2, 2), **kwargs):
    args = {
        'pool_size': pool_size,
        'stride': stride,
    }
    args.update(kwargs)
    return args


def dense_params(num_units, nonlinearity=leaky_rectify, **kwargs):
    args = {
        'num_units': num_units,
        'nonlinearity': nonlinearity,
        'W': init.Orthogonal(1.0),
        'b': init.Constant(0.05),
    }
    args.update(kwargs)
    return args


def cp(num_filters, filter_size=(4, 4), *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': filter_size,
    }
    args.update(kwargs)
    print args
    return get_conv_args(**args)


n = 32

# Different Batch iterator sizes
bi_16 = BatchIterator(batch_size=16)

bi_32 = BatchIterator(batch_size=32)

bi_64 = BatchIterator(batch_size=64)


# Stide - padding between features extracted
def createNet0(epoch=200):
    NET_256_4x4_32_16 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 256, 256)}),
            (Conv2DLayer, cp(n, stride=(2, 2))),
            (Conv2DLayer, cp(n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(2 * n, stride=(2, 2))),
            (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(4 * n)),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(8 * n)),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Pool2DLayer, get_pool_args()),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_16,
        batch_iterator_train=bi_16
    )

    return NET_256_4x4_32_16


def createNet1(epoch=100):
    NET_512_5x5_32_16 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 512, 512)}),
            (Conv2DLayer, get_conv_args(n, filter_size=(5, 5), stride=(2, 2))),
            (Conv2DLayer, get_conv_args(n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, get_conv_args(2 * n, filter_size=(5, 5), stride=(2, 2))),
            (Conv2DLayer, get_conv_args(2 * n)),
            (Conv2DLayer, get_conv_args(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, get_conv_args(4 * n)),
            (Conv2DLayer, get_conv_args(4 * n)),
            (Conv2DLayer, get_conv_args(4 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, get_conv_args(8 * n)),
            (Conv2DLayer, get_conv_args(8 * n)),
            (Conv2DLayer, get_conv_args(8 * n)),
            (Pool2DLayer, get_pool_args(stride=(3, 3))),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_16,
        batch_iterator_train=bi_16
    )
    return NET_512_5x5_32_16


def createNet2(epoch=200):
    NET_256_4x4_32_32 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 256, 256)}),
            (Conv2DLayer, cp(n, stride=(2, 2))),
            (Conv2DLayer, cp(n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(2 * n, stride=(2, 2))),
            (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(4 * n)),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(8 * n)),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Pool2DLayer, get_pool_args()),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_32,
        batch_iterator_train=bi_32
    )
    return NET_256_4x4_32_32


def createNet3(epoch=100):
    NET_512_4x4_32_16 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 512, 512)}),
            (Conv2DLayer, get_conv_args(n, filter_size=(4, 4), stride=(2, 2))),
            (Conv2DLayer, get_conv_args(n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, get_conv_args(2 * n, filter_size=(4, 4), stride=(2, 2))),
            (Conv2DLayer, get_conv_args(2 * n)),
            (Conv2DLayer, get_conv_args(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, get_conv_args(4 * n)),
            (Conv2DLayer, get_conv_args(4 * n)),
            (Conv2DLayer, get_conv_args(4 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, get_conv_args(8 * n)),
            (Conv2DLayer, get_conv_args(8 * n)),
            (Conv2DLayer, get_conv_args(8 * n)),
            (Pool2DLayer, get_pool_args(stride=(3, 3))),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_16,
        batch_iterator_train=bi_16
    )

    return NET_512_4x4_32_16


def createNet4(epoch=200):
    NET_256_2x2_32_32 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 256, 256)}),
            (Conv2DLayer, cp(n, filter_size=(2, 2), stride=(2, 2))),
            (Conv2DLayer, cp(n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(2 * n, filter_size=(2, 2), stride=(2, 2))),
            (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(4 * n)),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(8 * n)),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Pool2DLayer, get_pool_args()),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_32,
        batch_iterator_train=bi_32
    )
    return NET_256_2x2_32_32


def createNet5(epoch=200):
    NET_256_5x5_32_32 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 256, 256)}),
            (Conv2DLayer, cp(n, filter_size=(5, 5), stride=(2, 2))),
            (Conv2DLayer, cp(n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(2 * n, filter_size=(5, 5), stride=(2, 2))),
            (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(4 * n)),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(8 * n)),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Pool2DLayer, get_pool_args()),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_32,
        batch_iterator_train=bi_32
    )
    return NET_256_5x5_32_32


def createNet6(epoch=200):
    NET_256_3x3_32_32 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 256, 256)}),
            (Conv2DLayer, cp(n, filter_size=(3, 3), stride=(2, 2))),
            (Conv2DLayer, cp(n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(2 * n, filter_size=(3, 3), stride=(2, 2))),
            (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(4 * n)),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(8 * n)),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Pool2DLayer, get_pool_args()),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_32,
        batch_iterator_train=bi_32
    )

    return NET_256_3x3_32_32


def createNet7(epoch=200):
    NET_256_3x3_32_16 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 256, 256)}),
            (Conv2DLayer, cp(n, filter_size=(3, 3), stride=(2, 2))),
            (Conv2DLayer, cp(n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(2 * n, filter_size=(3, 3), stride=(2, 2))),
            (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(4 * n)),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(8 * n)),
            (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
            (Pool2DLayer, get_pool_args()),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=epoch,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_16,
        batch_iterator_train=bi_16
    )

    return NET_256_3x3_32_16


def createNet8(epoch=200):
    NET_256_4x4_32_32_v2 = NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 3, 256, 256)}),
            (Conv2DLayer, cp(n, filter_size=(4, 4), stride=(2, 2))),
            (Conv2DLayer, cp(n, border_mode=None, pad=2)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(2 * n, filter_size=(4, 4), stride=(2, 2))),
            (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(2 * n)),
            (MaxPool2DLayer, get_pool_args()),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Conv2DLayer, cp(4 * n)),
            (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
            (Pool2DLayer, get_pool_args()),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DropoutLayer, {'p': 0.5}),
            (DenseLayer, dense_params(1024)),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 1}),
        ]
        ,
        max_epochs=200,
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.cast['float32'](0.003)),
        update_momentum=0.9,
        objective=regularization_objective,
        objective_lambda2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        batch_iterator_test=bi_32,
        batch_iterator_train=bi_32
    )

    return NET_256_4x4_32_32_v2


# DO NOT CHANGE THE ORDERING OF THESE NETWORKS
NETWORKS = {
    0: createNet0(),
    1: createNet1(),
    2: createNet2(),
    3: createNet3(),
    4: createNet4(),
    5: createNet5(),
    6: createNet6(),
    7: createNet7(),
    8: createNet8(),
}
