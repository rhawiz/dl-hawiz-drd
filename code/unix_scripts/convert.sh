#!/usr/bin/env bash
THEANO_FLAGS=device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5 python convert_net.py