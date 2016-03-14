THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5 python train.py --limit 9000 --train_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/dataset_256_norm/train/" --labels_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/trainLabels.csv" --net_id 0 --size 256

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5 python train.py --limit 18000 --train_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/dataset_256_norm/train/" --labels_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/trainLabels.csv" --net_id 0 --size 256

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5 python train.py --limit 18000 --train_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/dataset_256_norm/train/" --labels_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/trainLabels.csv" --net_id 2 --size 256

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5 python train.py --limit 3000 --train_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/dataset_512_norm/train/" --labels_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/trainLabels.csv" --net_id 1 --size 512

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5 python train.py --limit 3000 --train_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/dataset_512_norm/train/" --labels_dir "/home/andrea/Desktop/rawand/dl-hawiz-drd/data/diabetic_ret/trainLabels.csv" --net_id 3 --size 512

