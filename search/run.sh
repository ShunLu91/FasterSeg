if [ ! -d logdir  ];then
  mkdir logdir
fi

#CUDA_VISIBLE_DEVICES=6 python train_search.py
CUDA_VISIBLE_DEVICES=6 nohup python -u train_search.py  > logdir/fs0.log  2>&1 &

