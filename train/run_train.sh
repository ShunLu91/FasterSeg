#if [ ! -d logdir  ];then
#  mkdir logdir
#fi

#******************************* search *******************************#
#CUDA_VISIBLE_DEVICES=6 python train_search.py
# pretrain supernet
#CUDA_VISIBLE_DEVICES=0 nohup python -u train_search.py  > ../logdir/fs0.log  2>&1 &
# search arch
#CUDA_VISIBLE_DEVICES=0 nohup python -u train_search.py  > ../logdir/fs0_s.log  2>&1 &

#******************************* full train *******************************#
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py  > ../logdir/fs0_f.log  2>&1 &


tail -f ../logdir/fs0_f.log

