python train.py \
      --output_path output/dp \
      --data_path /mnt/cfs/CV/lmn/data/document_process/dewarp/merged/train_lmn.lst \
      --batch_size 16 \
      --l_rate 0.001 \
      --parallel 0,1,2,3 \
      --print-freq 60 \
      --max_num 100000 \
