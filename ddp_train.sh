python -m torch.distributed.launch --nproc_per_node 4 --use_env train.py \
      --distributed \
      --output_path output/v2 \
      --data_path /mnt/cfs/CV/lmn/data/document_process/dewarp/total/color \
      --data_path_validate /mnt/cfs/CV/lmn/data/document_process/dewarp/total/validate/color \
      --batch_size 128 \
      --l_rate 0.001 \
      --parallel 0,1,2,3 \
      --print-freq 60 \

