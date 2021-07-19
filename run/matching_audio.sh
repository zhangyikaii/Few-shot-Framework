source activate zykycy
python ../main.py \
    --do_train \
    --do_test \
    --meta_batch_size 1 \
    --data_path /mnt/data3/lus/zhangyk/data \
    --max_epoch 20 \
    --gpu 0,1,2,3 \
    --model_class MatchingNet \
    --distance cosine \
    --backbone_class Res1d2TCN \
    --dataset LRW \
    --multimodal_option audio \
    --train_way 20 --val_way 20 --test_way 20 \
    --train_shot 1 --val_shot 1 --test_shot 1 \
    --train_query 15 --val_query 15 --test_query 15 \
    --logger_filename /logs \
    --temperature 64 \
    --lr 0.00001 --lr_mul 10 --lr_scheduler step \
    --step_size 10 \
    --gamma 0.5 \
    --val_interval 1 \
    --test_interval 0 \
    --loss_fn nn-cross_entropy \
    --init_weights "/home/lus/zhangyk/pre_trained_weights" \
    --epoch_verbose \
    --verbose \

