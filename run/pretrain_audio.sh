source activate zykycy
python ../main.py \
    --do_train \
    --do_test \
    --meta_batch_size 1 \
    --data_path /mnt/data3/lus/zhangyk/data \
    --max_epoch 1 \
    --gpu 0 \
    --model_class RTPretrainClassifier \
    --distance l2 \
    --backbone_class Res1d2TCN \
    --dataset LRW \
    --val_way 16 --test_way 5 \
    --val_shot 1 --test_shot 1 \
    --val_query 15 --test_query 15 \
    --logger_filename /logs \
    --temperature 64 \
    --lr 0.0001 \
    --lr_scheduler cosine \
    --val_interval 1 \
    --test_interval 0 \
    --loss_fn nn-cross_entropy \
    --multimodal_option audio \
    --batch_size 32 \
    --grad_scaler \
    --epoch_verbose \
    --verbose

