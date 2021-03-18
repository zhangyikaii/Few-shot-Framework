source activate zykycy
# nohup \
python ../main.py \
    --data_path /home/yehj/Few-Shot/data \
    --model_save_path /data/zhangyk/models \
    --max_epoch 200 \
    --gpu 0 \
    --model_class RelationNet \
    --distance l2 \
    --backbone_class Conv4NP \
    --dataset MiniImageNet \
    --train_way 5 --val_way 5 --test_way 5 \
    --train_shot 1 --val_shot 1 --test_shot 1 \
    --train_query 15 --val_query 15 --test_query 15 \
    --logger_filename /logs \
    --balance 1 \
    --temperature 64 \
    --temperature2 16 \
    --lr 0.001 --lr_mul 10 --lr_scheduler step \
    --step_size 20 \
    --gamma 0.5 \
    --val_interval 1 \
    --test_interval 0 \
    --loss_fn nn-mse_loss \
    --metric_func categorical_accuracy_onehot \
    --verbose \
    # --epoch_verbose \