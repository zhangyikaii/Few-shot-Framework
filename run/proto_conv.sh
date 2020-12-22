# nohup \
python ../main.py \
    --max_epoch 150 \
    --gpu 0 \
    --model_class ProtoNet \
    --distance l2 \
    --backbone_class ConvNet \
    --dataset MiniImageNet \
    --way 30 --val_way 16 --test_way 5 \
    --shot 1 --test_shot 1 \
    --query 15 --test_query 15 \
    --logger_filepath /logs/process \
    --balance 1 \
    --temperature 64 \
    --temperature2 16 \
    --lr 0.0001 --lr_mul 10 --lr_scheduler step \
    --step_size 20 \
    --gamma 0.5 \
    --val_interval 1 \
    --test_interval 0 \
    --loss_fn F-cross_entropy \
    --verbose \
    --epoch_verbose
    # --init_weights ./saves/initialization/miniimagenet/con-pre.pth
    #     > ./result_log/$0.txt 2>&1 &
