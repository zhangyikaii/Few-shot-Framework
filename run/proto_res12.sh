# nohup \
python ../main.py \
    --max_epoch 200 \
    --gpu 0 \
    --model_class ProtoNet \
    --distance l2 \
    --backbone_class Res12 \
    --dataset MiniImageNet \
    --way 5 --eval_way 5 \
    --shot 1 --eval_shot 1 \
    --query 15 --eval_query 15 \
    --logger_filepath /logs/process \
    --balance 1 \
    --temperature 64 \
    --temperature2 16 \
    --lr 0.0001 --lr_mul 10 --lr_scheduler step \
    --step_size 20 \
    --gamma 0.5 \
    --eval_interval 1 \
    --test_interval 10 \
    --loss_fn F-cross_entropy \
    --verbose
    # --init_weights ./saves/initialization/miniimagenet/con-pre.pth
    #     > ./result_log/$0.txt 2>&1 &
