source activate zykycy
python ../main.py \
    --data_path /mnt/data3/lus/zhangyk/data/ye \
    --gpu 4 \
    --model_class ProtoNet \
    --distance l2 \
    --backbone_class ConvNet \
    --dataset MiniImageNet \
    --test_way 5 \
    --test_shot 1 \
    --test_query 15 \
    --test_model_filepath '/mnt/data3/lus/zhangyk/models/ProtoNet/0222 14-47-53-193 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth' \
    --epoch_verbose \
    --verbose \