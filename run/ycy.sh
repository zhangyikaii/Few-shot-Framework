source activate zykycy
python ../models/ycy.py \
    --gpu 3 \
    --data_path /mnt/data3/lus/zhangyk/data \
    --dataset MiniImageNet \
    --test_way 5 \
    --test_shot 5 \
    --test_query 15 \
    --distance l2 \
    --backbone_class ConvNet \
    --logger_filename /logs \
    --simpleshot_episodes_per_epoch 2000 \
    --simpleshot_norm_type_list "ori__ori__prt" \
    --meta_batch_size 1 \
    --init_weights /home/lus/zhangyk/pre_trained_weights \
    --simpleshot_cache_path /home/lus/zhangyk/pre_trained_weights \
    --batch_size 128 \
    --verbose \
    --tr_pca_n_components 37 \
    --epoch_verbose

# ori__ycy_DC__svm_skln, c__ycy_DC__svm_skln, ori__ori__prt
# c__tr_pca_pure__prt, c__ori__tr_s_c_pca_pure, ori__ori__prt
# ori__ori__prt, l1n__ori__prt, l2n__ori__prt, c__ori__prt, cl1n__ori__prt, cl2n__ori__prt, z__ori__prt, zl1n__ori__prt, zl2n__ori__prt, c__tr_pca_pure__prt, cl1n__tr_pca_pure__prt, cl2n__tr_pca_pure__prt, c__tr_pca_pure__lr_skln, cl1n__tr_pca_pure__lr_skln, cl2n__tr_pca_pure__lr_skln

# PCA 不同特征值率:
# c__tr_pca_pure__prt, cl1n__tr_pca_pure__prt, cl2n__tr_pca_pure__prt

# ori,l2n,cl2n,z_score,centering,
# ycy_DC, ycy_DC_cl2n, ycy_MG
# sklearn_PCA, sklearn_LDA, sklearn_NCA
# meta_test_zca_whitening,meta_test_pca_whitening,meta_test_pca_pure,meta_test_zca_corr_whitening,meta_test_pca_corr_whitening,
# meta_train_zca_whitening,meta_train_pca_whitening,meta_train_pca_pure,meta_train_zca_corr_whitening,meta_train_pca_corr_whitening

# protonet_aug, protonet, lr_sklearn, k_nn

# un__protonet, centering__protonet, l2n__protonet, cl2n__protonet, z_score__protonet
# un__protonet, meta_train_pca_pure__protonet, meta_train_pca_pure__k_nn, meta_train_pca_pure__lr_sklearn

# un__protonet, ycy_DC__protonet_aug, ycy_DC_cl2n__protonet_aug, ycy_MG__protonet_aug
# un__protonet, ycy_DC__lr_sklearn, ycy_DC_cl2n__lr_sklearn, ycy_MG__lr_sklearn
# un__protonet, sklearn_PCA__protonet, sklearn_PCA__k_nn, meta_train_pca_pure__protonet, meta_train_pca_pure__k_nn
# un__protonet, sklearn_LDA__protonet
# un__protonet, meta_train_zca_whitening__protonet, meta_train_pca_whitening__protonet, meta_train_pca_pure__protonet, meta_train_zca_corr_whitening__protonet, meta_train_pca_corr_whitening__protonet