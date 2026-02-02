CUDA_VISIBLE_DEVICES=0 python train.py  --gpu_ids 0 \
    --name Finetune/PR --checkpoints_dir checkpoints --model cpt --CUT_mode CUT --n_epochs 20 --n_epochs_decay 0 \
    --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral \
    --lambda_GAN 1.0 --lambda_NCE 2.5 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --lambda_gp 1.0 \
    --gp_weights [0.015625,0.03125,0.0625,0.125,0.25,1.0] --lambda_asp 5.0 --asp_loss_mode lambda_linear \
    --dataset_mode aligned --direction AtoB \
    --num_threads 1 --batch_size 1 --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False \
    --display_winsize 512 --update_html_freq 100 --save_epoch_freq 1 \
    --display_id 0 \
    --gan_mode vanilla --continue_train --pretrained_name WarmUp/PR --epoch 90 --lr 4e-5 \
    --dino_sample 96 --dataset_name 'PR'\
    --use_scl --sample_per_cluster 1 --total_cluster_number 100 --lambda_scl 0.5 --temperature 0.07 --other_cluster_number 32 \
    --use_mask --bio_mask_dir BioMask/MIST/PR/ \
    --use_dino --dino_mask_dir dino-main/DINO-Attention/epoch800/MIST/PR/trainA/attn-head-5/ \
    --dataroot  /home/ubuntu/02.data/01.original_data/MIST/PR/TrainValAB/ \
    --fine_path_label_dictionary dino-main/extracted_feats/MIST_PR_256/trainval_clusters100_coarse_path_label.pt \
    --fine_label_path_dictionary dino-main/extracted_feats/MIST_PR_256/trainval_clusters100_coarse_label_path.pt 


