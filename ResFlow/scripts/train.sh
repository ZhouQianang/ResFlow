CHECKPOINT_DIR=/data/DSEC-Flow/TDF_CKPT/base_16bins_sca_n03 && \
RESUME_DIR=/data/DSEC-Flow/BASE_CKPT/baseline_16bins && \
CUDA_VISIBLE_DEVICES=0 
python train_Dsec_separate.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--num_steps 20000 \
--batch_size 6 \
--lr 2e-4 \
--ckpt_path ${RESUME_DIR}/checkpoint_200000.pth \
--global_frozen \
--bn_frozen \
--add_noise \
--noise_weight 0.3


