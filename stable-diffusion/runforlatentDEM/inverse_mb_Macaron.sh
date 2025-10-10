export CUDA_VISIBLE_DEVICES='0'
python ./scripts/inference.py \
    --file_id='Macaronseed20.png' \
    --task_config='configs/motion_deblur_config_psld_64_32-2.yaml' \
    --outdir='./PSLD_results/20251009_releasecheck_2' \
    --seed=20