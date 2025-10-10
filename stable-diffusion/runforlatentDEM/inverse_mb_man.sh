export CUDA_VISIBLE_DEVICES='0'
python ./scripts/inference.py \
    --file_id='manseed95.png' \
    --task_config='configs/motion_deblur_config_psld_64_32-2.yaml' \
    --outdir='./PSLD_results/20251009_releasecheck_3' \
    --seed=95