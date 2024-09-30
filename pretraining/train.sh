python main_visual.py \
    --gpus='0'  \
    --lr=2e-5 \
    --batch_size=100 \
    --num_workers=12 \
    --max_epoch=60 \
    --shaking_prob=0.2 \
    --max_magnitude=0.07 \
    --test=False \
    --n_dimention=500 \
    --temperture=0.07 \
    --save_prefix='checkpoints/' \
    --dataset='/content/LipLearner/pretraining/lrw_roi_63_99_191_227_size128_gray_jpeg' \
    --weights='/content/LipLearner/pretraining/best.pt' \
    
 
