window_size=2
batch_size=100
learn_rate=0.0001
channel_multiplier=2
disc=32
size=64
run=1

python ./trainTransGan.py \
    -pvW \
    --ndf $disc \
    --channel_multiplier $channel_multiplier \
    --batch_checkpoint 500 \
    --nheads 4,4,4,4,4,4,4 \
    --trans_layers 2,2,2,2,2,2 \
    --latent_size 100 \
    --batch_size $batch_size \
    --train \
    --trans_type swin \
    --size $size \
    --learning_rate $learn_rate \
    --epochs 40 \
    --path_cp checkpoints/s$size-win$window_size-bat$batch_size-mul$channel_multiplier-disc$disc-lr$learn_rate-run$run/ \
    --path_img_cp checkpoints/s$size-win$window_size-bat$batch_size-mul$channel_multiplier-disc$disc-lr$learn_rate-run$run/preview/ \
    --window_size $window_size \
    --workers 3