export CUDA_VISIBLE_DEVICES=0

python ./sub_adjacent_transformer-main/main.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 64  --mode monte_carlo --dataset SMD  --data_path ./sub_adjacent_transformer-main/dataset/SMD   --input_c 38 --output_c 38 --monte_carlo 10
python ./sub_adjacent_transformer-main/main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256     --mode test    --dataset SMD   --data_path ./sub_adjacent_transformer-main/dataset/SMD     --input_c 38     --pretrained_model 20