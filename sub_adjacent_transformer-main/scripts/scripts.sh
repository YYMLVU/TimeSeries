# # swat:
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 0.7 --num_epochs 1 --batch_size 256 --mode monte_carlo_search --dataset SWaT --data_path ./sub_adjacent_transformer-main/dataset/SWAT/ --input_c 51 --output_c 51 --span 20 30 --monte_carlo 1 --win_size 100
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 0.7 --num_epochs 2 --batch_size 256 --mode monte_carlo_search --dataset SWaT --data_path ./sub_adjacent_transformer-main/dataset/SWAT/ --input_c 51 --output_c 51 --span 20 30 --monte_carlo 1 --win_size 100

# # psm:
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 1  --batch_size 128  --mode monte_carlo_search --dataset PSM  --data_path ./sub_adjacent_transformer-main/dataset/PSM --input_c 25   --output_c 25 --monte_carlo 3 --win_size 100 --softmax_span_range 100 500 --softmax_span_step 50
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 2  --batch_size 128  --mode monte_carlo_search --dataset PSM  --data_path ./sub_adjacent_transformer-main/dataset/PSM --input_c 25   --output_c 25 --monte_carlo 3 --win_size 100 --softmax_span_range 100 500 --softmax_span_step 50
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 3  --batch_size 128  --mode monte_carlo_search --dataset PSM  --data_path ./sub_adjacent_transformer-main/dataset/PSM --input_c 25   --output_c 25 --monte_carlo 3 --win_size 100 --softmax_span_range 100 500 --softmax_span_step 50

# # smap
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 1   --batch_size 256  --mode monte_carlo_search --dataset SMAP  --data_path ./sub_adjacent_transformer-main/dataset/SMAP --input_c 25 --output_c 25 --monte_carlo 1 --win_size 100
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 2   --batch_size 256  --mode monte_carlo_search --dataset SMAP  --data_path ./sub_adjacent_transformer-main/dataset/SMAP --input_c 25 --output_c 25 --monte_carlo 1 --win_size 100
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode monte_carlo_search --dataset SMAP  --data_path ./sub_adjacent_transformer-main/dataset/SMAP --input_c 25 --output_c 25 --monte_carlo 1 --win_size 100

# # msl
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 1   --batch_size 128  --mode monte_carlo_search --dataset MSL  --data_path ./sub_adjacent_transformer-main/dataset/MSL --input_c 55 --output_c 55 --monte_carlo 1 --win_size 100
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 2   --batch_size 128  --mode monte_carlo_search --dataset MSL  --data_path ./sub_adjacent_transformer-main/dataset/MSL --input_c 55 --output_c 55 --monte_carlo 1 --win_size 100
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode monte_carlo_search --dataset MSL  --data_path ./sub_adjacent_transformer-main/dataset/MSL --input_c 55 --output_c 55 --monte_carlo 1 --win_size 100

bash ./sub_adjacent_transformer-main/scripts/swat.sh
# bash ./sub_adjacent_transformer-main/scripts/MSL.sh
# bash ./sub_adjacent_transformer-main/scripts/PSM.sh
bash ./sub_adjacent_transformer-main/scripts/SMAP.sh
# # wadi:
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 1  --batch_size 128  --mode monte_carlo_search --dataset wadi --data_path ./sub_adjacent_transformer-main/dataset/WADI  --input_c  93   --output_c  93  --monte_carlo 3 --win_size 100
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 2  --batch_size 128  --mode monte_carlo_search --dataset wadi --data_path ./sub_adjacent_transformer-main/dataset/WADI  --input_c  93   --output_c  93  --monte_carlo 3 --win_size 100
# python ./sub_adjacent_transformer-main/main.py --anormly_ratio 1 --num_epochs 3  --batch_size 128  --mode monte_carlo_search --dataset wadi --data_path ./sub_adjacent_transformer-main/dataset/WADI  --input_c  93   --output_c  93  --monte_carlo 3 --win_size 100
