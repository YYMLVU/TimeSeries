python train_cgraph_trans.py --dataset SMD --group 1-1 --lookback 32 \
--normalize True --spec_res False """ Data Params """ \
""" Model Params """ --kernel_size 7 --use_gatv2 True \
--feat_gat_embed_dim None --time_gat_embed_dim None \
--gru_n_layers 1 --gru_hid_dim 100 --fc_n_layers 2 \
--fc_hid_dim 100 --recon_n_layers 1 --recon_hid_dim 100 --alpha 0.2 \
""" Train Params""" --epochs 30 --val_split 0.1 --bs 512 \
--init_lr 5*1e-4 --shuffle_dataset True --dropout 0.3 \
--use_cuda True --print_every 1 --log_tensorboard True \
""" Predictor Params """ --scale_scores False --use_mov_av False \
--gamma 1 --level None --q None --dynamic_pot False \
--comment ""