import json
from datetime import datetime
import torch.nn as nn

from args import get_parser
from utils import *
#from model_cgraph_trans_d import MODEL_CGRAPH_TRANS
from models.model_cgraph_trans import MODEL_CGRAPH_TRANS
# from transformers import AutoModelForCausalLM
from prediction_recon_cgraph import Predictor
from training_recon_cgraph import Trainer
# from training import Trainer
import warnings
import SMD
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    id = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2]
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        # output_path = f'./code_MGCLAD/output/SMD/{args.group}'
        # (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
        SMD.SMD(args)
        # 结束程序运行
        exit()
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'./code_MGCLAD/output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        with open(f'./code_MGCLAD/datasets/data/processed/{dataset}_train_aug_neg.pkl', 'rb') as f:
            x_train_aug = pickle.load(f)
        x_train_aug, _ = normalize_data(x_train_aug, scaler=None)
    elif dataset in ['SWAT', 'WADI', 'PSM']:
        output_path = f'./code_MGCLAD/output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        dataset_lower = dataset.lower()
        # with open(f'./code_MGCLAD/datasets/{dataset_lower}/processed/{dataset}_train_aug.pkl', 'rb') as f:
        #     x_train_aug = pickle.load(f)
        # x_train_aug, _ = normalize_data(x_train_aug, scaler=None)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
    # save_path = './code_MGCLAD/output/MSL/20250407_152652'


    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    x_train_aug = torch.from_numpy(x_train_aug).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    train_aug_dataset = SlidingWindowDataset(x_train_aug, window_size, target_dims)

    # train_loader, val_loader, test_loader = create_data_loaders(
    #     train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    # )

    train_loader, train_aug_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset, train_aug_dataset=train_aug_dataset
    )

    model = MODEL_CGRAPH_TRANS(
        n_features,
        window_size,
        out_dim,
        dropout=args.dropout,
        device="cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    # device = "cuda:0" if torch.cuda.is_available() else "cpu" # add
    # model = AutoModelForCausalLM.from_pretrained('./weight/', trust_remote_code=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader, val_loader, train_aug_loader)

    plot_losses_recon(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test reconstruction loss: {test_loss[0]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "SWAT": (0.99, 0.001),
        "WADI": (0.99, 0.001),
        "PSM": (0.99, 0.001)
    }

    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "SWAT": 1, "WADI": 1, "PSM": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]
    # save_path = "/home/xiaoqpz/code_MGCLAD/output/MSL/20250303_231516"
    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
        # 'device': device, # add
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None
    print('label shape:', label.shape)
    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
