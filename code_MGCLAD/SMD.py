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
warnings.filterwarnings("ignore")


def SMD(args):

    id = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = args.dataset
    window_size = args.lookback
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

    group = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8", "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9", "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(group)):
        output_path = f'./code_MGCLAD/output/SMD/{group[i]}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group[i]}", normalize=normalize)
        group_index = group[i].split("-")[0]

        log_dir = f'{output_path}/logs'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_path = f"{output_path}/{id}"

        x_train = torch.from_numpy(x_train).float()
        x_test = torch.from_numpy(x_test).float()
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

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
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

        trainer.fit(train_loader, val_loader)

        plot_losses_recon(trainer.losses, save_path=save_path, plot=False)

        # Check test loss
        # test_loss = trainer.evaluate(test_loader)
        # print(f"Test reconstruction loss: {test_loss[0]:.5f}")

        # Some suggestions for POT args
        level_q_dict = {
            "SMD-1": (0.9950, 0.001),
            "SMD-2": (0.9925, 0.001),
            "SMD-3": (0.9999, 0.001),
        }

        key = "SMD-" + group_index
        level, q = level_q_dict[key]
        if args.level is not None:
            level = args.level
        if args.q is not None:
            q = args.q

        # Some suggestions for Epsilon args
        reg_level_dict = {"SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
        key = "SMD-" + group_index
        reg_level = reg_level_dict[key]
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
        
        # 读取summary.txt文件
        with open(f"{save_path}/summary.txt", "r") as f:
            summary = json.load(f)
        tp += summary["bf_result"]["TP"]
        tn += summary["bf_result"]["TN"]
        fp += summary["bf_result"]["FP"]
        fn += summary["bf_result"]["FN"]

    # 统计所有的结果
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    print(f"precision:{precision}, recall:{recall}, f1:{f1}")
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"./code_MGCLAD/output/SMD/summary.txt", "a") as f:
        json.dump({"time": time, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}, f, indent=2)
    return 

def SMD1(args):

    id = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = args.dataset
    window_size = args.lookback
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    # group_index = args.group[0]
    # index = args.group[2]
    args_summary = str(args.__dict__)

    # group = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8", "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9", "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # for i in range(len(group)):
    output_path = f'./code_MGCLAD/output/SMD/completed'
    # (x_train, _), (x_test, y_test) = get_data(f"machine-{group[i]}", normalize=normalize)
    prefix = "./code_MGCLAD/datasets"
    prefix += "/ServerMachineDataset/processed"

    x_dim = get_data_dim(dataset)
    train_end = None
    test_end = None
    train_start = 0
    test_start = 0
    f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")        
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)

    x_train, x_test, y_test = train_data, test_data, test_label

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
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

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
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

    trainer.fit(train_loader, val_loader)

    plot_losses_recon(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    # test_loss = trainer.evaluate(test_loader)
    # print(f"Test reconstruction loss: {test_loss[0]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMD": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
    }

    key = "SMD"
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level = 1
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

    return 