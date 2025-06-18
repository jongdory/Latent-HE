import numpy as np
import argparse, os, sys, datetime, glob, importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    print(get_obj_from_str(config["target"]))
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="train the model",
    )
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="evaluate the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="random seed",
    )

    return parser


def compute_multilabel_metrics(all_targets, all_preds, binary_preds):
    """
    all_targets: np.array, shape (N, C), target one-hot encoded
    all_preds:   np.array, shape (N, C), predicted probabilities
    binary_preds:np.array, shape (N, C), predicted binary labels
    """
    epoch_acc = accuracy_score(all_targets, binary_preds)
    
    # AUC (micro, macro, weighted)
    epoch_auc_micro = roc_auc_score(all_targets, all_preds, average='micro')
    epoch_auc_macro = roc_auc_score(all_targets, all_preds, average='macro')
    epoch_auc_weighted = roc_auc_score(all_targets, all_preds, average='weighted')
    
    # F1 (micro, macro, weighted)
    epoch_f1_micro = f1_score(all_targets, binary_preds, average='micro')
    epoch_f1_macro = f1_score(all_targets, binary_preds, average='macro')
    epoch_f1_weighted = f1_score(all_targets, binary_preds, average='weighted')
    
    metrics_dict = {
        'acc': epoch_acc,
        'auc_micro': epoch_auc_micro,
        'auc_macro': epoch_auc_macro,
        'auc_weighted': epoch_auc_weighted,
        'f1_micro': epoch_f1_micro,
        'f1_macro': epoch_f1_macro,
        'f1_weighted': epoch_f1_weighted
    }
    return metrics_dict


def format_metrics(epoch, num_epochs, loss, metrics_dict):
    s = (
        f"Epoch {epoch+1}/{num_epochs}: "
        f"Loss: {loss:.4f}, "
        f"Acc: {metrics_dict['acc']:.4f}, "
        f"AUC micro: {metrics_dict['auc_micro']:.4f}, "
        f"AUC macro: {metrics_dict['auc_macro']:.4f}, "
        f"AUC weighted: {metrics_dict['auc_weighted']:.4f}, "
        f"F1 micro: {metrics_dict['f1_micro']:.4f}, "
        f"F1 macro: {metrics_dict['f1_macro']:.4f}, "
        f"F1 weighted: {metrics_dict['f1_weighted']:.4f}"
    )
    return s

def compute_per_class_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    num_classes = y_true.shape[1]
    
    auc_list = []
    f1_list = []
    
    for c in range(num_classes):
        auc_val = roc_auc_score(y_true[:, c], y_pred_proba[:, c])
        f1_val = f1_score(y_true[:, c], y_pred_bin[:, c], zero_division=0)
        
        auc_list.append(auc_val)
        f1_list.append(f1_val)
    
    metrics_dict = {
        'auc': auc_list,
        'f1': f1_list
    }
    return metrics_dict

def format_per_class_metrics(metrics_dict):
    s = "Per class metrics:\n"
    for c, (auc, f1) in enumerate(zip(metrics_dict['auc'], metrics_dict['f1'])):
        s += f"Class {c}: AUC={auc:.4f}, F1={f1:.4f}\n"
    return s


def train(model, loss_fn, optimizer, param, loader_train, loader_val=None, logger=None):
    best_f1 = 0.0
    best_str = ""
    for epoch in range(param['num_epochs']):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        for t, (x, y) in enumerate(loader_train):
            x, y = x.cuda(), y.cuda().float()

            scores = model(x)
            loss = loss_fn(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            probs = torch.sigmoid(scores).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.detach().cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)     # (N, C)
        all_targets = np.concatenate(all_targets, axis=0)
        binary_preds = (all_preds >= 0.5).astype(int)

        epoch_loss = epoch_loss / (t + 1)

        metrics_dict = compute_multilabel_metrics(all_targets, all_preds, binary_preds)
        metrics_str = format_metrics(epoch, param['num_epochs'], epoch_loss, metrics_dict)
        print(metrics_str)
        
        if logger is not None:
            logger.log(epoch, epoch_loss, metrics_dict, mode="train")

        if (epoch + 1) % 10 == 0:
            eval_metrics_dict = eval(model, loader_val)
            eval_metrics_str = format_metrics(epoch, param['num_epochs'], epoch_loss, eval_metrics_dict)
            print("Eval -", eval_metrics_str)
            print("Best -", best_str)

            logger.save_checkpoint(model, optimizer, f"model_{epoch+1}.pth")
            if logger is not None:
                logger.log(epoch, epoch_loss, eval_metrics_dict, mode="val")

            f1_macro = eval_metrics_dict['f1_macro']
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_str = f"{eval_metrics_str}"
                logger.save_checkpoint(model, optimizer, f"best_model.pth")

    logger.save_checkpoint(model, optimizer, f"last_model.pth")
        

def eval(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_iter = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda().float()
            scores = model(x)

            probs = torch.sigmoid(scores).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.detach().cpu().numpy())
            n_iter += 1

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    binary_preds = (all_preds >= 0.5).astype(int)

    metrics_dict = compute_multilabel_metrics(all_targets, all_preds, binary_preds)

    return metrics_dict

def eval_all_classes(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_iter = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda().float()
            scores = model(x)

            probs = torch.sigmoid(scores).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.detach().cpu().numpy())
            n_iter += 1

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    binary_preds = (all_preds >= 0.5).astype(int)

    metrics_dict = compute_per_class_metrics(all_targets, all_preds, binary_preds)

    return metrics_dict


class DataModuleFromConfig:
    def __init__(self, batch_size, train=None, validation=None, test=None, num_workers=None):
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else 8
        self.datasets = {}

        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def prepare_data(self):
        # Prepares any data needed but does not load datasets (optional depending on use case)
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self):
        # Instantiates and optionally wraps datasets
        self.datasets = {
            k: instantiate_from_config(cfg) for k, cfg in self.dataset_configs.items()
        }

    def _train_dataloader(self):
        if "train" in self.datasets:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=True)
        else:
            raise ValueError("Train dataset not available. Check setup_datasets configuration.")

    def _val_dataloader(self):
        if "validation" in self.datasets:
            return DataLoader(self.datasets["validation"], batch_size=self.batch_size,
                              num_workers=self.num_workers)
        else:
            raise ValueError("Validation dataset not available. Check setup_datasets configuration.")

    def _test_dataloader(self):
        if "test" in self.datasets:
            return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                              num_workers=self.num_workers)
        else:
            raise ValueError("Test dataset not available. Check setup_datasets configuration.")

class Logger:
    def __init__(self, log_name):
        self.logdir = "logs/" + log_name
        os.makedirs(self.logdir, exist_ok=True)
        self.train_log = os.path.join(self.logdir, "log_train.csv")
        self.val_log = os.path.join(self.logdir, "log_val.csv")
        
        if not os.path.exists(self.train_log):
            with open(self.train_log, 'w') as f:
                f.write("epoch,loss,acc,auc_micro,auc_macro,auc_weighted,f1_micro,f1_macro,f1_weighted\n")
        if not os.path.exists(self.val_log):
            with open(self.val_log, 'w') as f:
                f.write("epoch,loss,acc,auc_micro,auc_macro,auc_weighted,f1_micro,f1_macro,f1_weighted\n")

    def log(self, epoch, loss, metrics_dict, mode):
        """
        metrics_dict: {'acc':..., 'auc_micro':..., 'auc_macro':..., 'auc_weighted':..., 
                       'f1_micro':..., 'f1_macro':..., 'f1_weighted':...}
        """
        acc = metrics_dict['acc']
        auc_micro = metrics_dict['auc_micro']
        auc_macro = metrics_dict['auc_macro']
        auc_weighted = metrics_dict['auc_weighted']
        f1_micro = metrics_dict['f1_micro']
        f1_macro = metrics_dict['f1_macro']
        f1_weighted = metrics_dict['f1_weighted']
        
        if mode == "train":
            with open(self.train_log, "a") as f:
                f.write(f"{epoch},{loss},{acc},{auc_micro},{auc_macro},{auc_weighted},{f1_micro},{f1_macro},{f1_weighted}\n")
        elif mode == "val":
            with open(self.val_log, "a") as f:
                f.write(f"{epoch},{loss},{acc},{auc_micro},{auc_macro},{auc_weighted},{f1_micro},{f1_macro},{f1_weighted}\n")

    def save_config(self, config):
        with open(os.path.join(self.logdir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f.name)

    def save_checkpoint(self, model, optimizer, ckpt_path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.logdir, ckpt_path))

    def load_checkpoint(self, model, optimizer, ckpt_path):
        checkpoint = torch.load(os.path.join(self.logdir, ckpt_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer


if __name__ == '__main__':
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    config = OmegaConf.load(opt.base)
    model = instantiate_from_config(config.model).cuda()

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    loader_train = data._train_dataloader()
    loader_val = data._val_dataloader()

    print("Number of training samples:", len(loader_train.dataset))
    print("Number of validation samples:", len(loader_val.dataset))
    print("Latent shape:", loader_train.dataset[0][0].shape)

    log_name = opt.base.split("/")[-1].split(".")[0]
    logger = Logger(log_name)

    param = config.params
    loss_fn = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'], amsgrad=True)

    # Train the model
    if opt.train:
        train(model, loss_fn, optimizer, param, loader_train, loader_val, logger)
    if opt.eval:
        model, _ = logger.load_checkpoint(model, optimizer, "best_model.pth")
        eval_metrics_dict = eval(model, loader_val)
        eval_metrics_str = format_metrics(0, param['num_epochs'], 0.0, eval_metrics_dict)
        print("Eval -", eval_metrics_str)
        eval_all_dict = eval_all_classes(model, loader_val)
        print(format_per_class_metrics(eval_all_dict))