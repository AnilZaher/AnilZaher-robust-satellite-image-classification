import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pickle

from models.backbones import get_model
from utils.data_utils import get_loaders
from utils.trainer import train, validate


def run_optuna_search(model_list, n_trials=15):
    # data and saving directories
    checkpoints_dir = "models/checkpoints"
    logs_dir = "models/logs"
    data_dir = "data/splits"

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_loaders(data_dir, batch_size=32)

    all_studies = {}

    for model_name in model_list:
        print(f"\n>>> Optimizing: {model_name}")

        # used for saving the best model in the study
        global_best = {"val_acc": 0.0, "trial": None, "epoch": None}

        study = optuna.create_study(study_name=f"{model_name}_study", direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))

        study.optimize(lambda t: objective(t, model_name, dataloaders, device, checkpoints_dir, logs_dir, global_best), n_trials=n_trials)

        # saving the studies (in order to have the ability to access them later)
        with open(os.path.join(logs_dir, f"{model_name}_study.pkl"), "wb") as f:
            pickle.dump(study, f)

        all_studies[model_name] = study

    return all_studies


def objective(trial, model_name, dataloaders, device, checkpoints_dir, logs_dir, global_best):
    # optimizing the learning rate, weight decay and optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])

    fine_tune = model_name != "dinov3"
    epochs = 30 if model_name == "dinov3" else 10 # training longer when doing just linear probing

    model = get_model(model_name, num_classes=10, fine_tune=fine_tune).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if optimizer_name == "AdamW" else optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = validate(model, dataloaders["val"], criterion, device)

        scheduler.step()
        trial.report(val_acc, epoch)

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        if val_acc > global_best["val_acc"]: # saving the best preforming model
            global_best.update({"val_acc": val_acc, "trial": trial.number, "epoch": epoch})

            torch.save({"model_state": model.state_dict(), "val_acc": val_acc, "trial": trial.number, "epoch": epoch, "params": trial.params}, os.path.join(checkpoints_dir, f"BEST_{model_name}.pth"))

            torch.save(metrics, os.path.join(logs_dir, f"{model_name}_metrics.pt"))

        if trial.should_prune(): # pruning trail that should be pruned
            raise optuna.exceptions.TrialPruned()

    return global_best["val_acc"]
