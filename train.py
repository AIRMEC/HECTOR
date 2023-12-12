import os
import random
import subprocess
import argparse
import time
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

from model import HECTOR
from im4MEC import Im4MEC
from utils import *
from utils_loss import NLLSurvLoss

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def evaluate_model(epoch, model, model_mol, device, loader, n_bins, writer, loss_fn, bins_values, train_BS, test_BS):
    model.eval()

    eval_loss = 0.

    all_survival_probs = np.zeros((len(loader), n_bins))
    all_risk_scores = np.zeros((len(loader))) # This is the computed risk score.
    all_censorships = np.zeros((len(loader))) # This is the binary censorship status: 1 censored; 0 uncensored (reccured).
    all_event_times = np.zeros((len(loader)))

    with torch.no_grad():
        for batch_idx, (data, features_flattened, label, event_time, censorship, stage, _) in enumerate(loader):
            data, label, censorship, stage = data.to(device), label.to(device), censorship.to(device), stage.to(device)
            _, _, Y_hat, _, _ = model_mol(features_flattened.to(device))

            hazards_prob, survival_prob, Y_hat, _, _ = model(data, stage, Y_hat.squeeze(1)) # Returns hazards, survival, Y_hat, A_raw, M.

            # We can emphasize on the contribution of uncensored patient cases only in training by minimizing a weighted sum of the 2 losses
            loss = loss_fn(hazards=hazards_prob, S=survival_prob, Y=label, c=censorship, alpha=0)
            eval_loss += loss.item()

            risk = -torch.sum(survival_prob, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = censorship.cpu().numpy()
            all_event_times[batch_idx] = event_time
            all_survival_probs[batch_idx] = survival_prob.cpu().numpy()

    eval_loss /= len(loader)

    # Compute a few survival metrics.
    c_index = concordance_index_censored(
        event_indicator=(1-all_censorships).astype(bool), 
        event_time=all_event_times, 
        estimate=all_risk_scores, tied_tol=1e-08)[0]
    
    # Years of interest can be adapted in utils.py
    (BS, years_of_interest), (IBS, yearI_of_interest, yearF_of_interest), (_, meanAUC), (c_index_ipcw) = compute_surv_metrics_eval(bins_values, all_survival_probs, all_risk_scores, train_BS, test_BS)
    
    print(f'Eval epoch: {epoch}, loss: {eval_loss}, c_index: {c_index}, BS at each {years_of_interest}Y: {BS}, IBS and mean cumAUC from {yearI_of_interest}Y to {yearF_of_interest}Y: {IBS} and {meanAUC}')

    writer.add_scalar("Loss/eval", eval_loss, epoch)
    writer.add_scalar("C_index/eval", c_index, epoch)
    for i in range(len(years_of_interest)):
        writer.add_scalar(f"eval_metrics/BS_{str(years_of_interest[i])}Y", BS[i], epoch)
    writer.add_scalar(f"eval_metrics/IBS_{str(yearI_of_interest)}Y-{str(yearF_of_interest)}Y", IBS, epoch)
    writer.add_scalar(f"eval_metrics/meanAUC_{str(yearI_of_interest)}Y-{str(yearF_of_interest)}Y", meanAUC, epoch)

    return eval_loss, c_index, (BS, IBS, meanAUC, c_index_ipcw)

def train_one_epoch(epoch, model, model_mol, device, train_loader, optimizer, n_bins, writer, loss_fn):
    
    model.train()
    epoch_start_time = time.time()
    train_loss = 0.

    all_risk_scores = np.zeros((len(train_loader))) # Computed risk score.
    all_censorships = np.zeros((len(train_loader))) # Binary censorship status: 1 censored; 0 uncensored.
    all_event_times = np.zeros((len(train_loader))) # Real t event time or last follow-up.

    batch_start_time = time.time()

    for batch_idx, (data, features_flattened, label, event_time, censorship, stage, _) in enumerate(train_loader):

        data_load_duration = time.time() - batch_start_time

        data, label, censorship, stage = data.to(device), label.to(device), censorship.to(device), stage.to(device)
        # To get the image-based molecular class, non-merged features were used as this model was trained with way. 
        # Merged features could be used alternatively. 
        _, _, Y_hat, _, _ = model_mol(features_flattened.to(device))

        # Returns hazards, survival, Y_hat, A_raw, M.
        hazards_prob, survival_prob, Y_hat, _, _ = model(data, stage, Y_hat.squeeze(1)) 

        # Loss.
        loss = loss_fn(hazards=hazards_prob, S=survival_prob, Y=label, c=censorship)
        train_loss += loss.item()

        # Store outputs.
        risk = -torch.sum(survival_prob, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censorship.item()
        all_event_times[batch_idx] = event_time

        # Backward pass.
        loss.backward()

        # Step.
        optimizer.step()
        optimizer.zero_grad()

        batch_duration = time.time() - batch_start_time
        batch_start_time = time.time()

        writer.add_scalar("duration/data_load", data_load_duration, epoch)
        writer.add_scalar("duration/batch", batch_duration, epoch)

    epoch_duration = time.time() - epoch_start_time
    print(f"Finished training on epoch {epoch} in {epoch_duration:.2f}s")

    train_loss /= len(train_loader)

    train_c_index = concordance_index_censored(
        event_indicator=(1-all_censorships).astype(bool), 
        event_time=all_event_times, 
        estimate=all_risk_scores, tied_tol=1e-08)[0]
     
    print(f'Epoch: {epoch}, epoch_duration : {epoch_duration}, train_loss: {train_loss}, train_c_index: {train_c_index}')

    filepath = os.path.join(writer.log_dir, f"{epoch}_checkpoint.pt")
    print(f"Saving model to {filepath}")
    torch.save(model.state_dict(), filepath)

    writer.add_scalar("duration/epoch", epoch_duration, epoch)
    writer.add_scalar("LR", get_lr(optimizer), epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("C_index/train", train_c_index, epoch)

def run_train_eval_loop(train_loader, val_loader, loss_fn, hparams, run_id, BS_data, checkpoint_model_molecular):
    writer = SummaryWriter(os.path.join("./runs", run_id))
    device = torch.device("cuda")
    n_bins = hparams["n_bins"]

    model = HECTOR(
        input_feature_size=hparams["input_feature_size"],
        precompression_layer=hparams["precompression_layer"],
        feature_size_comp=hparams["feature_size_comp"],
        feature_size_attn=hparams["feature_size_attn"],
        postcompression_layer=hparams["postcompression_layer"],
        feature_size_comp_post=hparams["feature_size_comp_post"],
        dropout=True,
        p_dropout_fc=hparams["p_dropout_fc"],
        p_dropout_atn=hparams["p_dropout_atn"],
        n_classes=n_bins,

        input_stage_size=hparams["input_stage_size"],
        embedding_dim_stage=hparams["embedding_dim_stage"],
        depth_dim_stage=hparams["depth_dim_stage"],
        act_fct_stage=hparams["act_fct_stage"],
        dropout_stage=hparams["dropout_stage"],
        p_dropout_stage=hparams["p_dropout_stage"],

        input_mol_size=4,
        embedding_dim_mol=hparams["embedding_dim_mol"],
        depth_dim_mol=hparams["depth_dim_mol"],
        act_fct_mol=hparams["act_fct_mol"],
        dropout_mol=hparams["dropout_mol"],
        p_dropout_mol=hparams["p_dropout_mol"],

        fusion_type=hparams["fusion_type"],
        use_bilinear=hparams["use_bilinear"],
        gate_hist=hparams["gate_hist"],
        gate_stage=hparams["gate_stage"],
        gate_mol=hparams["gate_mol"],
        scale=hparams["scale"],
    ).to(device)
    print('model')
    print_model(model)

    # This model is instance with the trained weights towards molecular classification and will be used in inference mode only.
    # NOTE: it is important that the molecular model, here im4MEC, has been trained on the same patients as training to avoid patient-level information leakage. 
    model_mol = Im4MEC(
        input_feature_size=hparams["input_feature_size"],
        precompression_layer=True,
        feature_size_comp=hparams["feature_size_comp_molecular"],
        feature_size_attn=hparams["feature_size_attn_molecular"],
        n_classes=hparams["n_classes_molecular"],
        dropout=True, # Not used in inference.
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
    ).to(device)

    msg = model_mol.load_state_dict(torch.load(checkpoint_model_molecular, map_location=device), strict=True)
    print(msg)

    for p in model_mol.parameters():
        p.requires_grad = False
    print(f"HECTOR and plugged-in im4MEC are built and checkpoints loaded")
    model_mol.eval()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["initial_lr"],
        weight_decay=hparams["weight_decay"],
    )

    # Using a multi-step LR decay routine.
    milestones = [int(x) for x in hparams["milestones"].split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=hparams["gamma_lr"]
    )

    monitor_tracker = MonitorBestModelEarlyStopping(
        patience=hparams["earlystop_patience"],
        min_epochs=hparams["earlystop_min_epochs"],
        saving_checkpoint=True,
    )

    for epoch in range(hparams["max_epochs"]):

        train_one_epoch(epoch, model, model_mol, device, train_loader, optimizer, n_bins, writer, loss_fn)

        # Evaluation on validation set.
        print("Evaluating model on validation set...")
        eval_loss, eval_cindex, eval_other_metrics = evaluate_model(epoch, model, model_mol, device, val_loader, n_bins, writer, loss_fn, hparams["bins_values"], *BS_data) 
        monitor_tracker(epoch, eval_loss, eval_cindex, eval_other_metrics, model, writer.log_dir)

        # Update LR decay.
        scheduler.step()

        if monitor_tracker.early_stop:
            print(f"Early stop criterion reached. Broke off training loop after epoch {epoch}.")
            break
    
    # Log the hyperparameters of the experiments.
    runs_history = {
        "run_id" : run_id,
        "best_epoch_CI" : monitor_tracker.best_epoch_CI,
        "best_CI_score" : monitor_tracker.best_CI_score,
        "best_epoch_loss": monitor_tracker.best_epoch_loss,
        "best_evalLoss" : monitor_tracker.eval_loss_min,
        "BS" : monitor_tracker.best_metrics_score[0],
        "IBS" : monitor_tracker.best_metrics_score[1],
        "cumMeanAUC" : monitor_tracker.best_metrics_score[2],
        "CI_ipwc" : monitor_tracker.best_metrics_score[3],
        **hparams,
    }
    with open('runs_history.txt', 'a') as filehandle:
        for _, value in runs_history.items():
            filehandle.write('%s;' % value)
        filehandle.write('\n')

    writer.close()

def prepare_datasets(args):

    df = pd.read_csv(args.manifest)

    n_bins = len(df['disc_label'].unique())
    assert n_bins == args.n_bins, 'mismatch between the number of bins passed in args and classes in dataset'
    bins_values = get_bins_time_value(df, n_bins, time_col_name='recurrence_years', label_time_col_name='disc_label')
    assert len(bins_values)==n_bins
    print(f'Read {args.manifest} dataset containing {len(df)} samples with {n_bins} bins of following values {bins_values}')

    # NOTE: you may need to use the two lines below depending on how the category is listed in the csv file. 
    #df.stage = df.stage.apply(lambda x : 'III' if 'III' in x else ('II' if 'II' in x else 'I')).astype("category")
    #df.stage = pd.Categorical(df['stage'], categories=['I', 'II', 'III'], ordered=True).codes
    print(f'stage taxonomy used: {df.stage.unique()}')

    try:
        training_set = df[df["split"] == "training"]
        validation_set = df[df["split"] == "validation"]
    except:
        raise Exception(
            f"Could not find training and validation splits in {args.manifest}"
        )

    train_split = FeatureBagsDataset(df=training_set,
                                    data_dir=args.data_dir,
                                    input_feature_size=args.input_feature_size, 
                                    stage_class=len(training_set.stage.unique()))

    val_split = FeatureBagsDataset(df=validation_set,
                                    data_dir=args.data_dir, 
                                    input_feature_size=args.input_feature_size, 
                                    stage_class=len(validation_set.stage.unique()))

    # To compute the Brier score (BS), you need a specific format of censorship and times.
    _, train_BS = get_survival_data_for_BS(training_set, time_col_name='recurrence_years')
    _, test_BS = get_survival_data_for_BS(validation_set, time_col_name='recurrence_years')

    return train_split, val_split, train_BS, test_BS, bins_values, len(df.stage.unique())


def main(args):

    # Set random seed for some degree of reproducibility. See PyTorch docs on this topic for caveats.
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    set_seed()

    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA device available. Training without one is not feasible."
        )

    git_sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    train_run_id = f"{git_sha}_hp{args.hp}_{time.strftime('%Y%m%d-%H%M')}"

    train_split, val_split, train_BS, test_BS, bins_values, stage_taxonomy = prepare_datasets(args)

    print(f"=> Run ID {train_run_id}")
    print(f"=> Training on {len(train_split)} samples")
    print(f"=> Validating on {len(val_split)} samples")

    base_hparams = dict( 
        # Preprocessing settings. This should be changed with the dataset called accordingly.
        # Storing values here for readibility.
        n_bins=args.n_bins, # Partion on the continuous time scale.
        bins_values=bins_values,
        input_feature_size=args.input_feature_size,
        features_extraction=os.path.dirname(args.data_dir),

        # Settings that be changed in the loop:
        # Training.
        sampling_method="random",
        max_epochs=100,
        earlystop_warmup=0,
        earlystop_patience=30,
        earlystop_min_epochs=30,

        # Loss.
        alpha_surv = 0.0,

        # Optimizer.
        initial_lr=0.00003,
        milestones="2, 5, 15, 25",
        gamma_lr=0.1,
        weight_decay=0.00001,

        # Model architecture parameters. See model class for details.
        precompression_layer=True,
        feature_size_comp=512,
        feature_size_attn=256,
        postcompression_layer=True,
        feature_size_comp_post=128,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,

        # Model of molecular classification. In our case only inference is used. 
        n_classes_molecular=args.n_classes_molecular,
        feature_size_comp_molecular=args.feature_size_comp_molecular,
        feature_size_attn_molecular=args.feature_size_attn_molecular,

        # Fusion parameters.
        input_stage_size=stage_taxonomy,
        embedding_dim_stage=16,
        depth_dim_stage=1,
        act_fct_stage='elu',
        dropout_stage=True,
        p_dropout_stage=0.25,
        embedding_dim_mol=16,
        depth_dim_mol=1,
        act_fct_mol='elu',
        dropout_mol=True,
        p_dropout_mol=0.25,
        fusion_type='bilinear',
        use_bilinear=[True,True,True],
        gate_hist=True,
        gate_stage=True,
        gate_mol=True,
        scale=[2,1,1],
    )

    hparam_sets = [
        {
            **base_hparams,
        },
    ]

    hps = hparam_sets[args.hp]


    train_loader, val_loader = define_data_sampling(
            train_split,
            val_split,
            method=hps["sampling_method"],
            workers=args.workers,
    )

    run_train_eval_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn = NLLSurvLoss(alpha=hps["alpha_surv"]), # Used the Negative log likelihood loss.
            hparams=hps,
            run_id=train_run_id,
            BS_data = (train_BS, test_BS),
            checkpoint_model_molecular=args.checkpoint_model_molecular, 
    )
    print("Finished training.")

def get_args_parser():
    
    parser = argparse.ArgumentParser('Training script', add_help=False)

    parser.add_argument(
        "--manifest",
        type=str,
        help="CSV file listing all slides, their labels, and which split (train/test/val) they belong to.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        help="Number of time intervals used to create the time labels. It should be the same as the manifest.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where all *_features.h5 files are stored",
    )
    parser.add_argument(
        "--input_feature_size",
        help="The size of the input features from the feature bags. Recommend going by blocks from these output size [96, 96, 192, 192, 384, 384, 384, 384, 768, 768]",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--checkpoint_model_molecular",
        type=str,
        default='',
        help="Path to checkpoint of im4MEC",
    )
    parser.add_argument(
        "--n_classes_molecular",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--feature_size_comp_molecular",
        type=int,
        required=True,
        help="Size of the model of the trained im4MEC. See in im4MEC.py",
    )
    parser.add_argument(
        "--feature_size_attn_molecular",
        type=int,
        required=True,
        help="Size of the model of the trained im4MEC. See in im4MEC.py",
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--hp",
        type=int,
        required=True,
    )

    return parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Training script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
