import os
import numpy as np
import pandas as pd
import random
import h5py
import pickle
import itertools
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from sklearn.utils.class_weight import compute_sample_weight
from sksurv.metrics import concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate(batch):
    # Keep a numpy array on items that don't need to be a tensor for training.
    img = torch.cat([item[0] for item in batch], dim=0)
    img2 = torch.cat([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    censorship = torch.FloatTensor([item[4] for item in batch])
    stage = torch.LongTensor([item[5] for item in batch])
    slide_id =  [item[6] for item in batch]
    
    return [img, img2, label, event_time, censorship, stage, slide_id]
    

class FeatureBagsDataset(Dataset):
    def __init__(self, df, data_dir, input_feature_size, stage_class):
        self.slide_df = df.copy().reset_index(drop=True)
        self.data_dir = data_dir
        self.input_feature_size = input_feature_size
        self.stage_class = stage_class
    
    def _get_feature_path(self, slide_id):
        return os.path.join(self.data_dir, f"{slide_id}_Mergedfeatures.pt")

    def __getitem__(self, idx):
        slide_id = self.slide_df["slide_id"][idx]
        stage = self.slide_df["stage"][idx]
        label = self.slide_df["disc_label"][idx]
        event_time = self.slide_df["recurrence_years"][idx]
        censorship = self.slide_df["censorship"][idx]

        full_path = self._get_feature_path(slide_id)

        features = torch.load(full_path)

        # Merged features. 
        features_merged = torch.from_numpy(np.array([x[0].mean(0) for x in features]))

        # Alternative would be all features depending on what works best. 
        features_flattened = torch.from_numpy(np.concatenate([x[0] for x in features]))
    	
        return features_merged, features_flattened, label, event_time, censorship, stage, slide_id

    def __len__(self):
        return len(self.slide_df)

def define_data_sampling(train_split, val_split, method, workers):
    # Reproducibility of DataLoader.
    g = torch.Generator()
    g.manual_seed(0)

    # Set up training data sampler.
    if method == "random":
        print("random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            shuffle=True,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    val_loader = DataLoader(
            dataset=val_split,
            batch_size=1,  # model expects one bag of features at the time.
            sampler=SequentialSampler(val_split),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
    )

    return train_loader, val_loader

class MonitorBestModelEarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience and save best model """
    def __init__(self, patience=15, min_epochs=20, saving_checkpoint=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            min_epochs (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        #self.warmup = warmup
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.early_stop = False
        
        self.eval_loss_min = np.Inf
        self.best_loss_score = None
        self.best_epoch_loss = None

        self.best_CI_score = 0.0
        self.best_metrics_score = None
        self.best_epoch_CI = None

        self.saving_checkpoint = saving_checkpoint

    def __call__(self, epoch, eval_loss, eval_cindex, eval_other_metrics, model, log_dir):

        loss_score = -eval_loss
        CI_score = eval_cindex
        metrics_score = eval_other_metrics

        # Save model at epoch 0 and starts monitoring.
        if self.best_loss_score is None:
            self._update_loss_scores(loss_score, eval_loss, epoch)
            self._update_metrics_scores(CI_score, metrics_score, epoch)
            #self.save_checkpoint(model, log_dir, epoch)

        # Eval loss starts increasing. Recommend running early stopping on the loss.
        elif loss_score < self.best_loss_score:
            self.counter += 1
            print(f'Evaluation loss does not decrease : Starting Early stopping counter {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True
        # Eval loss keeps decreasing.
        else:
            print(f'Epoch {epoch} validation loss decreased ({self.eval_loss_min:.6f} --> {eval_loss:.6f})')
            self._update_loss_scores(loss_score, eval_loss, epoch)
            #self.save_checkpoint(model, log_dir, epoch)
            self.counter = 0
        
        # We may have a tiny lag between min loss and the best C-index. With the patience and early stop, it is fine but better to save based on highest C-index too. 
        if CI_score > self.best_CI_score:
            self._update_metrics_scores(CI_score, metrics_score, epoch)
            #self.save_checkpoint(model, log_dir, epoch)

    def save_checkpoint(self, model, log_dir, epoch):
        filepath = os.path.join(log_dir, f"{epoch}_checkpoint.pt")
        if self.saving_checkpoint and not os.path.exists(filepath):
            print(f"Saving model")
            torch.save(model.state_dict(), filepath)
        
    def _update_loss_scores(self, loss_score, eval_loss, epoch):
        self.eval_loss_min = eval_loss
        self.best_loss_score = loss_score
        self.best_epoch_loss = epoch
        print(f'Updating loss at epoch {self.best_epoch_loss} -> {self.eval_loss_min:.6f}')
    
    def _update_metrics_scores(self, CI_score, metrics_score, epoch):
        # Even though loss would decrease, C-index does not necessarily increase. Keep track also on best C-index.
        self.best_CI_score = CI_score
        self.best_epoch_CI = epoch
        self.best_metrics_score = metrics_score
        print(f'Updating C-index at epoch {self.best_epoch_CI} -> {self.best_CI_score:.6f}')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")

def get_survival_data_for_BS(df, time_col_name, censorship_col_name='censorship'):
    # To compute one survival metric the Brier score (BS), you need a specific format of censorship and times 
    # This is to estimate the censoring distribution from. 
    # A structured array containing the binary event indicator as first field (1 event occured; 0 censored), and time of event or time of censoring as second field.
    
    val = df[[censorship_col_name, time_col_name]].values
    max_time = df[time_col_name].max()

    y = np.empty(len(df), dtype=[('cens', '?'), ('time', '<f8')])
    for i in range(len(df)):
        y[i] = tuple((bool(1-val[i][0]),val[i][1])) # Note that we take the uncensorship status.
    return max_time, y

def get_bins_time_value(df, n_bins, time_col_name, label_time_col_name='disc_label', censorship_col_name='censorship'):
    # Retrieve the values of each bin from the dataset.
    # Note that this was done on the uncensored cases.
    uncensored_df = df[df[censorship_col_name]==0]
    labels, q_bins = pd.qcut(uncensored_df[time_col_name], q=n_bins, retbins=True, labels=False)
    q_bins[0] = 0
    q_bins[-1] = float('inf')

    # Current q_bins list length == n bins + 1. There is no need to return q_bins[0]==0.
    return q_bins[1::]

def compute_surv_metrics_eval(
    bins_values, 
    all_survival_probs, 
    all_risk_scores,
    train_BS, 
    test_BS, 
    years_of_interest=[1.0, 2.0, 3.0, 5.0], 
    yearI_of_interest=1.0,
    yearF_of_interest=5.0,
    time_step = 0.5):
    
    # Note that we discretized the continuous time scale into N bins for training, thus having only N survival probabilities at each n time step.
    # The Brier score does not return the exact same value if we use the discretize time or the continuous. 
    # Because it uses the distriution of censoring, even within each interval with same probability of survival.  
    # BS, IBS and AUC are time-dependent and step-dependent scores. See the notebook for some examples.
    
    #years_of_interest = [1.0, 2.0, 3.0, 5.0] #This can be completely a choice, depending on what makes sense. 
    corresponding_bins = np.asarray([np.argwhere(i<bins_values)[0,0] for i in years_of_interest])
    
    # Note that this requires that survival times survival_test lie within the range of survival times survival_train. 
    # This can be achieved by specifying times accordingly, e.g. by setting times[-1] slightly below the maximum expected follow-up time.
    _ , BS = brier_score(
        survival_train=train_BS, 
        survival_test=test_BS, 
        estimate=all_survival_probs[:,corresponding_bins], 
        times=years_of_interest)

    # The Integrated Brier Score (IBS) provides an overall calculation of the model performance at all available times. 
    # Both time points (at least two) and time step is a pure choice. Note that the time steps has an impact on the value of IBS. 
    
    #yearI_of_interest, yearF_of_interest = 1.0, 5.0 # Between 1Y and 5Y included.
    #time_step = 0.5 #6month
    years_of_interest_steps = np.arange(yearI_of_interest, yearF_of_interest+time_step, step=time_step, dtype=float) # Otherwise final year is not included.
    corresponding_bins = np.asarray([np.argwhere(i<bins_values)[0,0] for i in years_of_interest_steps])

    IBS = integrated_brier_score(
        survival_train=train_BS, 
        survival_test=test_BS, 
        estimate=all_survival_probs[:,corresponding_bins], 
        times=years_of_interest_steps)

    # The AUC can be extended to survival data by defining sensitivity (true positive rate) and specificity (true negative rate) as time-dependent measures. 
    # Cumulative cases are all individuals that experienced an event prior to or at time t (ti≤t), whereas dynamic controls are those with ti>t. 
    # The associated cumulative/dynamic AUC quantifies how well a model can distinguish subjects who fail by a given time (ti≤t) from subjects who fail after this time (ti>t).
    cumAUC, meanAUC = cumulative_dynamic_auc(
        survival_train=train_BS, 
        survival_test=test_BS, 
        estimate=all_risk_scores, #the AUC uses the risk scores like the C-index and not the event-free probabilities.
        times=years_of_interest_steps, tied_tol=1e-08)
    
    c_index_ipwc = concordance_index_ipcw(
        survival_train=train_BS,
        survival_test=test_BS,
        estimate=all_risk_scores, 
        tied_tol=1e-08)[0]

    return (BS, years_of_interest), (IBS, yearI_of_interest, yearF_of_interest), (cumAUC, meanAUC), (c_index_ipwc)