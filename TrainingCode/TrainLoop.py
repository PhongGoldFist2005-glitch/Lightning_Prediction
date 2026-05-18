import sys
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, f1_score,
    recall_score, precision_score
)
from tqdm import tqdm
import numpy as np
import copy
import torch
from sklearn.metrics import auc

sys.path.append("src/Phong/Model/TrainingCode")

class trainLoop:
    def __init__(self, epochs, my_model, device, loss_fn, optimizer, scheduler, outputLabel, fullBand, train_dataset, val_dataset, wandbObject, threshold, metricInfo, earlyStop, patience, batch_size):
        self.epochs = epochs
        self.my_model = my_model
        
        self.device = device
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Output label is a list of label columns in df.
        self.outputLabel = outputLabel
        # fullBand is a list of all feature columns in df.
        
        self.fullBand = fullBand
        # train_dataset and val_dataset are Dataloader object
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.wandbObject = wandbObject
        self.metricInfo = metricInfo

        self.threshold = threshold

        self.earlyStop = earlyStop
        self.patience = patience

        self.batch_size = batch_size

    def trainBinaryOutput(self):
        # const params
        num_horizon = len(self.outputLabel)
        counter = 0

        # best information to save
        best_model_wts = None
        best_PR_AUC = float('-inf')
        best_precision_class_0 = None
        best_precision_class_1 = None
        best_recall_class_0 = None
        best_recall_class_1 = None
        best_f1_class_0 = None
        best_f1_class_1 = None
        best_avg_train_loss = None
        best_eval_loss = None


        for epoch in tqdm(range(self.epochs)):
            trainLoss = 0.0
            numTrain = 0.0

            val_losses_per_step = np.zeros(num_horizon)
            numEvalBatches = np.zeros(num_horizon)
            y_true_each_epoch = [[] for _ in range(num_horizon)]
            y_pred_each_epoch = [[] for _ in range(num_horizon)]

            self.my_model.train()
            for batch, (X, y) in tqdm(enumerate(self.train_dataset), desc=f"{epoch + 1} epoch", leave=False):
                X = X.to(self.device)
                y = y.to(self.device)
                
                output = self.my_model(X).squeeze()
                
                currSize = y.size(0)
                numTrain += currSize

                loss_train = self.loss_fn(output, y)
                trainLoss += loss_train.item() * currSize

                # Zero grad
                self.optimizer.zero_grad()
            
                # Backpropagation
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(self.my_model.parameters(), max_norm=1.0)

                # Optimizer
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Validation loop
            self.my_model.eval()
            for batch, (X_full, y_full) in enumerate(self.val_dataset):
                X_full = X_full.to(self.device)
                y_full = y_full.to(self.device)

                with torch.inference_mode():
                    for t in range(num_horizon):
                        X_curr = X_full[:, t : t + 6, :]
                        y_curr = y_full[:, t]

                        output = self.my_model(X_curr).squeeze()

                        # ✅ DEBUG: Check if output contains NaN or Inf
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            nan_count = torch.isnan(output).sum().item()
                            inf_count = torch.isinf(output).sum().item()
                            print(f"❌ ALERT at epoch {epoch}, t={t}: NaN={nan_count}, Inf={inf_count}")
                        
                        loss_eval = self.loss_fn(output, y_curr)
                        val_losses_per_step[t] += loss_eval.item() * y_curr.size(0)
                        numEvalBatches[t] += y_curr.size(0)

                        y_true_each_epoch[t].extend(y_curr.cpu().numpy())
                        sigmoid_output = torch.sigmoid(output).cpu().numpy()
                        
                        # ✅ DEBUG: Check if sigmoid output contains NaN
                        if np.isnan(sigmoid_output).any():
                            print(f"❌ ERROR at epoch {epoch}, t={t}: sigmoid output has {np.isnan(sigmoid_output).sum()} NaN values")
                        
                        y_pred_each_epoch[t].extend(sigmoid_output)
            
        
            # Calculate precsion, recall, f1, pr_auc for each step, each class
            precisions_0, recalls_0, f1s_0 = [], [], []
            precisions_1, recalls_1, f1s_1 = [], [], []
            pr_aucs = []

            for t in range(num_horizon):
                y_true_np = np.array(y_true_each_epoch[t])
                y_pred_np = np.array(y_pred_each_epoch[t])

                # ✅ Filter NaN/Inf before metrics calculation
                valid_mask = np.isfinite(y_pred_np)
                if not np.all(valid_mask):
                    nan_count = np.sum(~valid_mask)
                    print(f"⚠️ Filtering {nan_count} NaN/Inf at epoch {epoch}, t={t}")
                    y_true_np = y_true_np[valid_mask]
                    y_pred_np = y_pred_np[valid_mask]
                
                if len(y_pred_np) == 0:
                    print(f"❌ ERROR: All predictions are NaN at t={t}!")
                    precisions_0.append(0.0)
                    recalls_0.append(0.0)
                    f1s_0.append(0.0)
                    precisions_1.append(0.0)
                    recalls_1.append(0.0)
                    f1s_1.append(0.0)
                    pr_aucs.append(0.0)
                    continue

                y_pred_binary = (y_pred_np >= self.threshold).astype(int)


                p_per_class = precision_score(y_true_np, y_pred_binary, labels= [0, 1], average= None, zero_division= 0)
                r_per_class = recall_score(y_true_np, y_pred_binary, labels= [0, 1], average= None, zero_division= 0)
                f_per_class = f1_score(y_true_np, y_pred_binary, labels= [0,1], average= None, zero_division= 0)

                precisions_0.append(p_per_class[0])
                recalls_0.append(r_per_class[0])
                f1s_0.append(f_per_class[0])

                precisions_1.append(p_per_class[1])
                recalls_1.append(r_per_class[1])
                f1s_1.append(f_per_class[1])
                
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_pred_np)
                    pr_auc = auc(recall_curve, precision_curve)
                    pr_aucs.append(pr_auc)
                except Exception as e:
                    print(f"⚠️ Error calculating PR curve at t={t}: {str(e)}")
                    pr_auc = 0.0
                    pr_aucs.append(pr_auc)
            
            del y_true_np, y_pred_np, y_pred_binary

            # Calculate metrics in each epoch
            avg_train_loss = trainLoss / numTrain if numTrain > 0 else 0
            eval_losses = val_losses_per_step / numEvalBatches if np.any(numEvalBatches > 0) else np.zeros(num_horizon)
            avg_pr_auc = np.mean(pr_aucs)

            dict_result = {
                "avg_train_loss": avg_train_loss,
                "avg_pr_auc": avg_pr_auc,
                "epochs": epoch + 1
            }

            for t in range(num_horizon):
                dict_result.update({
                    f"eval_loss_t{t}": eval_losses[t],
                    f"precision_class_0_t{t}": precisions_0[t],
                    f"recall_class_0_t{t}": recalls_0[t],
                    f"f1_class_0_t{t}": f1s_0[t],
                    f"precision_class_1_t{t}": precisions_1[t],
                    f"recall_class_1_t{t}": recalls_1[t],
                    f"f1_class_1_t{t}": f1s_1[t]
                })
            
            self.wandbObject.log(dict_result)

            if avg_pr_auc > best_PR_AUC + 1e-8:
                best_PR_AUC = avg_pr_auc
                best_model_wts = copy.deepcopy(self.my_model.state_dict())
                best_precision_class_0 = precisions_0
                best_precision_class_1 = precisions_1
                best_recall_class_0 = recalls_0
                best_recall_class_1 = recalls_1
                best_f1_class_0 = f1s_0
                best_f1_class_1 = f1s_1
                best_avg_train_loss = avg_train_loss
                best_eval_loss = eval_losses
                counter = 0
            else:
                counter += 1
            
            if self.earlyStop and counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.wandbObject.finish()

        with open(self.metricInfo, "a") as f:
            f.write(f"Best PR AUC: {best_PR_AUC:.4f}\n")
            f.write(f"Best avg train loss: {best_avg_train_loss:.4f}\n")
            for t in range(num_horizon):
                f.write(f"Best eval loss at step {t}: {best_eval_loss[t]:.4f}\n")
                f.write(f"Step {t} precision class 0: {best_precision_class_0[t]:.4f}--recall: {best_recall_class_0[t]:.4f}--f1: {best_f1_class_0[t]:.4f}\n")
                f.write(f"Step {t} precision class 1: {best_precision_class_1[t]:.4f}--recall: {best_recall_class_1[t]:.4f}--f1: {best_f1_class_1[t]:.4f}\n")
        
        if best_model_wts is not None:
            self.my_model.load_state_dict(best_model_wts)
        
        return self.my_model
    
    def trainRegOutput(self):
        pass