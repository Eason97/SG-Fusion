import numpy as np
from lifelines.utils import concordance_index
import torch

def CoxLoss(survtime, censor, hazard_pred, device):
    current_batch_len = len(survtime)
    R_mat = np.zeros((current_batch_len, current_batch_len), dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.view(-1)  # Ensure that hazard_pred is a 1D tensor
    exp_theta = torch.exp(theta)
    weighted_sum_exp_theta = torch.sum(exp_theta * R_mat, dim=1)
    log_sum_exp_theta = torch.log(weighted_sum_exp_theta)
    loss_cox = -torch.mean((theta - log_sum_exp_theta) * censor)
    return loss_cox

def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))