import torch
from fused import fused_net
from train_dataset import multimodal_train_dataset
from test_dataset import multimodal_test_dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import csv
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from utils import CoxLoss,CIndex_lifeline
import pandas as pd
import torch.nn as nn
import os

class CustomLossWithPCARegularization(nn.Module):
    def __init__(self, alpha, n_components):
        super(CustomLossWithPCARegularization, self).__init__()
        self.alpha = alpha
        self.n_components = n_components
    def forward(self, feature_map):
        batch_size, num_features = feature_map.size(0), feature_map.size(1)
        flattened = feature_map.view(batch_size, num_features, -1)
        _, _, v = torch.svd(flattened)
        principal_components = v[:, :, :self.n_components]
        pca_divergence = torch.norm(principal_components.transpose(1, 2) @ principal_components -
                                    torch.eye(self.n_components).to(principal_components.device), p='fro')
        return pca_divergence
def create_dict_from_csv(csv_file):
    label_dict = {}
    mrna_dict = {}
    survtime_dict = {}
    censor_dict = {}
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            key = row[0]
            labels = row[1]
            survtime= row[2]
            censor = row[3]
            mrna = row[4:]
            mrna = [float(item) for item in mrna]
            label_dict[key] = labels
            mrna_dict[key] = mrna
            survtime_dict[key] = survtime
            censor_dict[key] = censor
    return label_dict,mrna_dict,survtime_dict,censor_dict
csv_file = "/home/eason/Pictures/all_datasets/3_ours_mrna/gbmlgg_selected.csv"
label_dict,mrna_dict,survtime_dict,censor_dict= create_dict_from_csv(csv_file)
file_path = '/home/eason/Pictures/all_datasets/3_ours_mrna/Adj.csv'
df = pd.read_csv(file_path, header=None)
adj_matrix = df.to_numpy()
row_indices, col_indices = np.nonzero(adj_matrix)
edge_index = np.stack((row_indices, col_indices), axis=0)
edge_index = torch.LongTensor(edge_index)
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MyEnsembleNet = fused_net()
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
train_dataset = multimodal_train_dataset(label_dict,mrna_dict,survtime_dict,censor_dict)
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
test_dataset = multimodal_test_dataset(label_dict,mrna_dict,survtime_dict,censor_dict)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
MyEnsembleNet = MyEnsembleNet.to(device)
writer = SummaryWriter()
train_epoch=100
pca_reg = CustomLossWithPCARegularization(alpha=0.1,n_components=32)
margin = 0.2
# --- Strat training --- #
iteration = 0
for epoch in range(train_epoch):
    MyEnsembleNet.train()
    for i, (wsi_img_org,wsi_img_aug,mrna_data,labels,survtime,censor) in enumerate(train_loader):
        iteration += 1
        MyEnsembleNet.zero_grad()
        wsi_img = wsi_img_org.to(device)
        wsi_aug = wsi_img_aug.to(device)
        mrna_data = mrna_data.to(device)
        mrna_data = mrna_data.unsqueeze(2)
        labels_gt = labels.to(device)
        survtime = survtime.to(device)
        censor = censor.to(device)
        edge_index = edge_index.to(device)
        img_anchor, features, hazard_prediction, category_prediction = MyEnsembleNet(wsi_img, mrna_data, edge_index)

        labels_negative = torch.roll(labels_gt, shifts=1, dims=0)
        wsi_negative = torch.roll(wsi_img, shifts=1, dims=0)

        label_different = labels_gt-labels_negative
        label_different = [abs(x) for x in label_different]
        label_different = torch.tensor(label_different)
        if label_different.sum().item() == 0:
            contrastive_loss = torch.tensor(0.0)
        else:
            img_positive, _, _, _ = MyEnsembleNet(wsi_aug, mrna_data, edge_index)
            img_negative, _, _, _ = MyEnsembleNet(wsi_negative, mrna_data, edge_index)
            mask = label_different > 0
            img_anchor_filtered = img_anchor[mask]
            img_positive_filtered = img_positive[mask]
            img_negative_filtered = img_negative[mask]
            '''
            similarity_positive = F.cosine_similarity(img_anchor, img_positive)
            similarity_negative = F.cosine_similarity(img_anchor, img_negative)
            contrastive_loss = torch.relu(similarity_positive - similarity_negative + margin).mean()
            '''
            similarity_positive = F.cosine_similarity(img_anchor, img_positive)**2
            similarity_negative = torch.relu(F.cosine_similarity(img_anchor, img_negative) - margin)**2
            contrastive_loss = (similarity_positive + similarity_negative).mean()

        loss_grade = F.cross_entropy(category_prediction, labels_gt)
        loss_hazard = CoxLoss(survtime, censor, hazard_prediction, device)
        pca_reg_loss = pca_reg(features)
        total_loss = loss_grade + loss_hazard + contrastive_loss + pca_reg_loss

        total_loss.backward()
        G_optimizer.step()
        writer.add_scalars('training', {
            'loss_grade': loss_grade.item(),
            'loss_hazard': loss_hazard.item(),
            'contrastive_loss': contrastive_loss.item(),
            'pca_reg_loss': pca_reg_loss.item(),
            'total_loss': total_loss.item()}, iteration)

    if epoch % 1 == 0:
        with torch.no_grad():
            MyEnsembleNet.eval()
            risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
            true_labels = []
            predicted_probs = []
            for i, (wsi_img,mrna_data,labels,survtime,censor) in enumerate(test_loader):
                wsi_img = wsi_img.to(device)
                grade_gt = labels.to(device)
                mrna_data = mrna_data.to(device)
                mrna_data = mrna_data.unsqueeze(2)
                labels_gt = labels.to(device)
                survtime = survtime.to(device)
                censor = censor.to(device)
                edge_index = edge_index.to(device)

                _,_,hazard_prediction, category_prediction = MyEnsembleNet(wsi_img, mrna_data, edge_index)
                pred_cpu = hazard_prediction.detach().cpu().numpy().reshape(-1)
                censor_cpu = censor.detach().cpu().numpy().reshape(-1)
                survtime_cpu = survtime.detach().cpu().numpy().reshape(-1)
                risk_pred_all = np.concatenate((risk_pred_all, pred_cpu))
                censor_all = np.concatenate((censor_all, censor_cpu))
                survtime_all = np.concatenate((survtime_all, survtime_cpu))

                probs = F.softmax(category_prediction, dim=1)
                predicted_probs.extend(probs.cpu().numpy())
                true_labels.extend(grade_gt.cpu().numpy())
            mlb = MultiLabelBinarizer()
            true_labels_bin = mlb.fit_transform(zip(true_labels))
            micro_ap = average_precision_score(true_labels_bin, predicted_probs, average='micro')
            predicted_labels = np.argmax(predicted_probs, axis=1)
            micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
            micro_auc = roc_auc_score(true_labels_bin, predicted_probs, average='micro')
            cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
            writer.add_scalar('metrics/Micro-AP', micro_ap, epoch)
            writer.add_scalar('metrics/Micro-F1', micro_f1, epoch)
            writer.add_scalar('metrics/Micro-AUC', micro_auc, epoch)
            writer.add_scalar('metrics/C-Index', cindex_test, epoch)
            torch.save(MyEnsembleNet.state_dict(), os.path.join('outputs','epoch'+ str(epoch) + '.pkl'))


