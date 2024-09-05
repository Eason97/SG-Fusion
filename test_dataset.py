from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
import torch

class multimodal_test_dataset(Dataset):
    def __init__(self,label_dict,mrna_dict,survtime_dict,censor_dict):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test=[]
        for line in open('change to your file'):
            line = line.strip('\n')
            if line!='':
                self.list_test.append(line)
        self.root_img='change to your file folder'
        self.dict_label=label_dict
        self.dict_mrna=mrna_dict
        self.dict_survtime=survtime_dict
        self.dict_censor=censor_dict
        self.file_len = len(self.list_test)

    def __getitem__(self, index):
        wsi_img = Image.open(self.root_img + self.list_test[index])
        wsi_img = self.transform(wsi_img)
        # get label
        gene_prefix=self.list_test[index][0:12]
        mrna_data=torch.tensor(self.dict_mrna[gene_prefix]).type(torch.FloatTensor)
        labels = torch.tensor(float(self.dict_label[gene_prefix]) - 2).type(torch.LongTensor)
        survtime = torch.tensor(float(self.dict_survtime[gene_prefix])).type(torch.LongTensor)
        censor = torch.tensor(float(self.dict_censor[gene_prefix])).type(torch.LongTensor)
        return wsi_img,mrna_data,labels,survtime,censor

    def __len__(self):
        return self.file_len
