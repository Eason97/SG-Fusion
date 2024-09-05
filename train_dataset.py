from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from random import randrange
import torch
import random

def data_aug(images):
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
    strong_aug = images
    if random.random() < 0.8:
        strong_aug = color_jitter(strong_aug)
    strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)
    if random.random() < 0.5:
        strong_aug = blurring_image(strong_aug)
    return strong_aug

def rotate(img,rotate_index):
    if rotate_index == 0:
        return img
    if rotate_index==1:
        return img.rotate(90)
    if rotate_index==2:
        return img.rotate(180)
    if rotate_index==3:
        return img.rotate(270)
    if rotate_index==4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)



class multimodal_train_dataset(Dataset):
    def __init__(self,label_dict,mrna_dict,survtime_dict,censor_dict):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train=[]
        for line in open('change to your file root'):
            line = line.strip('\n')
            if line!='':
                self.list_train.append(line)
        self.root_img='change to your file folder'
        self.dict_label=label_dict
        self.dict_mrna=mrna_dict
        self.dict_survtime=survtime_dict
        self.dict_censor=censor_dict
        self.file_len = len(self.list_train)


    def __getitem__(self, index):
        wsi_img_org = Image.open(self.root_img + self.list_train[index])
        i, j, h, w = transforms.RandomCrop.get_params(wsi_img_org, output_size=(512,512))
        wsi_img_org= TF.crop(wsi_img_org, i, j, h, w)
        rotate_index = randrange(0, 8)
        wsi_img_aug = rotate(wsi_img_org, rotate_index)
        wsi_img_aug = data_aug(wsi_img_aug)
        wsi_img_org = self.transform(wsi_img_org)
        wsi_img_aug = self.transform(wsi_img_aug)
        gene_prefix=self.list_train[index][0:12]
        mrna_data=torch.tensor(self.dict_mrna[gene_prefix]).type(torch.FloatTensor)
        labels = torch.tensor(float(self.dict_label[gene_prefix]) - 2).type(torch.LongTensor)
        survtime = torch.tensor(float(self.dict_survtime[gene_prefix])).type(torch.LongTensor)
        censor = torch.tensor(float(self.dict_censor[gene_prefix])).type(torch.LongTensor)
        return wsi_img_org,wsi_img_aug,mrna_data,labels,survtime,censor

    def __len__(self):
        return self.file_len
