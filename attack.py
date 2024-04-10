
import torch
import numpy as np
from cfg import cfg
from models.common import DetectMultiBackend
from PIL import Image
import os
import cv2
from torchvision import models
from torchvision import transforms



def find_info(file_path, image_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            columns = line.strip().split()
            if columns[0] == image_name:

                return columns

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1[0],box1[1],box1[2],box1[3]
    x1_, y1_, x2_, y2_ = box2[0],box2[1],box2[2],box2[3]

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou
def attack(cfg):

    rimg=carla_nr(cfg.model_obj,cfg.model_face,texture_size=cfg.texture_size)
    adv_t = np.random.random((1, 23145, 6, 6, 6, 3)).astype('float32')
    adv_t = torch.tensor(adv_t).to('cuda:0')

    if cfg.load_textures_flag:
        adv_t=torch.tensor(np.load(cfg.load_textures_path)).to('cuda:0')
    adv_t.requires_grad=True
    optimizer = torch.optim.Adam([adv_t], lr=cfg.optimizer_learn)

    yolov5s_model=DetectMultiBackend(cfg.yolov5s_parameter,device=torch.device('cuda:0'))

    for epoch in range(cfg.epoch):
        for file in os.listdir(cfg.image_path):
            w,h=0,0
            print("file",file[0:-4])
            file_path = './' + str(
                file[0:-4]) + '.txt'
            columns = find_info(file_path, '2')
            label_rough = [columns[1], columns[2], columns[3], columns[4]]
            w = float(label_rough[0]) * 800
            h = float(label_rough[1]) * 800
            data = np.load('./'+str(file[0:-4])+'.npz', allow_pickle=True)#.item()
            img = cv2.imread(
                './' +str(
                file[0:-4]) + '.png')
            img = img.transpose((2, 0, 1))[::-1] /255 # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to('cuda:0')
            img = img.unsqueeze(0)
            img = img.type(torch.cuda.FloatTensor)
            image_origins = img
            veh_trans = data['veh_trans']
            cam_trans = data['cam_trans']
            box = [w, h, 0, 0]
            adv_img ,original_img,renderer_white= rimg(img, cam_trans, veh_trans, cfg.tanh(adv_t), box, 800)
            img_path = './bg/'+str(file[0:-4])+'.png'
            padded_matrix = cv2.imread(img_path)
            padded_matrix = torch.from_numpy(padded_matrix).to(torch.float16).to('cuda')
            padded_matrix = padded_matrix.permute(2, 0, 1).unsqueeze(0)
            model1 = models.resnet50(pretrained=True)
            model1 = model1.to('cuda:0')
            trans_adv_img_1 = transform(adv_img)
            image_origins = transform(image_origins)
            image_origins = image_origins.type(torch.cuda.FloatTensor)
            image_origins = image_origins.to('cuda:0')
            model1 = model1.to('cuda:0')
            # 获取模型的特征提取部分
            feature_extractor = torch.nn.Sequential(*list(model1.children())[:-1])

            # 使用特征提取器处理输入图像
            intermediate_outputs1 = feature_extractor(image_origins)
            trans_adv_img_1 = trans_adv_img_1.type(torch.cuda.FloatTensor)
            intermediate_outputs2 = feature_extractor(trans_adv_img_1)
            padded_matrix = padded_matrix.type(torch.cuda.FloatTensor).to('cuda:0')
            intermediate_outputs3 = feature_extractor(padded_matrix)
            loss3 = torch.dist(intermediate_outputs1, intermediate_outputs2, p=2)
            loss4 = torch.dist(intermediate_outputs2, intermediate_outputs3, p=2)
            if cfg.save_image_flag:
                os.makedirs(cfg.save_image_path,exist_ok=True)
                cv_img = adv_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite(str(file[0:-4])+".png", cv2.cvtColor(cv_img * 255, cv2.COLOR_RGB2BGR))
            transform = transforms.Compose([
                transforms.RandomResizedCrop((800, 800), scale=(0.7, 1.3)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
            ])

            # 进行随机变换


            pre = yolov5s_model(adv_img)[0][0]

            loss_iou = calculate_iou(yolov5s_model(adv_img)[0][..., :4], label_rough)
            loss_yolo =  (((pre[:, 5] > 0.5) * 1).repeat(8, 1).T)*pre.sum()
            loss_yolo = loss_yolo.to(adv_t.device)

            loss=-loss3+loss4+loss_yolo+loss_iou
            print("epach:{},loss_yolo:{}".format(epoch,loss))
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

if __name__=="__main__":
    attack(cfg)