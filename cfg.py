import numpy as np
import torch
import random
import cv2

class cfg():

    labes_path="/home/Newdisk2/zhangximin/fca_data_12500/phy_attack_30/phy_attack_300/labels/data"#所有图片的真实框
    images_parameter="/home/Newdisk2/zhangximin/fca_data_12500/phy_attack_30/phy_attack_300/train/data"#图片的渲染参数
    model_obj="/home/Newdisk2/zhangximin/FCA_run_v3/Full/src/car_assets/audi_et_te.obj"
    model_face="/home/Newdisk2/zhangximin/FCA_run_v3/Full/src/car_assets/exterior_face.txt"
    texture_size=6
    save_textures_path="ours_v5_paper.npy"   #保存纹理的位置
    load_textures_flag = False#True#是否读取纹理
    load_textures_path = "gf_11_154_color_smooth.npy"#读取纹理的位置
    save_image_flag=True#是否保存图片
    save_image_path="./images/"#保存图片的文件路径
    yolov5s_parameter="yolov5x_new.pt"#yolov5的权重路径
    epoch=30
    image_path="/home/Newdisk2/zhangximin/fca_data_12500/phy_attack_30/phy_attack_300/images"#山地城市的图片路径
    pre_conf=0.3
    loss_yolo_weight=0.00001
    loss_color_weight=0.1
    loss_smooth_weight=0.0001
    optimizer_learn=0.1


    def nps(self,adv):
        adv1 = adv[0].view(-1, 3)
        color = torch.tensor(
            [[255, 174, 201],[249,181,187],[255,128,64],[0,0,128]]) / 255
        color = color.cuda()
        colorlist = []
        for i in range(len(color)):
            colorlist.append(color[i].unsqueeze(0).repeat_interleave(adv1.shape[0], dim=0))

        T = torch.abs(adv1 - colorlist[0]).sum(dim=1)
        for i in range(1, len(colorlist)):
            T = torch.min(T, torch.abs(adv1 - colorlist[i]).sum(dim=1))
        return T.sum()
    def rgb_to_hsv(self,img, k, eps=1e-8):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value * k], dim=1)
        return hsv

    def hsv_to_rgb(self,hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb

    def atanh(self,data):
        return torch.atanh_((data * 2) - 1)
    def EOT(self,adv_img,img):
        hsv_v = random.uniform(0.1, 1.8)
        hsv_img =self. rgb_to_hsv(adv_img, hsv_v)
        adv_hsv_img = self.hsv_to_rgb(hsv_img)
        noise_k = random.uniform(0.01, 0.2)
        noise = torch.rand(size=img.shape).cuda()
        noise = noise_k * (noise / noise.max())

        s_size = random.randint(8, 40)

        EOT_adv_img = torch.nn.functional.interpolate((adv_hsv_img + noise).clip(0, 1), size=(s_size * 32, s_size * 32))
        img = torch.nn.functional.interpolate(img, size=(s_size * 32, s_size * 32))

        return EOT_adv_img,img





    def zhuanhuan(self,adv):
        a = adv.view(1, adv.shape[1], -1, 3)
        r = a[:, :, :, [0]]
        g = a[:, :, :, [1]]
        b = a[:, :, :, [2]]
        r = r.mean(dim=2, keepdim=True).repeat_interleave(a.shape[2], dim=2)
        g = g.mean(dim=2, keepdim=True).repeat_interleave(a.shape[2], dim=2)
        b = b.mean(dim=2, keepdim=True).repeat_interleave(a.shape[2], dim=2)
        a[:, :, :, [0]] = r
        a[:, :, :, [1]] = g
        a[:, :, :, [2]] = b
        return a.view(size=adv.shape)

    def nps(self,adv):
        adv1 = adv[0].view(-1, 3)
        color = torch.tensor(
            [[129, 127, 38], [24, 62, 12], [24, 63, 63],
             [117, 249, 77], [117, 250, 97],
              ]) / 255
        color = color.cuda()
        colorlist = []
        for i in range(len(color)):
            colorlist.append(color[i].unsqueeze(0).repeat_interleave(adv1.shape[0], dim=0))

        T = torch.abs(adv1 - colorlist[0]).sum(dim=1)
        for i in range(1, len(colorlist)):
            T = torch.min(T, torch.abs(adv1 - colorlist[i]).sum(dim=1))
        return T.sum()

    def smooth(self,img):
        mask1 = (img[:, :, 1:, :-1] != 0) * 1
        mask2 = (img[:, :, :-1, :-1] != 0) * 1
        maska = (mask1 == mask2) * 1
        mask3 = (img[:, :, :-1, 1:] != 0) * 1
        mask4 = (img[:, :, :-1, :-1] != 0) * 1
        maskb = (mask3 == mask4) * 1
        s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2) * maska
        s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2) * maskb

        return torch.sum(s1 + s2)
    # def yolo_pre(self,pre):
    #     # print(type(pre))
    #     # pre <class 'torch.Tensor'>
    #     #
    #     # flag <class 'torch.Tensor'>
    #     # print("pre.shape",pre.shape)
    #
    #     # pre
    #     # torch.Size([39375, 85])
    #     # flag
    #     # torch.Size([39375, 8])
    #
    #     flag = ((pre[:, 5] > self.pre_conf) * 1).repeat(85, 1).T
    #     # flag = ((pre[:, 5] > self.pre_conf) * 1).repeat(1, 85 // 8 + 1) #flag.shape torch.Size([1, 433125])
    #     # print("flag.shape", flag.shape)
    #     return (flag*pre)[:,2].sum()

    def yolo_pre_white(self,pre):
        flag = ((pre[:, 5] > self.pre_conf) * 1).repeat(8, 1).T
        return (flag*pre)[:,4].sum()



    def yolo_pre(self, yolov5s_model, eot_adv_img, img):
        img_pre = yolov5s_model(img)[0][0]
        pre = yolov5s_model(eot_adv_img)[0][0]
        flag = ((img_pre[:, 5] > 0.2) * 1).repeat(8, 1).T

        loss_yolo = (flag * pre)[:, 4].sum() * 10000
        return loss_yolo

    def read_image(self,image_path):
        from PIL import Image
        from torchvision import datasets, transforms, models
        img = Image.open(image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(img).unsqueeze(0).contiguous()
        return image

    def save_textures(self,textures, Tpath):

        stextures = textures.detach().cpu().numpy()
        np.save(Tpath, stextures)

    def tanh(self,data):
        return (torch.tanh(data) + 1) / 2






cfg=cfg()
