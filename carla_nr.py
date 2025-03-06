import neural_renderer as nr
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import math
import random

class carla_nr(torch.nn.Module):
    def __init__(self,obj_path,face=None,texture_size=6,load_texture=True):
        super(carla_nr,self).__init__()
        #读取模型和可渲染面
        vertices, faces,texture_origin = nr.load_obj(filename_obj=obj_path, texture_size=texture_size, load_texture=load_texture)

        texture_mask = torch.zeros(faces.shape[0], texture_size, texture_size, texture_size, 3)
        list1=[]
        with open(face,"r") as f:
            faces_id = f.readlines()
            for face_id in faces_id:
                if face_id != '\n':
                    list1.append(int(face_id) - 1)
                    # texture_mask[int(face_id) - 1, :, :, :, :] = 1

        random.seed(0)
        random.shuffle(list1)
        for h in range(int(len(list1))):
            texture_mask[list1[h], :, :, :, :] = 1
        self.texture_mask=texture_mask[None,:,:,:,:,:].to('cuda')
        self.vertices = vertices[None, :, :].to('cuda')
        self.faces = faces[None, :, :].to('cuda')
        self.texture_origin=texture_origin[None,:,:,:,:,:].to('cuda')
    #转换角度
    def get_params(self,carlaTcam, carlaTveh):  # carlaTcam: tuple of 2*3
        scale = 0.40
        # scale = 0.38
        # calc eye
        eye = [0, 0, 0]
        for i in range(0, 3):
            # eye[i] = (carlaTcam[0][i] - carlaTveh[0][i]) * scale
            eye[i] = carlaTcam[0][i] * scale

        # calc camera_direction and camera_up
        # 欧拉角
        pitch = math.radians(carlaTcam[1][0])
        yaw = math.radians(carlaTcam[1][1])
        roll = math.radians(carlaTcam[1][2])

        cam_direct = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]  # 相机在相机坐标系的方向
        cam_up = [math.cos(math.pi / 2 + pitch) * math.cos(yaw), math.cos(math.pi / 2 + pitch) * math.sin(yaw),
                  # 相机顶部在相机坐标系的方向
                  math.sin(math.pi / 2 + pitch)]

        # 如果物体也有旋转，则需要调整相机位置和角度，和物体旋转方式一致
        # 先实现最简单的绕Z轴旋转
        p_cam = eye
        p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]  # 相机在世界坐标系的方向
        p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]  # 相机顶部在世界坐标系的方向
        p_l = [p_cam, p_dir, p_up]

        # 绕z轴
        trans_p = []
        for p in p_l:
            if math.sqrt(p[0] ** 2 + p[1] ** 2) == 0:
                cosfi = 0
                sinfi = 0
            else:
                cosfi = p[0] / math.sqrt(p[0] ** 2 + p[1] ** 2)
                sinfi = p[1] / math.sqrt(p[0] ** 2 + p[1] ** 2)
            cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + sinfi * math.sin(math.radians(carlaTveh[1][1]))
            sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - math.sin(math.radians(carlaTveh[1][1])) * cosfi
            trans_p.append([math.sqrt(p[0] ** 2 + p[1] ** 2) * cossum, math.sqrt(p[0] ** 2 + p[1] ** 2) * sinsum, p[2]])

        # 绕x轴
        trans_p2 = []
        for p in trans_p:
            if math.sqrt(p[1] ** 2 + p[2] ** 2) == 0:
                cosfi = 0
                sinfi = 0
            else:
                cosfi = p[1] / math.sqrt(p[1] ** 2 + p[2] ** 2)
                sinfi = p[2] / math.sqrt(p[1] ** 2 + p[2] ** 2)
            cossum = cosfi * math.cos(math.radians(carlaTveh[1][2])) + sinfi * math.sin(math.radians(carlaTveh[1][2]))
            sinsum = math.cos(math.radians(carlaTveh[1][2])) * sinfi - math.sin(math.radians(carlaTveh[1][2])) * cosfi
            trans_p2.append(
                [p[0], math.sqrt(p[1] ** 2 + p[2] ** 2) * cossum, math.sqrt(p[1] ** 2 + p[2] ** 2) * sinsum])

        # 绕y轴
        trans_p3 = []
        for p in trans_p2:
            if math.sqrt(p[0] ** 2 + p[2] ** 2) == 0:
                cosfi = 0
                sinfi = 0
            else:
                cosfi = p[0] / math.sqrt(p[0] ** 2 + p[2] ** 2)
                sinfi = p[2] / math.sqrt(p[0] ** 2 + p[2] ** 2)
            cossum = cosfi * math.cos(math.radians(carlaTveh[1][0])) + sinfi * math.sin(math.radians(carlaTveh[1][0]))
            sinsum = math.cos(math.radians(carlaTveh[1][0])) * sinfi - math.sin(math.radians(carlaTveh[1][0])) * cosfi
            trans_p3.append(
                [math.sqrt(p[0] ** 2 + p[2] ** 2) * cossum, p[1], math.sqrt(p[0] ** 2 + p[2] ** 2) * sinsum])

        trans_p = trans_p3
        return trans_p[0], \
            [trans_p[1][0] - trans_p[0][0], trans_p[1][1] - trans_p[0][1], trans_p[1][2] - trans_p[0][2]], \
            [trans_p[2][0] - trans_p[0][0], trans_p[2][1] - trans_p[0][1], trans_p[2][2] - trans_p[0][2]]

    def combinimg(self,img,reimg,boxes):
        t = reimg[0][0]
        wt = t.sum(dim=0)
        ht = t.sum(dim=1)
        wl = 0
        wr = len(wt) - 1
        hl = 0
        hr = len(ht) - 1
        for i in wt:
            if i != 0:
                break
            wl += 1
        for i in reversed(wt):
            if i != 0:
                break
            wr -= 1
        for i in ht:
            if i != 0:
                break
            hl += 1
        for i in reversed(ht):
            if i != 0:
                break
            hr -= 1

        lx = int(boxes[0]) - (wr - wl) // 2
        ly = int(boxes[1]) - (hr - hl) // 2
        rx = int(boxes[0]) + (wr - wl) // 2
        ry = int(boxes[1]) + (hr - hl) // 2

        rx += (wr - wl) - (rx - lx)
        ry += (hr - hl) - (ry - ly)
        if rx > t.shape[1]:
            lx -= rx - t.shape[1]
            rx = t.shape[1]
        if ry > t.shape[0]:
            ly -= ry - t.shape[0]
            ry = t.shape[0]
        if lx < 0:
            rx += abs(lx)
            lx = 0
        if ly < 0:
            ry += abs(ly)
            ly = 0

        # img[:, :, ly:ry, lx:rx] = torch.where(reimg[:, :, hl:hr, wl:wr] == 1,
        #                                       reimg[:, :, hl:hr, wl:wr] + img[:, :, ly:ry, lx:rx],
        #                                       img[:, :, ly:ry, lx:rx])
        # img[:, :, ly:ry, lx:rx] = torch.where(reimg[:, :, hl:hr, wl:wr] == 1,
        #                                       img[:, :, ly:ry, lx:rx],
        #                                       reimg[:, :, ly:ry, lx:rx])

        img[:, :, ly:ry, lx:rx] = reimg[:, :, hl:hr, wl:wr] + img[:, :, ly:ry, lx:rx] * (
                (reimg[:, :, hl:hr, wl:wr] == 0) * 1)
        return img
    def get_face(self):
        return self.faces
    def get_vertices(self):
        return self.vertices
    def get_start_textures(self):
        
        return self.texture_origin*self.texture_mask
    def forward(self,img,cam_trans,veh_trans,adv_textures,gbox,image_size,create_wl=False):
        # print("mean",torch.mean(adv_textures).item())
        if create_wl==False:
            # tensor = tensor.to('cuda')
            textures=self.texture_origin * (1 - self.texture_mask) + adv_textures * self.texture_mask
        eye, camera_direction, camera_up = self.get_params(cam_trans, veh_trans)
        render = nr.Renderer(camera_mode='look', image_size=image_size)
        render.eye = eye
        render.camera_direction = camera_direction
        render.camera_up = camera_up
        render.background_color = [0, 0, 0]
        render.viewing_angle = 45
        render.light_direction = [0, 0, 1]

        images1, _, _ = render(self.vertices, self.faces, textures)
        images1.to('cuda')
        render.background_color = [1, 1, 1]
        images2, _, _ = render(self.vertices, self.faces, textures)
        images2.to('cuda')


        combin_img=self.combinimg(img,images1,gbox)
        return combin_img,images1,images2