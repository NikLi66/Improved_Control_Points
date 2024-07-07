import os, sys
import argparse
import torch
import torch.onnx
import time
import cv2
import numpy as np
from models.network import ImprovedControlPoints
from utils.tps import createThinPlateSplineShapeTransformer


class Dewarper():
    def __init__(self, model_path, device="cuda:0"):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.tps = createThinPlateSplineShapeTransformer((320, 320), fiducial_num=(31, 31), device=self.device)


    def load_model(self, model_path):
        model = ImprovedControlPoints(n_classes=2, num_filter=32, BatchNorm='BN', in_channels=3)
        checkpoint = torch.load(model_path, map_location=self.device)
        model_parameter_dick = {}
        for k in checkpoint['state_dict']:
            model_parameter_dick[k.replace('module.', '')] = checkpoint['state_dict'][k]
        model.load_state_dict(model_parameter_dick)
        model.to(self.device)
        model.eval()

        return model

    def preprocess(self, img):
        img = cv2.resize(img, (992, 992), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img


    def predict(self, img):
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
        return output

    def postprocess(self, outputs, img):
        pred_regress = outputs[0].data.cpu().numpy().transpose(0, 2, 3, 1)
        pred_segment = outputs[1].data.round().int().cpu().numpy()
        dewarp_img = self.flatByfiducial_TPS(pred_regress[0], pred_segment[0], img)
        return dewarp_img

    def flatByfiducial_TPS(self, fiducial_points, segment, perturbed_img):
        '''
        flat_shap controls the output image resolution
        '''
        fiducial_points = fiducial_points / [992, 992]
        flat_shap = segment * [31, 31]
        perturbed_img_ = torch.tensor(perturbed_img.transpose(2,0,1)[None,:])

        fiducial_points_ = (torch.tensor(fiducial_points.transpose(1, 0,2).reshape(-1, 2))[None,:]-0.5)*2

        rectified = self.tps(perturbed_img_.double().to(self.device), fiducial_points_.to(self.device), list(flat_shap))
        flat_img = rectified[0].cpu().numpy().transpose(1,2,0)
        flat_img = flat_img.astype(np.uint8)

        return flat_img

    def process(self, img):
        outputs = self.preprocess(img)
        outputs = self.predict(outputs)
        dewarp_img  = self.postprocess(outputs, img)
        return dewarp_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', required=True, type=str,
                        help='Image file path')

    parser.add_argument('--model_path', required=True, type=str,
                        help='Model path')

    args = parser.parse_args()

    dewarper = Dewarper(args.model_path)
    img = cv2.imread(args.image_file)
    dewarp_img = dewarper.process(img)
    cv2.imwrite('result.jpg', dewarp_img)
