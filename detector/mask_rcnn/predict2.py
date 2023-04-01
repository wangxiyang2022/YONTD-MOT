from PIL import Image
import torch
from torchvision import transforms

from detector.mask_rcnn.network_files import MaskRCNN
from detector.mask_rcnn.backbone import resnet50_fpn_backbone


class MaskRCNNDetector(object):
    def __init__(self, cfg):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["MaskRCNN"].num_classes
        self.backbone = resnet50_fpn_backbone()
        self.model = MaskRCNN(self.backbone, num_classes=self.num_classes, rpn_score_thresh=0.5, box_score_thresh=0.5)
        self.weights_dict = torch.load(cfg["MaskRCNN"].ckpt, map_location='cpu')
        self.weights_dict = self.weights_dict["model"] if "model" in self.weights_dict else self.weights_dict
        self.model.load_state_dict(self.weights_dict)
        self.model.to(self.device)

    def predict(self, image_path, trks):
        original_img = Image.open(image_path).convert('RGB')
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        self.model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=self.device)
            self.model(init_img, trks=trks)

            predictions = self.model(img.to(self.device), trks=trks)[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            return predict_boxes, predict_scores

