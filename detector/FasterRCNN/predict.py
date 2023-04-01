
import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms

from detector.FasterRCNN.draw_box_utils import draw_objs
from detector.FasterRCNN.network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from detector.FasterRCNN.backbone import resnet50_fpn_backbone, MobileNetV2


class FasterRCNNDetector(object):
    def __init__(self, cfg):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["FasterRCNN"].num_classes
        self.backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
        self.model = FasterRCNN(backbone=self.backbone, num_classes=self.num_classes, rpn_score_thresh=0.5)
        # self.model = create_model(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(cfg["FasterRCNN"].ckpt, map_location='cpu')["model"])
        self.model.to(self.device)

    def predict(self, image_path, trks):

        # class_dict = {'Car':1, 'Cyclist':2, 'Pedestrian':3}
        # category_index = [class_dict[i] for i in trks_label]
        # category_index = torch.from_numpy(np.array(category_index)).cuda(non_blocking=True)

        # load image
        original_img = Image.open(image_path)

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        self.model.eval()  # 进入验证模式
        with torch.no_grad():

            predictions = self.model(img.to(self.device), trks=trks)[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            # plot_img = draw_objs(original_img,
            #                      predict_boxes,
            #                      predict_classes,
            #                      predict_scores,
            #                      category_index=category_index,
            #                      box_thresh=0.5,
            #                      line_thickness=3,
            #                      font='arial.ttf',
            #                      font_size=20)
            # plt.imshow(plot_img)
            # plt.show()
            # # 保存预测的图片结果
            # plot_img.save("test_result.jpg")
            return predict_boxes, predict_scores

