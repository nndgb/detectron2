#utils文件，帮助我们设置训练所需的配置参数，并且返回cfg参数，并且utils还包含了最后预测时的输出函数，我们可以对图片进行预测，也可以对视频文件进行预测，也可以调用外部摄像头进行实时预测。
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode

import random
import cv2
import matplotlib.pyplot as plt

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg

def on_Image(image_path, predictor):
    class_names = ["five","four","one","three","two"]
    # colors = ["red",]
    im = cv2.imread(image_path)
    outputs = predictor(im)

    # instance_mode:
    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """

    v = Visualizer(im[:,:,::-1], metadata={'thing_classes':class_names}, scale=0.5, instance_mode = ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(14, 10))
    plt.imshow(v.get_image())
    plt.show()

def on_Video(videoPath, predictor):
    class_names = ["five", "four", "one", "three", "two"]
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        print("Error opening file...")
        return

    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:,:,::-1], metadata={'thing_classes':class_names}, scale=0.5 ,instance_mode = ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        # cv2.imread("Reuslt", output.get_image()[:,:,::-1])
        # cv2.namedWindow("result", 0)
        # cv2.resizeWindow("result", 1200, 600)

        #调用电脑摄像头进行检测
        cv2.namedWindow("result", cv2.WINDOW_FREERATIO) # 设置输出框的大小，参数WINDOW_FREERATIO表示自适应大小
        cv2.imshow("result" , output.get_image()[:,:,::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()

