import os

import cv2
import numpy as np
import torch
from skimage import io
from skimage.transform import resize
from torchvision import transforms

from model import Unet_like, deeper_Unet_like

IMAGES_PATH = "./images"
OUTPUT_PATH = "./outputs"
IMAGE_SIZE = (512, 512)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
THRESHOLD_MIN = 127
THRESHOLD_MAX = 255

MODELS_PATH = "./models"
LARGER_MODEL = "colorShifts_deeper_zoomedOut_200epochs.pth"
SMALLER_MODEL = "less_dataAug_130epochs.pth"
MODEL_NAME = LARGER_MODEL  # or, SMALLER_MODEL


def tensor_to_image(tensor, inv_trans=True):
    if inv_trans:
        for t, m, s in zip(tensor, IMAGE_MEAN, IMAGE_STD):
            t.mul_(s).add_(m)
    image = tensor.cpu().numpy()
    image *= 255
    image = image.astype(np.uint8)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)
    return image


def image_to_tensor(image):
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    tensor = img_transform(image)
    tensor = torch.unsqueeze(tensor, 0).float()
    return tensor


def init_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 127
    params.maxThreshold = 129
    params.thresholdStep = 1
    params.filterByArea = True
    params.minArea = 10
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minDistBetweenBlobs = 1
    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def extract_blobs(out, detector):
    out = 1 - out
    out = cv2.threshold(out, THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)[1]
    keypoints = detector.detect(out)
    return keypoints


if __name__ == "__main__":
    # ensure output folder exists
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # select model class based on name
    if MODEL_NAME == SMALLER_MODEL:
        model = Unet_like()
    else:
        model = deeper_Unet_like()

    # load model for inference
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # initialise blob detector
    blob_detector = init_blob_detector()

    # for each image inside target folder
    for root, dirs, files in os.walk(IMAGES_PATH):
        for i, file in enumerate(files):
            print(f"Processing image {i}...")

            # load and prepare image
            img = io.imread(os.path.join(root, file))
            img = resize(img, IMAGE_SIZE)
            tensor_img = image_to_tensor(img)

            # apply model to image
            with torch.no_grad():
                prediction = model(tensor_img)[0]
            prediction = tensor_to_image(prediction, False)

            # extract keypoints from prediction using blob detector
            keypoints = extract_blobs(prediction, blob_detector)

            # make image human friendly
            img *= 255
            img = img.astype(np.uint8)
            img = img[:, :, ::-1]

            # draw keypoint regions on image
            img = cv2.drawKeypoints(
                img,
                keypoints,
                np.array([]),
                (0, 255, 0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            # save results
            cv2.imwrite(f"{OUTPUT_PATH}/{i}_raw.jpg", prediction)
            cv2.imwrite(f"{OUTPUT_PATH}/{i}_bounds.jpg", img)
