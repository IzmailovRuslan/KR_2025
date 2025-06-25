import base64
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
import pickle


coco_image_dir = 'refcoco2/train2014'
refcoco_anns = 'annotations/refcoco/refs(unc).p'
coco_anns = 'annotations/refcoco/instances.json'

with open(refcoco_anns, 'rb') as file:
  refcoco_data = pickle.load(file)
coco = COCO(coco_anns)

def load_image(image_id):
    image_info = coco.loadImgs(image_id)[0]
    image_path = f"{coco_image_dir}/{image_info['file_name']}"
    image = Image.open(image_path).convert('RGB')
    return image


def load_mask(ref):
    ann_id = ref['ann_id']
    ann = coco.loadAnns(ann_id)[0]
    mask = coco.annToMask(ann)
    return mask


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def draw_bounding_boxes(image, bounding_boxes, outline_color="red", line_width=2):
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)
    return image


def rescale_bounding_boxes(bounding_boxes, original_width, original_height, scaled_width=1000, scaled_height=1000):
    x_scale = original_width / scaled_width
    y_scale = original_height / scaled_height
    rescaled_boxes = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        rescaled_box = [
            xmin * x_scale,
            ymin * y_scale,
            xmax * x_scale,
            ymax * y_scale
        ]
        rescaled_boxes.append(rescaled_box)
    return rescaled_boxes

def get_boxes_centers(bboxes):
    centers = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        centers.append([center_x, center_y])
    return centers

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 40/255, 50/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        import cv2
        contours, _ = cv2.findContours(
            mask,cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        contours = [
            cv2.approxPolyDP(
                contour, epsilon=0.01, closed=True
            ) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image,
            contours,
            -1,
            (1, 1, 1, 0.5),
            thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color='green',
        marker='.',
        s=marker_size,
        edgecolor='white',
        linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color='red',
        marker='.',
        s=marker_size,
        edgecolor='white',
        linewidth=1.25
    )


def show_box(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor='green',
            facecolor=(0, 0, 0, 0),
            lw=2)
        )

def show_masks(
    image,
    masks,
    scores,
    title,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i == 0:
            show_mask(mask, plt.gca(), random_color=False, borders=borders)
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
    if box_coords is not None:
        show_box(box_coords, plt.gca())
    plt.axis('off')
    plt.title(title)
    return plt


def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return [xmin, ymin, xmax, ymax]

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

def compute_iou_for_boxes(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou