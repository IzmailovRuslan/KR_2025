from qwen_vl_utils import process_vision_info
import torch
import torchvision.transforms as transforms
import re
import numpy as np


from utils import get_boxes_centers, image_to_base64, rescale_bounding_boxes, show_masks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    transforms.Resize((64,64)),
    ])

def process_image_seg_net(image, text_input, model, processor, predictor):
    system_prompt = "You are a helpfull assistant to detect objects in images. When asked to detect element based on a description you return a list of bounding boxes for the element in the form of [[xmin, ymin, xmax, ymax]] whith the values beeing scaled to 1000 by 1000 pixels. Return xmin, ymin, xmax, ymax in integer format and do not add any additional words to the output."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_to_base64(image)}"},
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": text_input},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    pattern = r'\d+'
    matches = re.findall(pattern, str(output_text))
    parsed_boxes = [[int(match) for match in matches]]
    print(parsed_boxes)
    scaled_boxes = rescale_bounding_boxes(parsed_boxes, image.width, image.height)
    scaled_centres = get_boxes_centers(scaled_boxes)
    
    input_points = np.array(scaled_centres)
    input_labels = np.ones(len(input_points), dtype=np.int32)

    box = [int(t) for t in scaled_boxes[0]]

    with torch.inference_mode():
        img_cropped = (image.crop(box))
        size = list(img_cropped.size)
        pred_mask = predictor(transformations(img_cropped).unsqueeze(0).to(device))
    pred_mask = transforms.Resize((size[1], size[0]))(pred_mask)
    pred_mask = pred_mask.cpu().squeeze(0).squeeze(0).numpy()
    pred_mask = np.where(pred_mask>np.quantile(pred_mask, 0.25), 1, 0)
    final_mask = np.zeros((image.height, image.width))
    final_mask[box[1]:box[3], box[0]:box[2]] = pred_mask
    
    fig = show_masks(
        image,
        [final_mask],
        [1],
        point_coords=input_points,
        input_labels=input_labels,
        borders=True,
        box_coords=scaled_boxes,
        title=text_input,
    )

    return output_text, scaled_boxes, fig, [final_mask]