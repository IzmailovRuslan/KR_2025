from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import re
import numpy as np
from utils import get_boxes_centers, image_to_base64, rescale_bounding_boxes, show_masks


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(image, text_input, model, processor, predictor):
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

    if isinstance(image, Image.Image):
        image = np.array(image)

    predictor.set_image(image)
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    fig = show_masks(
        image,
        [masks[0]],
        [scores[0]],
        point_coords=input_points,
        input_labels=input_labels,
        borders=True,
        box_coords=scaled_boxes,
        title=text_input,
    )

    return output_text, scaled_boxes, fig, masks