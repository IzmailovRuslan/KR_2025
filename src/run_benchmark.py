from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
import numpy as np
import sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO
import pickle
import datetime

from utils import compute_iou, compute_iou_for_boxes, load_image, load_mask, mask_to_bbox
from process_image_sam import process_image
from process_image_seg_net import process_image_seg_net


coco_image_dir = 'refcoco/train2014'
refcoco_anns = 'annotations/refcoco/refs(unc).p'
coco_anns = 'annotations/refcoco/instances.json'

with open(refcoco_anns, 'rb') as file:
  refcoco_data = pickle.load(file)
coco = COCO(coco_anns)

subset_refs = [ref for ref in refcoco_data if ref['split'] == 'val']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

quant_config = BitsAndBytesConfig(load_in_4bit=True)

def run_inference(subset_refs, model, processor, predictor):
    ious = []
    box_ious = []
    errors = 0
    start = datetime.datetime.now()
                    
    for i, ref in enumerate(subset_refs):
        image_id = ref['image_id']
        query = ref['sentences'][0]['sent']
        gt_mask = load_mask(ref)
                    
        input_img = load_image(image_id)
                    
        text_input = f"Find a bounding box for {query}."
        print(i, query)

        try:
            if isinstance(predictor, SAM2ImagePredictor):
                model_output_text, scaled_boxes, annotated_image, masks = process_image(input_img, text_input, model, processor, predictor)
            else:
                model_output_text, scaled_boxes, annotated_image, masks = process_image_seg_net(input_img, text_input, model, processor, predictor)

            iou = compute_iou(masks[0], gt_mask)
            ious.append(iou)
                        
            gt_box = mask_to_bbox(gt_mask)
            box_iou = compute_iou_for_boxes(gt_box, scaled_boxes[0])
            box_ious.append(box_iou)
            print(iou, box_iou)
        except Exception as e:
            print(f"Error: {e}")
            ious.append(0)
            box_ious.append(0)
            errors += 1
                    
    end = datetime.datetime.now()
    total_time = end-start
    return ious,box_ious,errors,input_img,model_output_text,scaled_boxes,annotated_image,masks,total_time


for model_name in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]:
    if model_name ==  "Qwen/Qwen2-VL-2B-Instruct":
        quantizations = [quant_config, None]
    else:
        quantizations = [quant_config]
    
    for q in quantizations:
        if q is not None:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                offload_folder='offload',    
                quantization_config=q
            ).eval()
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                offload_folder='offload',    
            ).eval().cuda()
        
        processor = AutoProcessor.from_pretrained(model_name, location=device)
        
        for predictor_name in ["SAM", "Segmentation Net"]:
            if predictor_name == "SAM":
                for predictor_size in ["tiny", "small", "large"]:
                    predictor = SAM2ImagePredictor.from_pretrained(f'facebook/sam2.1-hiera-{predictor_size}', location=device)
                    torch.cuda.empty_cache()

                    ious, box_ious, errors, input_img, model_output_text, scaled_boxes, annotated_image, masks, total_time = run_inference(subset_refs, model, processor, predictor)

                    print(model_name, predictor_name, predictor_size, q)
                    print(torch.cuda.memory_reserved(0),"GB", torch.cuda.memory_allocated(0), "GB")
                    print(f"Mean IoU {np.mean(ious)}, mean box IoU {np.mean(box_ious)}, error_rate {errors/len(subset_refs)*100}%, mean time per sample {(total_time)/len(subset_refs)}")
                    del predictor, input_img, model_output_text, scaled_boxes, annotated_image, masks
                    torch.cuda.empty_cache()  

            else:
                for predictor_size in ["base"]:
                    predictor = smp.Unet("resnet18", encoder_weights="imagenet", classes=1)
                    predictor = predictor.eval().to(device)
                    torch.cuda.empty_cache()

                    ious, box_ious, errors, input_img, model_output_text, scaled_boxes, annotated_image, masks, total_time = run_inference(subset_refs, model, processor, predictor)

                    print(model_name, predictor_name, predictor_size, q)
                    print(torch.cuda.memory_reserved(0),"GB", torch.cuda.memory_allocated(0), "GB")
                    print(f"Mean IoU {np.mean(ious)}, mean box IoU {np.mean(box_ious)}, error_rate {errors/len(subset_refs)*100}%, mean time per sample {(total_time)/len(subset_refs)}")
                    del predictor, input_img, model_output_text, scaled_boxes, annotated_image, masks
                    torch.cuda.empty_cache()  
