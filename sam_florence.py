import os
from contextlib import nullcontext
import torch
from PIL import Image
import numpy as np
import supervision as sv
from utils.florence import load_florence_model, run_florence_inference, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.sam import load_sam_image_model, run_sam_inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

def main(input_folder, output_folder, text):
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(input_folder, fname)
        img = Image.open(img_path).convert("RGB")
        with torch.inference_mode():
            ctx = torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if DEVICE.type == "cuda" else nullcontext()
            with ctx:
                _, result = run_florence_inference(
                    model=FLORENCE_MODEL,
                    processor=FLORENCE_PROCESSOR,
                    device=DEVICE,
                    image=img,
                    task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                    text=text
                )
                detections = sv.Detections.from_lmm(lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=img.size)
                detections = run_sam_inference(SAM_IMAGE_MODEL, img, detections)
        mask_array = np.zeros((*img.size[::-1], 4), dtype=np.uint8)
        mask_array[detections.mask.any(axis=0), :3] = 255
        mask_array[detections.mask.any(axis=0), 3] = 255
        if detections.mask is not None and len(detections.mask):
            for m in detections.mask:
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)
                mask_array[m] = 255
        out_fname = os.path.splitext(fname)[0] + ".png"
        Image.fromarray(mask_array, mode="RGBA").save(os.path.join(output_folder, out_fname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM2 Dual-Model Interactive Masking")
    parser.add_argument("--input_folder", type=str, default="folderA")
    parser.add_argument("--output_folder", type=str, default="folderB")
    parser.add_argument("--object", type=str, default="person")
    args = parser.parse_args()
    os.makedirs(output_folder, exist_ok=True)
    main(args.input_folder, args.output_folder)