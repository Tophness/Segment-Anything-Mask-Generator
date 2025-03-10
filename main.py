import os, cv2, torch, numpy as np, argparse, keyboard, time, requests, threading, tkinter as tk, yaml
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pystray

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.yaml")
TINY_CONFIG = os.path.join(BASE_DIR, "configs", "sam2.1_hiera_t.yaml")
TINY_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_tiny.pt")
LARGE_CONFIG = os.path.join(BASE_DIR, "configs", "sam2.1_hiera_l.yaml")
LARGE_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_large.pt")

use_gpu = False
skip_existing_files = False
save_transparency_black = False

def load_settings():
    global use_gpu, skip_existing_files, save_transparency_black
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            data = yaml.safe_load(f)
        use_gpu = data.get("use_gpu", torch.cuda.is_available())
        skip_existing_files = data.get("skip_existing_files", True)
        save_transparency_black = data.get("save_transparency_black", False)
    else:
        save_settings()

def save_settings():
    global use_gpu, skip_existing_files, save_transparency_black
    data = {
        "use_gpu": use_gpu,
        "skip_existing_files": skip_existing_files,
        "save_transparency_black": save_transparency_black
    }
    with open(SETTINGS_FILE, "w") as f:
        yaml.safe_dump(data, f)

device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

added_masks = []; removed_masks = []; history = []; locked_points = []
grey_masks = []; grey_locked_points = []; remove_locked_points = []
preview_point = None; orig_image = None; display_image = None; current_preview_mask = None; current_preview_color = None
preview_scale = 1.0; preview_offset_x = 0.0; preview_offset_y = 0.0; manual_filemode = False; fullscreen = False
ai_mode = False; brush_radius = 20; painting = False; current_brush = "white"; paint_cursor = None
current_paint_mask = None; current_paint_type = None; last_update_time = 0
lasso_mode = True; drawing_lasso = False; lasso_points = []; lasso_current_point = None; lasso_type = None
tiny_model = None; large_model = None; tiny_predictor = None; large_predictor = None

def loadmodels():
    global tiny_model, large_model, tiny_predictor, large_predictor
    tiny_model = build_sam2(TINY_CONFIG, TINY_CHECKPOINT, device=device)
    large_model = build_sam2(LARGE_CONFIG, LARGE_CHECKPOINT, device=device)
    tiny_predictor = SAM2ImagePredictor(tiny_model)
    large_predictor = SAM2ImagePredictor(large_model)

def download_checkpoint(url, save_path):
    if os.path.exists(save_path): return
    print(f"Downloading {os.path.basename(save_path)}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url}")

def update_display_throttled():
    global last_update_time
    now = time.time()
    if now - last_update_time > 0.2:
        update_display(); last_update_time = now

def apply_mask_overlay(base, mask, color, alpha=0.75):
    ov = base.copy()
    cm = np.zeros_like(base); cm[:] = color; bm = mask.astype(bool)
    ov[bm] = cv2.addWeighted(base, 1 - alpha, cm, alpha, 0)[bm]
    return ov

def generate_add_preview_mask():
    global current_preview_mask, current_preview_color
    if preview_point is None: return
    pts = np.array([preview_point]); lbls = np.ones(len(pts))
    img_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    tiny_predictor.set_image(img_rgb)
    masks, scores, _ = tiny_predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
    current_preview_mask = masks[np.argmax(scores)].astype(bool)
    current_preview_color = (255, 0, 0)

def generate_remove_preview_mask():
    global current_preview_mask, current_preview_color
    if preview_point is None: return
    pts = np.array([preview_point]); lbls = np.ones(len(pts))
    img_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    large_predictor.set_image(img_rgb)
    masks, scores, _ = large_predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
    current_preview_mask = masks[np.argmax(scores)].astype(bool)
    current_preview_color = (0, 0, 255)

def generate_grey_action():
    global current_preview_mask, grey_locked_points
    if not grey_locked_points: return
    pts = np.array(grey_locked_points); lbls = np.ones(len(pts))
    img_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    large_predictor.set_image(img_rgb)
    masks, scores, _ = large_predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
    grey_masks.append(masks[np.argmax(scores)].astype(bool))
    history.append('grey'); current_preview_mask = None; update_display()

def generate_add_action():
    global current_preview_mask
    if not locked_points:
        current_preview_mask = None; update_display(); return
    pts = np.array(locked_points); lbls = np.ones(len(pts))
    img_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    large_predictor.set_image(img_rgb)
    masks, scores, _ = large_predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
    added_masks.append(masks[np.argmax(scores)].astype(bool)); history.append('add')
    current_preview_mask = None; update_display()

def generate_remove_action():
    global current_preview_mask, remove_locked_points
    if not remove_locked_points: return
    pts = np.array(remove_locked_points); lbls = np.ones(len(pts))
    img_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    large_predictor.set_image(img_rgb)
    masks, scores, _ = large_predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
    removed_masks.append(masks[np.argmax(scores)].astype(bool)); history.append('remove')
    current_preview_mask = None; update_display()

def update_display():
    global display_image, paint_cursor, current_preview_mask, current_preview_color
    temp = orig_image.copy()
    ai_white = np.any(added_masks, axis=0) if added_masks else np.zeros(orig_image.shape[:2], bool)
    ai_removed = np.any(removed_masks, axis=0) if removed_masks else np.zeros(orig_image.shape[:2], bool)
    final = ai_white & ~ai_removed
    temp = apply_mask_overlay(temp, final, (255,255,255), 0.4)
    grey = np.any(grey_masks, axis=0) if grey_masks else np.zeros(orig_image.shape[:2], bool)
    temp = apply_mask_overlay(temp, grey, (128,128,128), 0.75)
    if current_preview_mask is not None and current_preview_color is not None:
        temp = apply_mask_overlay(temp, current_preview_mask, current_preview_color, 0.4)
    if not ai_mode and current_paint_mask is not None:
        col = (255,255,255) if current_paint_type=="white" else (128,128,128) if current_paint_type=="grey" else (0,0,255)
        temp = apply_mask_overlay(temp, current_paint_mask.astype(bool), col, 0.4)
    if lasso_mode and drawing_lasso:
        pts = np.array(lasso_points + ([lasso_current_point] if lasso_current_point is not None else []), np.int32)
        pts = pts.reshape((-1,1,2))
        overlay = temp.copy()
        if lasso_type == "grey":
            fill_color = (128,128,128); poly_alpha = 0.75
        elif lasso_type == "remove":
            fill_color = (0,0,255); poly_alpha = 0.4
        else:
            fill_color = (255,255,255); poly_alpha = 0.4
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, poly_alpha, temp, 1 - poly_alpha, 0, temp)
        outline_color = (0,0,0) if lasso_type=="remove" else (0,0,255)
        cv2.polylines(temp, [pts], True, outline_color, 1)
    if not ai_mode:
        if lasso_mode:
            if drawing_lasso:
                pts = np.array(lasso_points + ([lasso_current_point] if lasso_current_point is not None else []), np.int32)
                pts = pts.reshape((-1, 1, 2))
                overlay = temp.copy()
                cv2.fillPoly(overlay, [pts], (255,255,255))
                cv2.addWeighted(overlay, 0.25, temp, 0.75, 0, temp)
                cv2.polylines(temp, [pts], True, (0,0,255), 1)
        elif paint_cursor is not None:
            col = (255,255,255) if current_brush=="white" else (128,128,128) if current_brush=="grey" else (0,0,255)
            cv2.circle(temp, paint_cursor, brush_radius, col, 2)
    win_rect = cv2.getWindowImageRect("Image")
    win_width, win_height = (win_rect[2], win_rect[3]) if win_rect[2]>0 and win_rect[3]>0 else (1280,720)
    h, w = temp.shape[:2]
    scale_x = (win_width / w) * preview_scale; scale_y = (win_height / h) * preview_scale
    scale = min(scale_x, scale_y)
    scaled_w, scaled_h = int(w*scale), int(h*scale)
    scaled_temp = cv2.resize(temp, (scaled_w, scaled_h))
    canvas = np.zeros((win_height, win_width, 3), dtype=np.uint8)
    paste_x = (win_width - scaled_w) // 2 + int(preview_offset_x)
    paste_y = (win_height - scaled_h) // 2 + int(preview_offset_y)
    if scaled_w > win_width or scaled_h > win_height or paste_x < 0 or paste_y < 0 or paste_x+scaled_w>win_width or paste_y+scaled_h>win_height:
        start_x = max(0, -paste_x); start_y = max(0, -paste_y)
        end_x = min(scaled_w, win_width-paste_x); end_y = min(scaled_h, win_height-paste_y)
        canvas[max(paste_y,0):min(paste_y+scaled_h,win_height),
               max(paste_x,0):min(paste_x+scaled_w,win_width)] = scaled_temp[start_y:end_y, start_x:end_x]
    else:
        canvas[paste_y:paste_y+scaled_h, paste_x:paste_x+scaled_w] = scaled_temp
    display_image = canvas
    cv2.imshow("Image", display_image)

def mouse_callback(event, x, y, flags, param):
    global preview_point, paint_cursor, painting, current_brush, current_paint_mask, current_paint_type
    global preview_scale, preview_offset_x, preview_offset_y, lasso_mode, drawing_lasso, lasso_points, lasso_current_point, lasso_type
    win_rect = cv2.getWindowImageRect("Image")
    win_width, win_height = (win_rect[2], win_rect[3]) if win_rect[2]>0 and win_rect[3]>0 else (1280,720)
    h, w = orig_image.shape[:2]
    scale_x = (win_width / w) * preview_scale; scale_y = (win_height / h) * preview_scale
    scale = min(scale_x, scale_y)
    scaled_w, scaled_h = int(w*scale), int(h*scale)
    paste_x = (win_width - scaled_w)//2 + int(preview_offset_x)
    paste_y = (win_height - scaled_h)//2 + int(preview_offset_y)
    img_x = x - paste_x; img_y = y - paste_y
    orig_x = int(img_x / scale) if scale!=0 else 0; orig_y = int(img_y / scale) if scale!=0 else 0
    if orig_x<0 or orig_x>=w or orig_y<0 or orig_y>=h: return
    if event == cv2.EVENT_MOUSEWHEEL:
        delta = flags; old_scale = preview_scale
        scale_x_old = (win_width / w) * old_scale; scale_y_old = (win_height / h) * old_scale
        scale_old = min(scale_x_old, scale_y_old)
        scaled_w_old = w * scale_old; scaled_h_old = h * scale_old
        old_paste_x = (win_width - scaled_w_old) // 2 + preview_offset_x
        old_paste_y = (win_height - scaled_h_old) // 2 + preview_offset_y
        orig_x = (x - old_paste_x) / scale_old; orig_y = (y - old_paste_y) / scale_old
        preview_scale *= 1.1 if delta > 0 else 0.9
        preview_scale = max(0.1, min(4.0, preview_scale))
        scale_x_new = (win_width / w) * preview_scale; scale_y_new = (win_height / h) * preview_scale
        scale_new = min(scale_x_new, scale_y_new)
        scaled_w_new = w * scale_new; scaled_h_new = h * scale_new
        new_paste_x_center = (win_width - scaled_w_new) // 2; new_paste_y_center = (win_height - scaled_h_new) // 2
        desired_paste_x = x - orig_x * scale_new; desired_paste_y = y - orig_y * scale_new
        preview_offset_x = desired_paste_x - new_paste_x_center; preview_offset_y = desired_paste_y - new_paste_y_center
        max_offset_x = max(0, (scaled_w_new - win_width) // 2); max_offset_y = max(0, (scaled_h_new - win_height) // 2)
        preview_offset_x = np.clip(preview_offset_x, -max_offset_x, max_offset_x)
        preview_offset_y = np.clip(preview_offset_y, -max_offset_y, max_offset_y)
        update_display(); return
    if not ai_mode:
        if lasso_mode:
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                drawing_lasso = True; lasso_points = [(orig_x, orig_y)]; lasso_current_point = (orig_x, orig_y)
                if event == cv2.EVENT_LBUTTONDOWN: lasso_type = "white"
                elif event == cv2.EVENT_MBUTTONDOWN: lasso_type = "grey"
                elif event == cv2.EVENT_RBUTTONDOWN: lasso_type = "remove"
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing_lasso:
                    if ((lasso_type == "white" and flags & cv2.EVENT_FLAG_LBUTTON) or
                        (lasso_type == "grey" and flags & cv2.EVENT_FLAG_MBUTTON) or
                        (lasso_type == "remove" and flags & cv2.EVENT_FLAG_RBUTTON)):
                        if np.linalg.norm(np.array([orig_x, orig_y]) - np.array(lasso_points[-1])) > 5:
                            lasso_points.append((orig_x, orig_y))
                    lasso_current_point = (orig_x, orig_y); update_display_throttled()
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
                if drawing_lasso:
                    pts = np.array(lasso_points, np.int32).reshape((-1, 1, 2))
                    mask = np.zeros(orig_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    if lasso_type == "white":
                        added_masks.append(mask.astype(bool)); history.append('add')
                    elif lasso_type == "grey":
                        grey_masks.append(mask.astype(bool)); history.append('grey')
                    elif lasso_type == "remove":
                        erase_mask = mask.astype(bool)
                        added_masks[:] = [np.bitwise_and(m, ~erase_mask) for m in added_masks]
                        grey_masks[:] = [np.bitwise_and(m, ~erase_mask) for m in grey_masks]; history.append('remove')
                    drawing_lasso = False; lasso_points = []; lasso_current_point = None; lasso_type = None; update_display()
            return
        paint_cursor = (orig_x, orig_y)
        if event == cv2.EVENT_MOUSEMOVE:
            if painting and current_paint_mask is not None:
                update_display_throttled()
                cv2.circle(current_paint_mask, (orig_x, orig_y), brush_radius, 255, -1)
                if current_paint_type=="erase":
                    for mask_list in [added_masks, grey_masks]:
                        if mask_list: mask_list[-1] = np.bitwise_and(mask_list[-1], ~current_paint_mask.astype(bool))
        elif event == cv2.EVENT_LBUTTONDOWN:
            current_paint_type = "white"; current_paint_mask = np.zeros(orig_image.shape[:2], dtype=np.uint8)
            cv2.circle(current_paint_mask, (orig_x, orig_y), brush_radius, 255, -1); painting = True; update_display_throttled()
        elif event == cv2.EVENT_MBUTTONDOWN:
            current_paint_type = "grey"; current_paint_mask = np.zeros(orig_image.shape[:2], dtype=np.uint8)
            cv2.circle(current_paint_mask, (orig_x, orig_y), brush_radius, 255, -1); painting = True; update_display_throttled()
        elif event == cv2.EVENT_RBUTTONDOWN:
            current_paint_type = "erase"; current_paint_mask = np.zeros(orig_image.shape[:2], dtype=np.uint8)
            cv2.circle(current_paint_mask, (orig_x, orig_y), brush_radius, 255, -1); painting = True
            for mask_list in [added_masks, grey_masks]:
                if mask_list: mask_list[-1] = np.bitwise_and(mask_list[-1], ~current_paint_mask.astype(bool))
            update_display_throttled()
        elif event == cv2.EVENT_RBUTTONUP:
            if current_paint_mask is not None:
                erase_mask = current_paint_mask.astype(bool)
                new_added = []
                for mask in added_masks: new_added.append(np.bitwise_and(mask, ~erase_mask))
                added_masks[:] = new_added
                new_grey = []
                for mask in grey_masks: new_grey.append(np.bitwise_and(mask, ~erase_mask))
                grey_masks[:] = new_grey
                current_paint_mask = None; painting = False; update_display_throttled()
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            if current_paint_mask is not None:
                mask = current_paint_mask.astype(bool)
                if current_paint_type=="white": added_masks.append(mask); history.append('add')
                elif current_paint_type=="grey": grey_masks.append(mask); history.append('grey')
                elif current_paint_type=="erase": removed_masks.append(mask); history.append('remove')
                current_paint_mask = None; painting = False; update_display()
        return
    if event == cv2.EVENT_MOUSEMOVE:
        preview_point = (orig_x, orig_y)
        if flags & cv2.EVENT_FLAG_RBUTTON: generate_remove_preview_mask()
        else: generate_add_preview_mask()
        update_display()
    elif event == cv2.EVENT_LBUTTONDOWN:
        if preview_point:
            locked_points.append(preview_point); preview_point = None; generate_add_action()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if preview_point:
            remove_locked_points.append(preview_point); preview_point = None; generate_remove_action()
    elif event == cv2.EVENT_MBUTTONDOWN:
        if preview_point:
            grey_locked_points.append(preview_point); preview_point = None; generate_grey_action()

def create_colored_mask(white, grey):
    h, w = white.shape; mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[white] = [255,255,255,255]
    mask_rgba[np.logical_and(grey, ~white)] = [128,128,128,255]
    if save_transparency_black:
        transparent = mask_rgba[:,:,3] == 0
        mask_rgba[transparent] = [0,0,0,255]
    return mask_rgba

def open_settings():
    root = tk.Tk(); root.title("Settings")
    var_gpu = tk.BooleanVar(value=use_gpu)
    var_skip = tk.BooleanVar(value=skip_existing_files)
    var_save_black = tk.BooleanVar(value=save_transparency_black)
    def update_settings():
        global use_gpu, skip_existing_files, save_transparency_black
        use_gpu = var_gpu.get(); skip_existing_files = var_skip.get(); save_transparency_black = var_save_black.get()
        save_settings()
    tk.Checkbutton(root, text="Use GPU", variable=var_gpu, command=update_settings).pack(anchor='w')
    tk.Checkbutton(root, text="Skip existing files", variable=var_skip, command=update_settings).pack(anchor='w')
    tk.Checkbutton(root, text="Save transparency as black", variable=var_save_black, command=update_settings).pack(anchor='w')
    tk.Button(root, text="Close", command=root.destroy).pack()
    root.mainloop()

def create_image():
    image = Image.new('RGB', (64, 64), color=(0,0,0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([16,16,48,48], fill=(255,255,255))
    return image

def exit_app(icon):
    os._exit(0)

def setup_systray():
    icon = pystray.Icon("SAM2", create_image(), "SAM2", menu=pystray.Menu(
        pystray.MenuItem("Settings", lambda: threading.Thread(target=open_settings).start()),
        pystray.MenuItem("Exit", lambda: exit_app(icon))
    ))
    icon.run()

def start_systray():
    t = threading.Thread(target=setup_systray, daemon=True)
    t.start()

def main(input_folder, output_folder):
    global orig_image, display_image, preview_scale, added_masks, removed_masks, grey_masks, history, locked_points, preview_point, current_preview_mask, current_preview_color
    global ai_mode, brush_radius, painting, current_brush, paint_cursor, current_paint_mask, current_paint_type, manual_filemode, preview_offset_x, preview_offset_y, fullscreen
    global lasso_mode, drawing_lasso, lasso_points, lasso_current_point, grey_locked_points, remove_locked_points
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    if not image_files: print("No valid images found."); return
    existing = {f for f in os.listdir(output_folder)}
    current_index = 0
    while 0 <= current_index < len(image_files):
        filename = image_files[current_index]
        if skip_existing_files and filename in existing and not manual_filemode:
            print(f"Skipping {filename}, mask exists."); current_index += 1; continue
        path = os.path.join(input_folder, filename)
        orig_image = cv2.imread(path)
        if orig_image is None:
            print(f"Failed to load {filename}."); current_index += 1; continue
        h, w = orig_image.shape[:2]
        preview_scale = 1.0; preview_offset_x = 0.0; preview_offset_y = 0.0
        added_masks = []; removed_masks = []; history = []; locked_points = []; grey_masks = []
        grey_locked_points = []; remove_locked_points = []
        if manual_filemode:
            mask_path = os.path.join(output_folder, filename)
            if os.path.exists(mask_path):
                mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_image is not None and mask_image.ndim==3 and mask_image.shape[2]==4:
                    black_bg = (mask_image[:,:,0]==0) & (mask_image[:,:,1]==0) & (mask_image[:,:,2]==0) & (mask_image[:,:,3]==255)
                    mask_image[black_bg] = [0,0,0,0]
                    white_mask = (mask_image[:,:,0]==255) & (mask_image[:,:,1]==255) & (mask_image[:,:,2]==255)
                    grey_mask = (mask_image[:,:,0]==128) & (mask_image[:,:,1]==128) & (mask_image[:,:,2]==128)
                    added_masks = [white_mask.astype(bool)]
                    grey_masks = [grey_mask.astype(bool)]
        preview_point = None; current_preview_mask = None; current_preview_color = None
        display_image = orig_image.copy()
        brush_radius = 20; painting = False; current_brush = "white"; paint_cursor = None
        current_paint_mask = None; current_paint_type = None
        lasso_mode = True; drawing_lasso = False; lasso_points = []; lasso_current_point = None
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 1280, 720)
        if fullscreen: cv2.setWindowProperty("Image", cv2.WINDOW_KEEPRATIO, cv2.WND_PROP_AUTOSIZE)
        else: cv2.setWindowProperty("Image", cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Image", mouse_callback)
        update_display()
        prev_ctrl = False
        while True:
            if keyboard.is_pressed('ctrl'):
                if not prev_ctrl:
                    ai_mode = not ai_mode; current_preview_mask = None; current_preview_color = None
                    if ai_mode:
                        lasso_mode = True; drawing_lasso = False; lasso_points = []; lasso_current_point = None
                prev_ctrl = True
            else: prev_ctrl = False
            k = cv2.waitKeyEx(1)
            if k in (2490368,):
                brush_radius += 1; update_display()
            elif k in (2621440,):
                brush_radius = max(1, brush_radius - 1); update_display()
            elif k == ord('r'):
                added_masks.clear(); removed_masks.clear(); history.clear(); locked_points.clear()
                preview_point = None; current_preview_mask = None; grey_masks.clear(); update_display()
            elif k == ord('s'):
                ai_white = np.any(added_masks, axis=0) if added_masks else np.zeros(orig_image.shape[:2], bool)
                ai_removed = np.any(removed_masks, axis=0) if removed_masks else np.zeros(orig_image.shape[:2], bool)
                white = ai_white & ~ai_removed
                grey = np.any(grey_masks, axis=0) if grey_masks else np.zeros(orig_image.shape[:2], bool)
                mask_rgba = create_colored_mask(white, grey)
                out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
                cv2.imwrite(out_path, cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGRA))
                print(f"Saved mask: {out_path}"); current_index += 1; break
            elif k == 2424832:
                manual_filemode = True; current_index = max(0, current_index - 1); break
            elif k == 2555904:
                manual_filemode = True; current_index = min(len(image_files)-1, current_index+1); break
            elif k == ord('u'):
                if history:
                    last = history.pop()
                    if last=='add' and added_masks:
                        added_masks.pop(); 
                        if locked_points: locked_points.pop()
                    elif last=='remove' and removed_masks:
                        removed_masks.pop()
                    elif last=='grey' and grey_masks:
                        grey_masks.pop()
                    update_display()
            elif k == ord('l'):
                paint_cursor = None; lasso_mode = not lasso_mode
                if not lasso_mode and drawing_lasso:
                    drawing_lasso = False; lasso_points = []; lasso_current_point = None
                update_display()
            elif k == 27:
                if fullscreen:
                    cv2.setWindowProperty("Image", cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL); fullscreen = False
                else:
                    cv2.setWindowProperty("Image", cv2.WINDOW_KEEPRATIO, cv2.WND_PROP_AUTOSIZE); fullscreen = True
            update_display()
        cv2.destroyWindow("Image")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM2 Dual-Model Interactive Masking")
    parser.add_argument("--input_folder", type=str, default="folderA")
    parser.add_argument("--output_folder", type=str, default="folderB")
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    load_settings()
    base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    checkpoints = {
        "sam2.1_hiera_tiny.pt": f"{base_url}/sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_large.pt": f"{base_url}/sam2.1_hiera_large.pt"
    }
    for filename, url in checkpoints.items():
        download_checkpoint(url, os.path.join("checkpoints", filename))
    start_systray()
    loadmodels()
    main(args.input_folder, args.output_folder)