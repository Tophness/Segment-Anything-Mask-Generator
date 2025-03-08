# Segment Anything 2.1 / Florence 2 Automasker

This Python script allows for interactive image masking using either AI-based segmentation with [Segment Anything 2.1](https://github.com/facebookresearch/segment-anything) or manual painting/lasso selection in a UI.

## Features

- **AI Mode (Segment Anything 2.1)**
  - Hovering over objects previews a mask using the tiny model.
  - Clicking applies a mask using the large model.
- **Manual Masking Mode**
  - **Lasso Selection (Default Mode)**
    - Press `L` to toggle lasso mode.
    - Left click: Lasso a white mask.
    - Middle click: Lasso a grey mask.
    - Right click: Erase mask.
  - **Paintbrush Mode**
    - Left click: Paint a white mask.
    - Middle click: Paint a grey mask.
    - Right click: Erase mask.
- **Zooming**
  - Use the mouse wheel to zoom in and out.
- **Saving**
  - Press `S` to save mask
- **Reset**
  - Press `R` to reset masks
- **Navigation**
  - Press `left` and `right` keys to navigate between files.

## Installation

Ensure you have Python installed, then run the following commands:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Usage

Run the script with:

```sh
python main.py --input_folder "folderpath" --output_folder "folderpath"
```
You can also try automatically masking using Florence 2 to select parts of the image that match a text description of an object and passing that to SAM 2.1 with:
```sh
python sam_florence.py --input_folder "folderpath" --output_folder "folderpath" --object "person"
```
This will allow you to further edit these masks within the main.py UI script, as it will pick up the same autogenerated masks. I'm working on combining both scripts.
For black and white masking (like kohya training scripts use), you can use this:
```sh
python sam_florence_bw.py --input_folder "folderpath" --output_folder "folderpath" --object "person"
```
Ultimately, all masks need transparency filled in black afterward, but it's much easier to edit the masks with transparency first, as you very rarely get everything right automatically.

## License

This project is released under the MIT License.