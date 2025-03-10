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
python main.py
```
You can also provide input and output folders to automatically launch in manual / SAM 2.1 mask editing mode using those folders:
```sh
python main.py --input_folder "folderpath" --output_folder "folderpath"
```


## Screenshots
Selecting a hand in AI mode

![seg1](https://github.com/user-attachments/assets/38e606a9-6fe4-4fa7-bd25-834f47f48a18)

Drawing a white mask in lasso mode

![seg2](https://github.com/user-attachments/assets/c33d7536-530b-4e98-a844-94fecef11b22)

Removing part of a mask in lasso mode

![seg3](https://github.com/user-attachments/assets/f4acd5f6-2838-4a87-ba0a-5019c4e25f80)

Drawing a grey mask in paintbrush mode

![seg4](https://github.com/user-attachments/assets/1583447b-f373-4b9e-a0e3-71e057a1d190)


## License

This project is released under the MIT License.
