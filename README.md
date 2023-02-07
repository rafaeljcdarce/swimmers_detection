# Swimmers Detection

Fork of [`njacquelin/swimmers_detection`](https://github.com/njacquelin/swimmers_detection) with light refactor of inference script. See original repo for further documentation. Training was not tested, just model inference.

## Set up

Tested on Python 3.8.16

Install requirements inside virtual environment:
```bash
# only do this once
python -m venv venv

# do this every time
source venv/bin/activate

# only do this once
pip install -r requirements.txt
```

## Usage

Put the images you want to run the model on inside a folder called `images` and execute:
```bash
python swimmer_tracking.py
```
Results (raw model output and original image with bounding circles) will be saved to a folder called `outputs`.

The original author provided two pretrained models, a smaller one and a larger one, which you can switch between by changing the `MODEL_NAME` global variable at the top of the script. There, you can also tweak some other variables (e.g. thresholds for blob detection).
