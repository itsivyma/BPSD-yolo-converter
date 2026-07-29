# BPSD YOLO Converter & QA

A local Streamlit application for validating YOLO bounding-box annotations, converting them to CSV or Excel, and drawing the boxes back onto score images for visual quality assurance.

## Features

- Read Label Studio YOLO TXT files.
- Read `notes.json` class ID mappings.
- Validate bounding-box values and class IDs.
- Match TXT files with images by filename.
- Convert normalized YOLO coordinates to pixel coordinates.
- Draw bounding boxes back onto score images.
- Download a five-column CSV for the selected page.
- Download a QA overlay PNG.
- Group pages from the same sonata into a multi-sheet Excel workbook.
- Run automated coordinate tests with pytest.

## YOLO format

Each annotation row contains five values:

```text
class_id x_center y_center width height
```

Example:

```text
18 0.201429 0.228247 0.022286 0.017574
```

The four bounding-box coordinates are normalized values between `0` and `1`.

## Validation colors

- Red: error
- Dark orange: warning
- Blue: valid bounding box

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

## Input files

The application accepts:

- One `notes.json`
- One or more YOLO `.txt` files
- One or more `.jpg`, `.jpeg`, or `.png` images

TXT files and images are matched by filename stem:

```text
Beethoven_Op090-01-01.txt
Beethoven_Op090-01-01.jpeg
```

## Outputs

### Page CSV

Each selected page can be downloaded as a five-column CSV:

```text
class_id,x,y,w,h
```

### QA PNG

The selected page can be downloaded with its bounding boxes drawn over the original image.

### Sonata Excel workbook

Pages with the same sonata prefix can be downloaded as one `.xlsx` workbook.

Example:

```text
Beethoven_Op090.xlsx
├── Summary
├── Beethoven_Op090-01-01
├── Beethoven_Op090-01-02
└── ...
```

Each page worksheet contains:

```text
class_id | x | y | w | h
```

## Run tests

```bash
python -m pytest -v
```

## Project structure

```text
.
├── app.py
├── converter.py
├── requirements.txt
└── tests
    └── test_converter.py
```

## Data and privacy

This repository does not include score images, annotations, exported CSV files, Excel workbooks, or other dataset files.

Uploaded files are processed by the locally running Streamlit application.

## Current scope

The current version covers the first BPS-OMR annotation stage:

```text
class_id, x, y, w, h
```

It does not yet generate the complete semantic annotation fields such as musical time, measure positions, note IDs, connected notes, or stem direction.

## Disclaimer

This is an independent annotation-conversion and QA utility. Dataset files must be obtained and used according to their original licenses and terms.