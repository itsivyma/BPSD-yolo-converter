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
- Reject malformed, oversized, or duplicate uploaded files.
- Sanitize Excel worksheet names and filename-like text.
- Experimentally align scan-page YOLO symbols with BPSD MusicXML and
  note annotations for offline QA.

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

Each file can be up to 5 MB for JSON/TXT or 50 MB for
images. At most 100 TXT files and 100 images can be processed
in one run. Duplicate TXT filenames and duplicate image stems
are rejected instead of being overwritten silently.

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

Page filenames must end with two numeric parts in this form:

```text
<sonata name>-<movement number>-<page number>
```

The sonata name itself may contain hyphens. For example,
`Composer-Name_Op090-01-06.txt` belongs to
`Composer-Name_Op090.xlsx`.

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

## Experimental MusicXML alignment

The alignment tools are a separate offline research workflow. They do
not change the five-column output of the Streamlit application.

The workflow uses external copies of:

- a scanned score page and its YOLO TXT file;
- the matching `notes.json` class mapping;
- the matching BPSD MusicXML file;
- BPSD note annotations when note IDs are required; and
- a clean rendered score page for cross-system visual comparison.

`bps_xml_alignment.py` parses MusicXML timing and creates page-level
alignment candidates. The current implementation includes pickup-aware
measure timing, note and chord endpoint handling, and BPSD note-ID
attachment when the required source annotation is available.

The supporting slur tools create visual QA material:

- `slur_endpoint_check.py`: inspect one MusicXML slur's endpoint notes;
- `scan_only_slur_check.py`: inspect a slur visible in the scan but not
  matched to MusicXML;
- `cross_system_slur_check.py`: inspect a slur that crosses a system
  break;
- `slur_batch_candidates.py`: produce ranked endpoint candidates; and
- `slur_batch_endpoint_sheet.py`: generate batch review sheets.

Candidate or scan-only results are not treated as confirmed annotations.
Uncertain fields remain blank until they can be supported by the source
data or human review.

Example:

```bash
python bps_xml_alignment.py \
  --image /path/to/page.jpeg \
  --yolo /path/to/page.txt \
  --notes-json /path/to/notes.json \
  --xml /path/to/score.xml \
  --bps-notes /path/to/ann_score_note.csv \
  --output-dir /path/to/output \
  --all-symbols
```

Run `python bps_xml_alignment.py --help` for the complete set of
options. Dataset files, human-review CSV files, and generated QA images
remain local and are not committed.

## Project structure

```text
.
├── app.py
├── bps_xml_alignment.py
├── converter.py
├── cross_system_slur_check.py
├── requirements.txt
├── scan_only_slur_check.py
├── slur_batch_candidates.py
├── slur_batch_endpoint_sheet.py
├── slur_endpoint_check.py
└── tests
    ├── test_bps_xml_alignment.py
    ├── test_converter.py
    └── test_slur_batch_candidates.py
```

## Data and privacy

This repository does not include score images, annotations, exported CSV files, Excel workbooks, or other dataset files.

Uploaded files are processed by the locally running Streamlit application.

## License

No software license has been selected yet. Until a license is
added, copyright law reserves reuse, modification, and
redistribution rights to the copyright holder.

## Current scope

The Streamlit application covers the first BPS-OMR annotation stage:

```text
class_id, x, y, w, h
```

The offline research scripts can generate MusicXML-based alignment
candidates for selected symbol classes. They do not yet provide a fully
automatic, fully reviewed export of every semantic annotation field.

## Disclaimer

This is an independent annotation-conversion and QA utility. Dataset files must be obtained and used according to their original licenses and terms.
