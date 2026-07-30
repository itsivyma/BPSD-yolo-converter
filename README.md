# BPSD MusicXML–YOLO Alignment & QA

Offline research tools for aligning YOLO symbol boxes on scanned BPSD
score pages with MusicXML events and BPSD note annotations.

The project creates candidate semantic links and visual QA material. It
does not assume that a YOLO box and a MusicXML event share an ID, and it
does not treat geometric proximity as proof. Unsupported or uncertain
BPS-OMR fields remain blank until they can be verified from source data
or human review.

## Current capabilities

- Read a scanned score page, YOLO TXT annotations, and `notes.json`.
- Detect piano systems, staves, and approximate measure boundaries from
  the target scan.
- Parse MusicXML notes, measures, divisions, time signatures, dynamics,
  slurs, ties, voices, and staves.
- Convert MusicXML event positions to pickup-aware BPSD musical time.
- Attach BPSD note IDs when a corresponding note annotation is
  available, including notes that fall within a BPSD tied span.
- Match `dynamicF`, `dynamicP`, and `dynamicS` to MusicXML dynamic
  events in page reading order.
- Preserve `fingering1`–`fingering5` boxes while leaving unsupported
  semantic links blank by default.
- Generate slur endpoint candidates and visual QA sheets, including
  scan-only and cross-system cases.
- Keep confirmed, candidate, unresolved, and scan-only results
  distinguishable during review.

## Evidence and review rules

The input sources contribute different information:

| Source | Information used |
| --- | --- |
| YOLO TXT | Class ID and normalized bounding-box geometry |
| `notes.json` | Class ID to class-name mapping |
| Scan image | Staff, system, barline, and glyph geometry |
| MusicXML | Musical structure, timing, pitch, voice, staff, slur, and tie events |
| BPSD note annotations | BPSD note IDs and note timing |

There is no universal ID shared by YOLO and MusicXML. The tools therefore
produce alignments by combining score structure and geometry, then
expose uncertain cases for review.

The default policy is conservative:

- MusicXML-supported values may be written when the correspondence is
  established.
- Candidate values remain labeled as candidates.
- Unknown fields stay blank rather than being guessed.
- `--infer-fingerings` is optional and non-authoritative because the
  current source MusicXML contains no fingering elements.
- Repeat mapping must be checked before extending the workflow across
  a whole sonata.

## Alignment inputs

The main alignment command uses external copies of:

- a scanned score page;
- the matching YOLO `.txt` file;
- the matching `notes.json`;
- the corresponding BPSD MusicXML file; and
- the corresponding BPSD note annotation CSV.

A clean rendered score page is also used by the cross-system slur QA
tool.

Dataset files are not included in this repository.

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

## Run the alignment

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

Run the following command for all available options:

```bash
python bps_xml_alignment.py --help
```

The alignment command writes a CSV, QA overlays, and a JSON report to
the selected output directory. With `--all-symbols`, unsupported
semantic fields are retained as blank values.

## Slur QA tools

- `slur_endpoint_check.py`: inspect the endpoint notes of one MusicXML
  slur.
- `scan_only_slur_check.py`: inspect a slur visible in the scan but not
  matched to MusicXML.
- `cross_system_slur_check.py`: inspect the two visible segments of a
  slur crossing a system break.
- `slur_batch_candidates.py`: rank endpoint candidates and combine
  earlier human-review decisions.
- `slur_batch_endpoint_sheet.py`: generate batch endpoint review sheets.

Batch results use explicit review states:

- `locked_xml_match`: confirmed MusicXML match.
- `locked_scan_only`: confirmed scan-only slur.
- `high_confidence_candidate`: promising candidate, not yet confirmed.
- `needs_review`: insufficient or conflicting evidence.
- `possible_scan_only`: no sufficiently supported MusicXML match yet.

Only the two `locked_*` states represent previously confirmed review
decisions.

## YOLO format

Each YOLO annotation row contains:

```text
class_id x_center y_center width height
```

Example:

```text
18 0.201429 0.228247 0.022286 0.017574
```

The four bounding-box coordinates are normalized values between `0`
and `1`.

## Separate TXT-to-CSV project

The earlier Streamlit annotation converter is maintained separately on
the
[`txt-to-csv`](https://github.com/itsivyma/BPSD-yolo-converter/tree/txt-to-csv)
branch. The Streamlit application, converter module, dependencies, and
tests are intentionally excluded from `main`.

To work on that project locally:

```bash
git switch txt-to-csv
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Run tests

```bash
python -m pytest -v
```

## Project structure

```text
.
├── bps_xml_alignment.py
├── cross_system_slur_check.py
├── requirements.txt
├── scan_only_slur_check.py
├── slur_batch_candidates.py
├── slur_batch_endpoint_sheet.py
├── slur_endpoint_check.py
└── tests
    ├── test_bps_xml_alignment.py
    └── test_slur_batch_candidates.py
```

## Data and privacy

This repository does not include score images, MusicXML files, YOLO
annotations, BPSD annotations, human-review CSV files, generated QA
images, exported spreadsheets, or other dataset files.

Machine-specific prototypes and input paths are intentionally excluded
from version control.

## Current scope

This is a research alignment and QA workflow, not a finished
whole-dataset converter. Dynamic MusicXML timing is currently supported
for `dynamicF`, `dynamicP`, and `dynamicS`. Fingering links and
unconfirmed slur correspondences require review, and unsupported
BPS-OMR semantic fields remain blank.

## License

No software license has been selected yet. Until a license is added,
copyright law reserves reuse, modification, and redistribution rights
to the copyright holder.

## Disclaimer

This is an independent annotation-alignment and QA utility. Dataset
files must be obtained and used according to their original licenses
and terms.
