import csv
import io

from PIL import Image, ImageDraw

from bps_xml_alignment import (
    OUTPUT_FIELDS,
    StaffGeometry,
    SystemGeometry,
    align_barlines_from_reference,
    attach_bps_note_ids,
    build_slur_candidates,
    conservative_all_symbol_rows,
    detect_barlines,
    detect_systems,
    load_categories,
    load_yolo,
    match_fingerings,
    snap_notehead_x,
    unresolved_fingering_rows,
    write_csv,
)


def test_load_yolo_keeps_original_line_number(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text(
        "18 0.2 0.3 0.1 0.1\n"
        "\n"
        "25 0.4 0.5 0.02 0.03\n",
        encoding="utf-8",
    )

    boxes = load_yolo(path)

    assert [box["txt_line"] for box in boxes] == [1, 3]
    assert [box["class"] for box in boxes] == ["dynamicF", "fingering1"]


def test_attach_bps_note_ids_matches_time_and_pitch():
    xml_notes = [
        {"bps_time": 1.0, "midi": 60, "staff": 1, "x_norm": 0.2},
        {"bps_time": 1.0, "midi": 64, "staff": 1, "x_norm": 0.2},
    ]
    bps_notes = [
        {"note_id": 10, "bps_time": 1.0, "midi": 60},
        {"note_id": 11, "bps_time": 1.0, "midi": 64},
    ]

    attach_bps_note_ids(xml_notes, bps_notes)

    assert [note["note_id"] for note in xml_notes] == [10, 11]


def test_attach_bps_note_ids_reuses_tied_note_span():
    xml_notes = [
        {"bps_time": 2.0, "midi": 60, "staff": 1, "x_norm": 0.2},
    ]
    bps_notes = [
        {
            "note_id": 12,
            "bps_time": 1.5,
            "end_time": 2.5,
            "midi": 60,
        },
    ]

    attach_bps_note_ids(xml_notes, bps_notes)

    assert xml_notes[0]["note_id"] == 12


def test_build_slur_candidates_pairs_endpoints_and_keeps_bps_time():
    xml_notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 1.0,
            "midi": 67,
            "pitch_name": "G4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 2,
            "note_id": 9,
            "slur_marks": [
                {
                    "type": "start",
                    "number": "1",
                    "orientation": "over",
                }
            ],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 1.5,
            "midi": 66,
            "pitch_name": "F#4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 2,
            "note_id": 12,
            "slur_marks": [
                {
                    "type": "stop",
                    "number": "1",
                    "orientation": "over",
                }
            ],
        },
    ]
    bps_notes = [
        {
            "note_id": 9,
            "bps_time": 1.0,
            "end_time": 1.5,
            "midi": 67,
        },
        {
            "note_id": 12,
            "bps_time": 1.5,
            "end_time": 1.667,
            "midi": 66,
        },
    ]

    candidates, issues = build_slur_candidates(xml_notes, bps_notes)

    assert issues == []
    assert len(candidates) == 1
    assert candidates[0]["start_meas"] == "1.000"
    assert candidates[0]["end_meas"] == "1.500"
    assert candidates[0]["start_pitch"] == "G4"
    assert candidates[0]["end_pitch"] == "F#4"
    assert candidates[0]["status"] == "time_confirmed"


def test_build_slur_candidates_reports_unpaired_endpoints():
    xml_notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 2.0,
            "midi": 60,
            "pitch_name": "C4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 3,
            "note_id": None,
            "slur_marks": [
                {
                    "type": "stop",
                    "number": "1",
                    "orientation": "",
                }
            ],
        }
    ]

    candidates, issues = build_slur_candidates(xml_notes, [])

    assert candidates == []
    assert issues[0]["issue"] == "stop_without_start"


def test_build_slur_candidates_allows_cross_staff_slur():
    xml_notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 15.0,
            "midi": 64,
            "pitch_name": "E4",
            "staff": 1,
            "voice": "1",
            "system": 2,
            "xml_measure": 16,
            "note_id": None,
            "slur_marks": [
                {
                    "type": "start",
                    "number": "1",
                    "orientation": "under",
                }
            ],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 15.667,
            "midi": 52,
            "pitch_name": "E3",
            "staff": 2,
            "voice": "1",
            "system": 2,
            "xml_measure": 16,
            "note_id": None,
            "slur_marks": [
                {
                    "type": "stop",
                    "number": "1",
                    "orientation": "under",
                }
            ],
        },
    ]

    candidates, issues = build_slur_candidates(xml_notes, [])

    assert issues == []
    assert len(candidates) == 1
    assert candidates[0]["start_staff"] == 1
    assert candidates[0]["end_staff"] == 2


def test_detect_systems_finds_paired_staves():
    image = Image.new("L", (1000, 500), "white")
    draw = ImageDraw.Draw(image)
    for staff_start in (80, 180, 300, 400):
        for offset in range(5):
            y = staff_start + offset * 10
            draw.line((80, y, 920, y), fill="black", width=2)

    systems = detect_systems(image.convert("RGB"))

    assert len(systems) == 2
    assert systems[0].upper.center < systems[0].lower.center
    assert systems[1].upper.center < systems[1].lower.center


def test_detect_barlines_uses_continuous_vertical_ink():
    image = Image.new("L", (1000, 400), "white")
    draw = ImageDraw.Draw(image)
    upper_lines = [80, 90, 100, 110, 120]
    lower_lines = [230, 240, 250, 260, 270]
    for y in upper_lines + lower_lines:
        draw.line((100, y, 900, y), fill="black", width=2)
    for x in (100, 300, 600, 900):
        draw.line((x, 80, x, 270), fill="black", width=3)

    # A note-like pair of vertical segments has substantial ink but does not
    # continuously connect the two staves, so it must not become a barline.
    draw.line((450, 80, 450, 145), fill="black", width=4)
    draw.line((450, 205, 450, 270), fill="black", width=4)

    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=upper_lines,
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=lower_lines,
            ),
            x_left=100,
            x_right=900,
        )
    ]

    boundaries = detect_barlines(
        image.convert("RGB"),
        systems,
        expected_boundary_counts=[4],
    )

    assert boundaries == [[100, 300, 600, 900]]


def test_align_barlines_from_reference_marks_occluded_line_for_review():
    reference = Image.new("L", (1000, 400), "white")
    target = Image.new("L", (1000, 400), "white")
    reference_draw = ImageDraw.Draw(reference)
    target_draw = ImageDraw.Draw(target)
    upper_lines = [80, 90, 100, 110, 120]
    lower_lines = [230, 240, 250, 260, 270]
    for draw in (reference_draw, target_draw):
        for y in upper_lines + lower_lines:
            draw.line((100, y, 900, y), fill="black", width=2)
    for x in (100, 300, 600, 900):
        reference_draw.line((x, 80, x, 270), fill="black", width=3)
        target_draw.line((x, 80, x, 270), fill="black", width=3)

    # Simulate a barline interrupted by a printed symbol in the target scan.
    target_draw.rectangle((598, 145, 602, 195), fill="white")
    geometry = SystemGeometry(
        number=1,
        upper=StaffGeometry(
            center=100,
            line_spacing=10,
            lines=upper_lines,
        ),
        lower=StaffGeometry(
            center=250,
            line_spacing=10,
            lines=lower_lines,
        ),
        x_left=100,
        x_right=900,
    )

    aligned = align_barlines_from_reference(
        target.convert("RGB"),
        [geometry],
        [geometry],
        [[100, 300, 600, 900]],
    )

    assert [item["x"] for item in aligned[0]] == [100, 300, 600, 900]
    assert aligned[0][1]["status"] == "detected"
    assert aligned[0][2]["status"] == "review_occluded"


def test_snap_notehead_x_ignores_staff_line_and_finds_dense_oval():
    image = Image.new("L", (800, 240), "white")
    draw = ImageDraw.Draw(image)
    lines = [80, 90, 100, 110, 120]
    for y in lines:
        draw.line((50, y, 750, y), fill="black", width=2)
    draw.ellipse((522, 95, 538, 105), fill="black")
    staff = StaffGeometry(
        center=100,
        line_spacing=10,
        lines=lines,
    )

    snapped = snap_notehead_x(
        image.convert("RGB"),
        predicted_x=510,
        predicted_y=100,
        staff=staff,
        search_radius=30,
    )

    assert 528 <= snapped["x"] <= 532


def test_stacked_fingerings_use_distinct_chord_notes():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=[80, 90, 100, 110, 120],
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=[230, 240, 250, 260, 270],
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [
        {
            "txt_line": 1,
            "class_id": 29,
            "class": "fingering5",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
        {
            "txt_line": 2,
            "class_id": 27,
            "class": "fingering3",
            "x": 0.5,
            "y": 0.20,
            "w": 0.01,
            "h": 0.01,
        },
    ]
    notes = [
        {
            "note_id": 10,
            "system": 1,
            "staff": 1,
            "x_norm": 0.5,
            "bps_time": 2.0,
            "xml_measure": 3,
            "pitch_name": "F5",
            "diatonic": 38,
            "clef": {"sign": "G", "line": 2},
        },
        {
            "note_id": 11,
            "system": 1,
            "staff": 1,
            "x_norm": 0.5,
            "bps_time": 2.0,
            "xml_measure": 3,
            "pitch_name": "B4",
            "diatonic": 34,
            "clef": {"sign": "G", "line": 2},
        },
    ]

    rows = match_fingerings(
        boxes,
        notes,
        systems,
        image_width=1000,
        image_height=400,
    )

    assert len(rows) == 2
    assert {row["start_note"] for row in rows} == {10, 11}


def test_unresolved_fingering_semantics_are_blank():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=[80, 90, 100, 110, 120],
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=[230, 240, 250, 260, 270],
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [
        {
            "txt_line": 1,
            "class_id": 29,
            "class": "fingering5",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
    ]

    rows = unresolved_fingering_rows(boxes, systems, image_height=400)

    assert rows[0]["class"] == "fingering5"
    assert rows[0]["musical_time"] == 0
    assert rows[0]["start_meas"] == ""
    assert rows[0]["start_note"] == ""
    assert rows[0]["connected_note"] == ""
    assert rows[0]["status"] == "unresolved"


def test_official_csv_has_only_bps_omr_fields(tmp_path):
    path = tmp_path / "output.csv"
    row = {
        "class_id": 18,
        "x": "0.2",
        "y": "0.3",
        "w": "0.01",
        "h": "0.02",
        "class": "dynamicF",
        "musical_time": 0,
        "start_meas": "0.667",
        "end_meas": "0.667",
        "start_note": "NA",
        "end_note": "NA",
        "connected_note": "NA",
        "stem_dir": "NA",
        "xml_measure": 1,
        "status": "matched",
        "confidence": "1.000",
    }

    write_csv(path, [row])

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        output_rows = list(reader)

    assert reader.fieldnames == OUTPUT_FIELDS
    assert output_rows[0]["class"] == "dynamicF"
    assert "xml_measure" not in output_rows[0]
    assert "status" not in output_rows[0]


def test_load_categories_uses_notes_json_names(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text(
        '{"categories":[{"id":56,"name":"slur"}]}',
        encoding="utf-8",
    )

    assert load_categories(path) == {56: "slur"}


def test_all_symbol_policy_leaves_undocumented_flags_blank():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=[80, 90, 100, 110, 120],
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=[230, 240, 250, 260, 270],
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [
        {
            "txt_line": 1,
            "class_id": 23,
            "class": "fermataAbove",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
        {
            "txt_line": 2,
            "class_id": 62,
            "class": "tempoInTempo",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
        {
            "txt_line": 3,
            "class_id": 107,
            "class": "tie",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
    ]

    rows = conservative_all_symbol_rows(
        boxes,
        systems,
        image_height=400,
    )

    assert rows[0]["musical_time"] == ""
    assert rows[1]["musical_time"] == 1
    assert rows[1]["start_note"] == "NA"
    assert rows[2]["musical_time"] == 0
    assert rows[2]["start_note"] == ""
    assert all(row["stem_dir"] == "NA" for row in rows)
