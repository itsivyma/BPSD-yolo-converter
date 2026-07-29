from converter import yolo_to_pixels


def test_first_reference_box():
    box = {
        "x": 0.201429,
        "y": 0.228247,
        "w": 0.022286,
        "h": 0.017574,
    }

    result = yolo_to_pixels(
        box,
        image_width=1750,
        image_height=2333,
    )

    assert result["left"] == 333
    assert result["top"] == 512
    assert result["width"] == 39
    assert result["height"] == 41