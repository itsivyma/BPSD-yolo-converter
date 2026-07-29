import io
from PIL import Image

from pathlib import Path
import streamlit as st

from converter import (
    draw_boxes,
    load_categories,
    load_yolo_txt,
    make_csv,
    make_xlsx,
    validate_box,
)


st.set_page_config(
    page_title="BPSD YOLO 轉換系統",
    layout="wide",
)

st.title("BPSD YOLO 自動轉換系統")

st.write(
    "將 YOLO TXT 轉成五欄 CSV，"
    "並把 bounding boxes 畫回圖片進行確認。"
)


st.subheader("1. 上傳檔案")

notes_file = st.file_uploader(
    "上傳 notes.json",
    type=["json"],
)

txt_files = st.file_uploader(
    "上傳 YOLO TXT",
    type=["txt"],
    accept_multiple_files=True,
)

image_files = st.file_uploader(
    "上傳圖片",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)


categories = None

if notes_file is None:
    st.info("請先上傳 notes.json")
else:
    try:
        notes_file.seek(0)
        categories = load_categories(notes_file)

        st.success(
            f"notes.json 讀取成功："
            f"{len(categories)} 個 classes"
        )
    except (ValueError, TypeError) as error:
        st.error(f"notes.json 讀取失敗：{error}")


boxes = None
selected_txt = None

if not txt_files:
    st.info("請上傳至少一個 YOLO TXT")
else:
    txt_options = {
        txt_file.name: txt_file
        for txt_file in txt_files
    }

    selected_txt_name = st.selectbox(
        "選擇要檢查的 YOLO TXT",
        options=list(txt_options.keys()),
    )

    selected_txt = txt_options[selected_txt_name]

    try:
        selected_txt.seek(0)
        boxes = load_yolo_txt(selected_txt)

        st.success(
            f"{selected_txt_name} 讀取成功："
            f"{len(boxes)} 個 bounding boxes"
        )
    except ValueError as error:
        st.error(f"YOLO TXT 讀取失敗：{error}")


if categories is not None and boxes is not None:
    st.subheader("2. 資料驗證")

    validation_errors = []
    validation_warnings = []

    for box in boxes:
        errors, warnings = validate_box(
            box,
            categories,
        )

        for message in errors:
            validation_errors.append(
                f'第 {box["line_number"]} 列：{message}'
            )

        for message in warnings:
            validation_warnings.append(
                f'第 {box["line_number"]} 列：{message}'
            )

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "Bounding boxes",
        len(boxes),
    )
    column2.metric(
        "Errors",
        len(validation_errors),
    )
    column3.metric(
        "Warnings",
        len(validation_warnings),
    )

    if validation_errors:
        st.error("\n".join(validation_errors))
    else:
        st.success("沒有發現座標或 class ID 錯誤")

    if validation_warnings:
        st.warning("\n".join(validation_warnings))


if categories is not None and boxes is not None:
    st.subheader("3. Bounding box 預覽")

    txt_stem = Path(selected_txt_name).stem

    image_options = {
        Path(image_file.name).stem: image_file
        for image_file in image_files
    }

    if txt_stem not in image_options:
        st.warning(
            f"找不到與 {selected_txt_name} "
            "同名的圖片"
        )
    else:
        selected_image = image_options[txt_stem]
        selected_image.seek(0)

        image = Image.open(selected_image)

        show_labels = st.checkbox(
            "顯示 class ID 與名稱",
            value=False,
        )

        preview = draw_boxes(
            image,
            boxes,
            categories,
            show_labels=show_labels,
        )

        st.caption(
            f"圖片尺寸："
            f"{image.width} × {image.height} px"
        )

        st.image(
            preview,
            caption=selected_image.name,
            use_container_width=True,
        )
        qa_buffer = io.BytesIO()

        preview.save(
            qa_buffer,
            format="PNG",
        )

        st.download_button(
            label="下載 QA 疊圖 PNG",
            data=qa_buffer.getvalue(),
            file_name=f"{txt_stem}_QA.png",
            mime="image/png",
        )


if categories is not None and boxes is not None:
    st.subheader("4. 下載 CSV")

    csv_data = make_csv(boxes)

    st.download_button(
        label="下載五欄 CSV",
        data=csv_data,
        file_name=f"{txt_stem}.csv",
        mime="text/csv",
        disabled=bool(validation_errors),
    )

    if validation_errors:
        st.caption(
            "資料有 errors，修正後才能下載 CSV。"
        )


if categories is not None and txt_files:
    st.subheader("5. 整首 Sonata Excel")

    sonata_groups = {}
    batch_read_errors = []

    for uploaded_txt in txt_files:
        page_name = Path(uploaded_txt.name).stem
        name_parts = page_name.split("-")

        if len(name_parts) < 3:
            batch_read_errors.append(
                f"無法辨識檔名格式："
                f"{uploaded_txt.name}"
            )
            continue

        sonata_name = name_parts[0]

        try:
            uploaded_txt.seek(0)
            page_boxes = load_yolo_txt(
                uploaded_txt
            )
        except ValueError as error:
            batch_read_errors.append(
                f"{uploaded_txt.name}：{error}"
            )
            continue

        if sonata_name not in sonata_groups:
            sonata_groups[sonata_name] = {}

        sonata_groups[sonata_name][
            page_name
        ] = page_boxes

    if batch_read_errors:
        st.error(
            "\n".join(batch_read_errors)
        )

    if sonata_groups:
        selected_sonata = st.selectbox(
            "選擇要輸出的 Sonata",
            options=sorted(
                sonata_groups.keys()
            ),
        )

        selected_pages = sonata_groups[
            selected_sonata
        ]

        st.write(
            f"{selected_sonata}："
            f"{len(selected_pages)} 張圖片"
        )

        st.code(
            "\n".join(
                sorted(selected_pages.keys())
            )
        )
        sonata_errors = []
        sonata_warnings = []

        for page_name, page_boxes in (
            selected_pages.items()
        ):
            for box in page_boxes:
                errors, warnings = validate_box(
                    box,
                    categories,
                )

                for message in errors:
                    sonata_errors.append(
                        f"{page_name} "
                        f'第 {box["line_number"]} 列：'
                        f"{message}"
                    )

                for message in warnings:
                    sonata_warnings.append(
                        f"{page_name} "
                        f'第 {box["line_number"]} 列：'
                        f"{message}"
                    )

        total_boxes = sum(
            len(page_boxes)
            for page_boxes
            in selected_pages.values()
        )

        excel_column1, excel_column2, excel_column3 = (
            st.columns(3)
        )

        excel_column1.metric(
            "Sonata 總框數",
            total_boxes,
        )
        excel_column2.metric(
            "Sonata Errors",
            len(sonata_errors),
        )
        excel_column3.metric(
            "Sonata Warnings",
            len(sonata_warnings),
        )

        if sonata_errors:
            st.error(
                "\n".join(sonata_errors)
            )

        if sonata_warnings:
            st.warning(
                "\n".join(sonata_warnings)
            )

        xlsx_data = make_xlsx(
            selected_pages
        )

        st.download_button(
            label="下載整首 Sonata Excel",
            data=xlsx_data,
            file_name=(
                f"{selected_sonata}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            disabled=bool(sonata_errors),
        )