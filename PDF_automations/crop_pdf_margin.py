import fitz  # PyMuPDF


def crop_page_to_content(page, margin=10):
    blocks = page.get_text("blocks")
    if not blocks:
        return page.rect
    x0 = min(block[0] for block in blocks)
    y0 = min(block[1] for block in blocks)
    x1 = max(block[2] for block in blocks)
    y1 = max(block[3] for block in blocks)
    bbox = fitz.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    bbox = bbox & page.rect
    return bbox


def combine_pages(input_pdf, output_pdf, page_numbers, layout="horizontal"):
    doc = fitz.open(input_pdf)
    cropped_items = []  # will hold tuples of (pixmap, width, height)

    for pn in page_numbers:
        page = doc[pn]
        bbox = crop_page_to_content(page, margin=10)
        pix = page.get_pixmap(clip=bbox)
        cropped_items.append((pix, bbox.width, bbox.height))
    doc.close()

    if layout == "horizontal":
        total_width = sum(width for _, width, _ in cropped_items)
        max_height = max(height for _, _, height in cropped_items)
        new_page_rect = fitz.Rect(0, 0, total_width, max_height)
    elif layout == "vertical":
        total_height = sum(height for _, _, height in cropped_items)
        max_width = max(width for _, width, _ in cropped_items)
        new_page_rect = fitz.Rect(0, 0, max_width, total_height)
    else:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    new_doc = fitz.open()
    new_page = new_doc.new_page(
        width=new_page_rect.width, height=new_page_rect.height
    )

    # cropped images to new page
    if layout == "horizontal":
        x_offset = 0
        for pix, width, height in cropped_items:
            img_bytes = pix.tobytes("png")
            rect = fitz.Rect(x_offset, 0, x_offset + width, height)
            new_page.insert_image(rect, stream=img_bytes)
            x_offset += width
    elif layout == "vertical":
        y_offset = 0
        for pix, width, height in cropped_items:
            img_bytes = pix.tobytes("png")
            rect = fitz.Rect(0, y_offset, width, y_offset + height)
            new_page.insert_image(rect, stream=img_bytes)
            y_offset += height

    new_doc.save(output_pdf)
    new_doc.close()
    print(f"Combined PDF saved as {output_pdf}")


if __name__ == "__main__":
    lst = list(
        range(0, 527)
    )  # num pages. This can be configurable, but im lazy rn.
    chunk_size = 3
    result = [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
    print(result)

    input_file = "/Users/aniket/TU_Eindhoven/2_Study/1_Thesis_Prep/1_extract_paper_content/MLE_Notes_Combined_1_to_10.pdf"
    for i, chunk in enumerate(result):
        # create the folders.
        output_file = f"MLE_notes_3x3_split/note_chunk_{i}_combined.pdf"
        combine_pages(input_file, output_file, chunk, layout="horizontal")
        output_file = f"MLE_notes_3x3_split_vert/note_chunk_{i}_combined.pdf"
        combine_pages(input_file, output_file, chunk, layout="vertical")
