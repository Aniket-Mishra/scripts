import argparse
import glob
import io
import os
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image


def sanitize(string: str) -> str:
    """
    Function to replace special chars with _.

    Args:
        string (str): Input.

    Returns:
        str: Sanitized output.
    """
    sanitized_string = re.sub(r"[^a-zA-Z0-9]+", "_", string)
    sanitized_string = sanitized_string.strip("_")
    return sanitized_string


def extract_formulas(text: str) -> List[str]:
    """
    Function to extract formulas in plain text.

    Args:
        text (str): The input text to search for formulas.

    Returns:
        List[str]: A list of extracted formula strings.
    """
    print("Formula extraction doesnt always work as expected, pls be careful.")
    formulas = []
    math_indicators = [
        "=",
        "∑",
        "∫",
        "√",
        "π",
        "\\\\",
        "^",
        "α",
        "β",
        "γ",
        "δ",
        "ε",
        "ζ",
        "η",
        "θ",
        "ι",
        "κ",
        "λ",
        "μ",
        "ν",
        "ξ",
        "ο",
        "π",
        "ρ",
        "σ",
        "τ",
        "υ",
        "φ",
        "χ",
        "ψ",
        "ω",
        "Γ",
        "Δ",
        "Θ",
        "Λ",
        "Ξ",
        "Π",
        "Σ",
        "Φ",
        "Ψ",
        "Ω",
    ]
    for line in text.splitlines():
        if (
            any(symbol in line for symbol in math_indicators)
            and len(line.strip()) > 3
        ):
            formulas.append(line.strip())
    return formulas


def extract_pdf_content(
    pdf_obj: fitz.Document, output_base_dir: Path, paper_name: str
) -> None:
    """
    Function to extract text, formulas, and images from a PDF and saves them.
    For images, it saves both a .npy array and the image file.

    Args:
        pdf_obj (fitz.Document): The PDF object opened with fitz.open().
        output_base_dir (Path): The parent output directory where subfolders will be created.
        paper_name (str): The paper name used to generate filenames.
    """
    text_dir = output_base_dir / "text"
    image_dir = output_base_dir / "images"
    formula_dir = output_base_dir / "formulas"

    text_dir.mkdir(exist_ok=True, parents=True)
    image_dir.mkdir(exist_ok=True, parents=True)
    formula_dir.mkdir(exist_ok=True, parents=True)

    print(f"Extracting content for '{paper_name}'...")

    for page_num in range(len(pdf_obj)):
        page = pdf_obj.load_page(page_num)
        text = page.get_text()

        txt_path = text_dir / f"{paper_name}_page_{page_num + 1}.txt"
        np.savetxt(str(txt_path), np.array([text]), fmt="%s", encoding="utf-8")

        md_text = f"# Page {page_num + 1}\n\n{text}"
        md_text_path = text_dir / f"{paper_name}_page_{page_num + 1}.md"
        np.savetxt(
            str(md_text_path), np.array([md_text]), fmt="%s", encoding="utf-8"
        )

        # Extract and save formulas
        formulas = extract_formulas(text)
        if formulas:
            formula_md = (
                f"# Formulas from Page {page_num + 1}\n\n"
                + "\n\n".join([f"$$\n{formula}\n$$" for formula in formulas])
            )
            formula_md_path = (
                formula_dir / f"{paper_name}_page_{page_num + 1}_formulas.md"
            )
            np.savetxt(
                str(formula_md_path),
                np.array([formula_md]),
                fmt="%s",
                encoding="utf-8",
            )

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_obj.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            npy_image_filename = (
                image_dir
                / f"{paper_name}_page_{page_num + 1}_img_{img_index + 1}.npy"
            )
            np.save(npy_image_filename, image_array)

            image_file_path = (
                image_dir
                / f"{paper_name}_page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            )
            try:
                image_obj = Image.open(io.BytesIO(image_bytes))
                image_obj.save(image_file_path)
            except Exception as e:
                print(
                    f"Warning: Could not save image {image_file_path}. Error: {e}"
                )

    print(
        f"Text, images, and formulas extracted to subfolders within: {output_base_dir}"
    )


def extract_tables_from_pdf(
    pdf_path: str, output_base_dir: Path, paper_name: str
) -> None:
    """
    Function to extract tables from a PDF and saves them as CSV and Markdown.

    Args:
        pdf_path (str): The path to the PDF file.
        output_base_dir (Path): The parent output directory where subfolders will be created.
        paper_name (str): The paper name used to generate filenames.
    """
    table_dir = output_base_dir / "tables"
    table_dir.mkdir(exist_ok=True, parents=True)

    print(f"Extracting tables for '{paper_name}'...")

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                for table_index, table in enumerate(tables):
                    df = pd.DataFrame(table)
                    filename_base = (
                        table_dir
                        / f"{paper_name}_page_{page_num + 1}_table_{table_index + 1}"
                    )

                    csv_filename = f"{filename_base}.csv"
                    df.to_csv(
                        csv_filename,
                        index=False,
                        header=False,
                        encoding="utf-8",
                    )

                    md_filename = f"{filename_base}.md"
                    md_str = df.to_markdown(index=False)
                    np.savetxt(
                        str(md_filename),
                        np.array([md_str]),
                        fmt="%s",
                        encoding="utf-8",
                    )

                    print(
                        f"Page {page_num + 1}, Table {table_index + 1} extracted."
                    )
    print(f"Tables extracted to: {table_dir}")


def combine_text_files(text_folder_path: Path, output_base_dir: Path) -> None:
    """
    Combine text files for each paper in the specified text_folder_path into combined files.
    The combined files are saved in the output_base_dir/combined_text folder in both '.txt' and '.md' formats.

    Args:
        text_folder_path (Path): Path to the folder containing individual page text files (e.g., output_base_dir/text).
        output_base_dir (Path): The parent output directory.
    """
    output_folder = output_base_dir / "combined_text"
    output_folder.mkdir(exist_ok=True, parents=True)

    print(
        f"Combining text files from '{text_folder_path}' into '{output_folder}'..."
    )

    file_list = sorted(
        glob.glob(os.path.join(str(text_folder_path), "*_page_*.txt"))
    )
    paper_content = {}

    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = re.match(r"(.+)_page_\d+\.txt$", filename)
        if match:
            paper_name = match.group(1)
            if paper_name not in paper_content:
                paper_content[paper_name] = []
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                paper_content[paper_name].append(content)

    for paper_name, contents in paper_content.items():
        combined_content = "\n\n".join(contents)

        output_filename_txt = f"{paper_name}_combined_all_text.txt"
        output_path_txt = output_folder / output_filename_txt
        np.savetxt(
            str(output_path_txt),
            np.array([combined_content]),
            fmt="%s",
            encoding="utf-8",
        )

        output_filename_md = f"{paper_name}_combined_all_text.md"
        output_path_md = output_folder / output_filename_md
        markdown_content = (
            f"# {paper_name} Combined Text\n\n" + combined_content
        )
        np.savetxt(
            str(output_path_md),
            np.array([markdown_content]),
            fmt="%s",
            encoding="utf-8",
        )
        print(f"Combined files:\n{output_path_txt}\n{output_path_md}")
    print("Text combination complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract content (text, images, tables, formulas) from a PDF."
    )
    # parser.add_argument(
    #     "pdf_path",
    #     type=str,
    #     help="Path to the input PDF file (e.g., /path/to/paper.pdf)",
    # )
    # parser.add_argument(
    #     "output_parent_folder",
    #     type=str,
    #     help="Path to the parent output folder where extracted content will be saved "
    #          "(e.g., /path/to/extracted_papers). Subfolders will be created inside.",
    # )

    parser.add_argument(
        "--pdf_path",
        type=str,
        required=True,
        help="Path to the input PDF file (e.g., /path/to/paper.pdf)",
    )
    parser.add_argument(
        "--output_parent_folder",
        type=str,
        required=True,
        help="Path to the parent output folder where extracted content will be saved "
        "(e.g., /path/to/extracted_papers). Subfolders will be created inside.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_parent_folder = Path(args.output_parent_folder)

    if not pdf_path.is_file():
        print(f"Error: PDF file not found at '{pdf_path}'")
        return

    paper_name = sanitize(pdf_path.stem)
    paper_output_dir = output_parent_folder / paper_name
    paper_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output will be saved to: {paper_output_dir}")

    try:
        pdf_fitz = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF with PyMuPDF (fitz): {e}")
        return

    extract_pdf_content(pdf_fitz, paper_output_dir, paper_name)
    pdf_fitz.close()

    extract_tables_from_pdf(str(pdf_path), paper_output_dir, paper_name)
    combine_text_files(paper_output_dir / "text", paper_output_dir)

    print("\nExtraction process completed successfully!")


if __name__ == "__main__":
    main()
