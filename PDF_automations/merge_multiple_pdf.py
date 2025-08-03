import glob

import fitz  # PyMuPDF

pdf_files = sorted(
    glob.glob(
        "/Users/aniket/TU_Eindhoven/2_Study/Q3_2AMM15_Machine_Learning_Engineering/1_Notes/*.pdf"
    )
)

merged_pdf = fitz.open()

pdf_files = pdf_files[1:]

print(pdf_files)

for pdf in pdf_files:
    with fitz.open(pdf) as mdoc:
        merged_pdf.insert_pdf(mdoc)

merged_pdf.save("MLE_Notes_Combined_1_to_10.pdf")
merged_pdf.close()

print("Done")
