from pathlib import Path

import pypdfium2 as pdfium

root = Path(__file__).resolve().parent

jobs = [
    (root / "llm-runtime-architecture.pdf", 0, root / "llm-runtime-architecture-render.png"),
    (root.parent / "final-presentation-beamer.pdf", 34, root / "final-beamer-architecture-slide.png"),
]

for pdf_path, page_index, out_path in jobs:
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_index]
    bitmap = page.render(scale=2.0)
    pil_image = bitmap.to_pil()
    pil_image.save(out_path)
    print(out_path)
