for name in ("fitz", "pypdfium2", "pdf2image"):
    try:
        __import__(name)
        print(f"{name}: yes")
    except Exception as exc:
        print(f"{name}: no ({type(exc).__name__})")
