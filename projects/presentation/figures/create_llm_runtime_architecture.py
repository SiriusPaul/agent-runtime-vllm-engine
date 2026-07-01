from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "llm-runtime-architecture.pdf"

W, H = 720, 405

BG = colors.HexColor("#F8FAFC")
WHITE = colors.HexColor("#FFFFFF")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#CBD5E1")

BLUE = colors.HexColor("#1E3A5F")
BLUE_L = colors.HexColor("#DBEAFE")
PURPLE = colors.HexColor("#6D28D9")
PURPLE_L = colors.HexColor("#DDD6FE")
GREEN = colors.HexColor("#047857")
GREEN_L = colors.HexColor("#A7F3D0")
ORANGE = colors.HexColor("#C2410C")
ORANGE_L = colors.HexColor("#FED7AA")
RED = colors.HexColor("#DC2626")
RED_L = colors.HexColor("#FEE2E2")
YELLOW = colors.HexColor("#B45309")
YELLOW_L = colors.HexColor("#FEF3C7")


def rr(c, x, y, w, h, fill, stroke, r=10, lw=1.2):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(lw)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1)


def text(c, x, y, s, size=9, color=INK, bold=False, center=False):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if center:
        c.drawCentredString(x, y, s)
    else:
        c.drawString(x, y, s)


def arrow(c, points, color=BLUE, lw=1.7, dashed=False):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(lw)
    c.setDash(4, 3) if dashed else c.setDash()
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p, fill=0, stroke=1)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        tri = [(x2, y2), (x2 - sign * 8, y2 + 4), (x2 - sign * 8, y2 - 4)]
    else:
        sign = 1 if y2 >= y1 else -1
        tri = [(x2, y2), (x2 - 4, y2 - sign * 8), (x2 + 4, y2 - sign * 8)]
    tp = c.beginPath()
    tp.moveTo(*tri[0])
    tp.lineTo(*tri[1])
    tp.lineTo(*tri[2])
    tp.close()
    c.drawPath(tp, fill=1, stroke=0)
    c.setDash()


def box(c, x, y, w, h, title, subtitle, fill, stroke, title_size=10.5):
    rr(c, x, y, w, h, fill, stroke, 10, 1.2)
    text(c, x + w / 2, y + h - 18, title, title_size, stroke, True, True)
    if subtitle:
        text(c, x + w / 2, y + h - 35, subtitle, 8.8, INK, False, True)


def label_pill(c, x, y, w, s, fill, stroke):
    rr(c, x, y, w, 17, fill, stroke, 8, 0.9)
    text(c, x + w / 2, y + 5, s, 7, stroke, True, True)


def main():
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Android / Java layer.
    rr(c, 18, 325, 684, 62, WHITE, LINE, 13, 0.9)
    text(c, 34, 371, "Android / Java layer", 10, MUTED, True)
    box(c, 42, 339, 122, 33, "MainActivity", "UI", ORANGE_L, ORANGE, 11)
    box(c, 188, 339, 154, 33, "LlmHttpServer", "/v1/chat /health", ORANGE_L, ORANGE, 11)
    box(c, 386, 339, 142, 33, "LlamaNative", "JNI facade", BLUE_L, BLUE, 11)
    box(c, 560, 339, 112, 33, "Callbacks", "stream", BLUE_L, BLUE, 11)
    arrow(c, [(164, 356), (188, 356)], ORANGE)
    arrow(c, [(342, 356), (386, 356)], BLUE)
    arrow(c, [(528, 356), (560, 356)], BLUE)

    # JNI bridge.
    box(c, 292, 280, 140, 36, "native-lib.cpp", "JNI bridge", BLUE_L, BLUE, 11.5)
    arrow(c, [(458, 339), (412, 316)], BLUE)

    # C++ engine.
    rr(c, 72, 145, 576, 122, WHITE, BLUE, 14, 1.5)
    text(c, 94, 248, "OSH26 C++ Engine: osh26::ComputeBackend", 14.5, BLUE, True)
    box(c, 98, 186, 146, 42, "Request Queue", "worker / cancel", BLUE_L, BLUE, 11.2)
    box(c, 286, 186, 152, 42, "Prefix Cache", "pinned + LRU", PURPLE_L, PURPLE, 11.2)
    box(c, 482, 186, 142, 42, "llama.cpp State", "model / ctx", BLUE_L, BLUE, 11.2)
    arrow(c, [(432, 280), (360, 267)], BLUE)
    arrow(c, [(244, 207), (286, 207)], BLUE)
    arrow(c, [(438, 207), (482, 207)], PURPLE)

    # CPU and GPU branch.
    box(c, 42, 54, 156, 50, "llama.cpp CPU", "fallback path", RED_L, RED, 11)
    box(c, 250, 54, 188, 50, "OSH26 Vulkan", "osh26_vk_gpu.c", GREEN_L, GREEN, 12)
    box(c, 520, 54, 146, 50, "Vulkan Wrapper", "Adreno device", GREEN_L, GREEN, 11)
    arrow(c, [(552, 186), (552, 126), (120, 104)], RED, 1.5, True)
    arrow(c, [(552, 186), (552, 126), (344, 104)], GREEN)
    arrow(c, [(438, 79), (520, 79)], GREEN)

    # Vulkan internals.
    rr(c, 214, 11, 324, 34, GREEN_L, GREEN, 8, 1.1)
    text(c, 376, 32, "Q8 GEMV  |  Attention  |  LM Head Top-k", 11.6, GREEN, True, True)
    rr(c, 558, 11, 126, 34, PURPLE_L, PURPLE, 8, 1.1)
    text(c, 621, 32, "GPU KV Cache", 12, PURPLE, True, True)
    arrow(c, [(344, 54), (344, 45)], GREEN)
    arrow(c, [(590, 54), (620, 45)], PURPLE)

    # Important design note.
    label_pill(c, 46, 281, 192, "GGML OpenCL/Vulkan disabled", RED_L, RED)
    label_pill(c, 478, 281, 158, "handwritten quant backend", YELLOW_L, YELLOW)
    arrow(c, [(535, 281), (394, 104)], YELLOW)

    # Telemetry loop.
    arrow(c, [(620, 54), (620, 145), (600, 186)], PURPLE)
    arrow(c, [(624, 207), (680, 207), (680, 339)], BLUE)
    text(c, 648, 216, "stats", 8, BLUE, True)

    text(c, 360, 274, "native generate / load / resetCache / getEngineStats", 9, MUTED, False, True)

    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
