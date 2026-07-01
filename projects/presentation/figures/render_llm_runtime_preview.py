from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "llm-runtime-architecture-preview.png"

S = 2
CW, CH = 720, 405
W, H = CW * S, CH * S


def rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


BG = rgb("#F8FAFC")
INK = rgb("#1F2937")
MUTED = rgb("#64748B")
LINE = rgb("#CBD5E1")
BLUE_D = rgb("#1E3A5F")
BLUE_L = rgb("#DBEAFE")
PURPLE = rgb("#6D28D9")
PURPLE_L = rgb("#DDD6FE")
GREEN = rgb("#047857")
GREEN_L = rgb("#A7F3D0")
ORANGE = rgb("#C2410C")
ORANGE_L = rgb("#FED7AA")
YELLOW = rgb("#B45309")
YELLOW_L = rgb("#FEF3C7")
RED = rgb("#DC2626")
RED_L = rgb("#FEE2E2")
WHITE = rgb("#FFFFFF")


def font(size, bold=False):
    for name in ("arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size * S)
        except OSError:
            pass
    return ImageFont.load_default()


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def sx(x):
    return int(round(x * S))


def sy(y):
    return int(round((CH - y) * S))


def rect_coords(x, y, w, h):
    return [sx(x), sy(y + h), sx(x + w), sy(y)]


def rr(x, y, w, h, r, fill, outline, width=2):
    d.rounded_rectangle(rect_coords(x, y, w, h), radius=sx(r), fill=fill, outline=outline, width=width)


def txt(x, y, s, size=8, color=INK, bold=False, center=False):
    f = font(size, bold)
    px = sx(x)
    py = sy(y) - sx(size)
    if center:
        bbox = d.textbbox((0, 0), s, font=f)
        px -= (bbox[2] - bbox[0]) // 2
    d.text((px, py), s, font=f, fill=color)


def line_arrow(points, color=BLUE_D, width=3):
    pts = [(sx(x), sy(y)) for x, y in points]
    d.line(pts, fill=color, width=width, joint="curve")
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        tri = [(x2, y2), (x2 - sign * sx(8), y2 + sx(4)), (x2 - sign * sx(8), y2 - sx(4))]
    else:
        sign = 1 if y2 >= y1 else -1
        tri = [(x2, y2), (x2 - sx(4), y2 + sign * sx(8)), (x2 + sx(4), y2 + sign * sx(8))]
    d.polygon(tri, fill=color)


def lane(x, y, w, h, title, subtitle, color):
    rr(x, y, w, h, 12, WHITE, LINE, 2)
    d.rectangle(rect_coords(x, y + h - 22, w, 22), fill=color)
    txt(x + 12, y + h - 15, title, 8, WHITE, True)
    txt(x + 110, y + h - 15, subtitle, 7, rgb("#E5E7EB"))


def card(x, y, w, h, title, lines, fill, stroke, tag=None):
    rr(x, y, w, h, 9, fill, stroke, 3)
    txt(x + 10, y + h - 16, title, 8, stroke, True)
    d.ellipse(rect_coords(x + w - 15, y + h - 15, 6, 6), fill=stroke)
    txt(x + 10, y + h - 30, lines[0], 6, INK)
    for i, item in enumerate(lines[1:], start=1):
        txt(x + 10, y + h - 30 - i * 10, item, 6, MUTED)
    if tag:
        rr(x + 9, y + 7, 42, 12, 6, WHITE, stroke, 2)
        txt(x + 30, y + 10, tag, 5, stroke, True, True)


def pill(x, y, w, label, fill, stroke):
    rr(x, y, w, 18, 9, fill, stroke, 2)
    txt(x + w / 2, y + 6, label, 6, stroke, True, True)


txt(24, 374, "OSH26 On-device LLM Runtime", 15, BLUE_D, True)
txt(24, 358, "OS mechanisms inside an Android llama.cpp fork: scheduling, memory hierarchy, cache reuse, and heterogeneous execution", 7, MUTED)

lane(24, 264, 672, 76, "Control plane", "agent requests, admission, cancel, health", BLUE_D)
lane(24, 151, 672, 92, "Memory plane", "unified memory policy + vLLM-inspired KV cache", PURPLE)
lane(24, 39, 672, 90, "Execution plane", "CPU fallback + Adreno Vulkan quantized backend", GREEN)

card(44, 272, 108, 44, "Agent UI", ["chat / subagent", "streaming tokens"], ORANGE_L, ORANGE)
card(184, 272, 126, 44, "JNI + local API", ["/v1/chat  /health", "request-local state"], BLUE_L, BLUE_D)
card(342, 272, 124, 44, "Scheduler Lite", ["queue, cancel, aging", "prefix-aware admission"], BLUE_L, BLUE_D)
card(504, 272, 138, 44, "Health gate", ["TTFT / TPS / cache hit", "logits + descriptor sanity"], BLUE_L, BLUE_D)

rr(72, 160, 206, 56, 14, PURPLE_L, PURPLE, 3)
txt(175, 197, "Paged Prefix KV Cache", 11, PURPLE, True, True)
txt(175, 182, "pinned system prompt + dynamic request pages", 7, INK, False, True)
pill(91, 166, 58, "restore", WHITE, PURPLE)
pill(158, 166, 48, "evict", WHITE, PURPLE)
pill(215, 166, 42, "reuse", WHITE, PURPLE)

card(332, 168, 118, 42, "llama.cpp base", ["GGUF + tokenizer", "sampler + CPU ctx"], BLUE_L, BLUE_D, "base")
card(490, 168, 134, 42, "Unified memory", ["CPU control state", "GPU-resident hot tensors"], PURPLE_L, PURPLE, "policy")
card(60, 56, 126, 42, "CPU fallback", ["correctness oracle", "small ops + sampler"], RED_L, RED)
card(254, 56, 154, 42, "OSH26 Vulkan backend", ["Q8 GEMV / prefill / decode", "single-submit + top-k gate"], GREEN_L, GREEN)
card(478, 56, 132, 42, "Adreno GPU", ["device-local weights", "packed FP16 KV pages"], GREEN_L, GREEN)

line_arrow([(152, 294), (184, 294)], ORANGE)
line_arrow([(310, 294), (342, 294)], BLUE_D)
line_arrow([(466, 294), (504, 294)], BLUE_D)
line_arrow([(404, 272), (404, 252), (212, 252), (212, 216)], PURPLE)
txt(222, 255, "prefix lookup", 6, PURPLE, True)
line_arrow([(278, 188), (332, 188)], PURPLE)
line_arrow([(450, 188), (490, 188)], BLUE_D)
line_arrow([(557, 168), (557, 129)], PURPLE)
txt(566, 143, "placement", 6, PURPLE, True)
line_arrow([(390, 168), (390, 134), (330, 98)], GREEN)
line_arrow([(390, 168), (390, 134), (123, 98)], RED)
txt(250, 137, "backend select", 6, MUTED, True)
line_arrow([(408, 77), (478, 77)], GREEN)
line_arrow([(544, 98), (544, 137), (642, 137), (642, 272)], BLUE_D)
txt(562, 130, "telemetry feedback", 6, BLUE_D, True)
rr(43, 246, 173, 17, 8, YELLOW_L, YELLOW, 2)
txt(129, 251, "stateless subagent: no private history, reuse pinned prefix", 6, YELLOW, True, True)
line_arrow([(128, 272), (128, 263)], YELLOW)
line_arrow([(128, 246), (128, 216)], YELLOW)

txt(544, 360, "Legend", 7, MUTED, True)
pill(584, 354, 40, "OS", BLUE_L, BLUE_D)
pill(630, 354, 42, "Cache", PURPLE_L, PURPLE)
pill(584, 331, 42, "GPU", GREEN_L, GREEN)
pill(630, 331, 44, "Agent", ORANGE_L, ORANGE)

img.save(OUT)
print(OUT)
