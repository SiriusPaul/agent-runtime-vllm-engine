from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "svg_output"
OUT.mkdir(exist_ok=True)

C = {
    "bg": "#0B1020",
    "panel": "#111827",
    "panel2": "#16213A",
    "primary": "#2563EB",
    "primary2": "#60A5FA",
    "accent": "#22C55E",
    "purple": "#A78BFA",
    "warn": "#EF4444",
    "text": "#F8FAFC",
    "sub": "#CBD5E1",
    "muted": "#64748B",
    "border": "#1F2A44",
    "grid": "#334155",
}

FONT = "Microsoft YaHei, Arial, sans-serif"
MONO = "Consolas, monospace"


def text(x, y, s, size=18, color=None, weight=None, anchor=None, family=None):
    attrs = [
        f'x="{x}"', f'y="{y}"',
        f'font-family="{family or FONT}"',
        f'font-size="{size}"',
        f'fill="{color or C["sub"]}"',
    ]
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    return f"<text {' '.join(attrs)}>{escape(s)}</text>"


def multiline(x, y, lines, size=18, color=None, gap=30, weight=None):
    attrs = [
        f'x="{x}"', f'y="{y}"',
        f'font-family="{FONT}"',
        f'font-size="{size}"',
        f'fill="{color or C["sub"]}"',
    ]
    if weight:
        attrs.append(f'font-weight="{weight}"')
    body = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else gap
        body.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    return f"<text {' '.join(attrs)}>" + "".join(body) + "</text>"


def rect(x, y, w, h, fill, stroke=None, rx=10, opacity=None):
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{rx}"', f'fill="{fill}"']
    if stroke:
        attrs.append(f'stroke="{stroke}"')
    if opacity:
        attrs.append(f'fill-opacity="{opacity}"')
    return f"<rect {' '.join(attrs)}/>"


def header(title, sub, page):
    return f"""
  <g id="header">
    {text(56, 54, title, 32, C["text"], "bold")}
    {rect(56, 75, 1168, 46, C["panel2"], C["border"], 8)}
    {text(78, 105, sub, 18, C["sub"])}
  </g>
  <g id="footer">
    {text(1196, 700, f"{page:02d}", 11, C["muted"], anchor="end", family=MONO)}
  </g>
"""


def base(content):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <g id="background">
    <rect x="0" y="0" width="1280" height="720" fill="{C["bg"]}"/>
    <rect x="0" y="0" width="1280" height="8" fill="{C["primary"]}"/>
  </g>
{content}
</svg>
'''


def cards(items, y=165):
    out = ['<g id="cards">']
    for i, (title, body, color) in enumerate(items):
        x = 70 + i * 400
        out.append(rect(x, y, 360, 300, C["panel"], C["border"]))
        out.append(text(x + 28, y + 48, title, 23, color, "bold"))
        out.append(multiline(x + 28, y + 95, body, 18, C["sub"], 32))
    out.append("</g>")
    return "\n".join(out)


def bar_chart(x, y, w, h, title, labels, values, colors, maxv, unit="ms"):
    plot_x = x + 70
    plot_y = y + 80
    plot_w = w - 120
    plot_h = h - 150
    gap = plot_w / len(values)
    out = [f'<g id="chartArea">', rect(x, y, w, h, C["panel"], C["border"]), text(x + 30, y + 42, title, 18, C["sub"])]
    out.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="{C["grid"]}" stroke-width="2"/>')
    out.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="{C["grid"]}" stroke-width="2"/>')
    out.append(f'<!-- chart-plot-area: {plot_x},{plot_y},{plot_x + plot_w},{plot_y + plot_h} -->')
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        bw = min(110, gap * 0.45)
        bx = plot_x + gap * i + gap * 0.28
        bh = plot_h * val / maxv
        by = plot_y + plot_h - bh
        out.append(rect(round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1), color, rx=4))
        out.append(text(round(bx + bw / 2, 1), round(by - 14, 1), f"{val:g}", 18, C["text"], "bold", "middle", MONO))
        out.append(text(round(bx + bw / 2, 1), y + h - 32, label, 14, C["sub"], anchor="middle"))
    out.append(text(x + w - 35, y + 42, unit, 13, C["muted"], anchor="end", family=MONO))
    out.append("</g>")
    return "\n".join(out)


slides = []

slides.append(("01_cover.svg", base(f"""
  <g id="title">
    {text(72, 154, "OSH26 COURSE PROJECT", 18, C["primary2"], "bold")}
    {text(72, 235, "OSH26 Runtime", 58, C["text"], "bold")}
    {text(72, 292, "Android-side Agent Runtime for local LLM deployment", 32, C["sub"], "bold")}
    {multiline(72, 348, ["Built on llama.cpp, but optimized for deployment, memory reuse,", "streaming, cancellation, telemetry, and small-concurrency Agent workloads."], 22, C["sub"], 34)}
  </g>
  <g id="claim">
    {rect(72, 446, 1088, 86, C["panel2"], C["border"], 12)}
    {text(108, 484, "Core claim", 18, C["text"], "bold")}
    {text(108, 517, "Faster first token, reusable memory, concurrent-ready service, and real-device evidence.", 22, C["sub"])}
  </g>
  <g id="metrics">
    {text(78, 625, "9.34x", 28, C["accent"], "bold", family=MONO)}
    {text(78, 655, "Q8 prefill TTFT gain", 14, C["sub"])}
    {text(338, 625, "3.10x", 28, C["accent"], "bold", family=MONO)}
    {text(338, 655, "Prefix cache TTFT gain", 14, C["sub"])}
    {text(622, 625, "8K", 28, C["primary2"], "bold", family=MONO)}
    {text(622, 655, "Validated context", 14, C["sub"])}
    {text(860, 625, "0", 28, C["primary2"], "bold", family=MONO)}
    {text(860, 655, "Hot descriptor alloc", 14, C["sub"])}
  </g>
""")))

slides.append(("02_executive_summary.svg", base(header("We turn llama.cpp from a model runner into an Android Agent Runtime", "The innovation is the endpoint runtime layer: lifecycle, memory reuse, scheduling, and telemetry.", 2) + cards([
    ("1. Android loop", ["JNI and local HTTP service", "Streaming token callback", "cancel / release / health", "OpenAI-style local endpoint"], C["primary2"]),
    ("2. vLLM ideas", ["Prefix KV Cache", "Chunked prefill", "Scheduler Lite", "Small-concurrency latency focus"], C["purple"]),
    ("3. Device data", ["Q8 prefill: 9.34x", "Prefix cache: 3.10x", "8K context validation", "1.7B Q8 model path"], C["accent"]),
] ))))

slides.append(("03_agent_runtime_problem.svg", base(f"""
  <g id="hero">
    {text(90, 140, "PROBLEM FRAMING", 18, C["primary2"], "bold")}
    {multiline(90, 225, ["Endpoint Agent inference is not", "just running a model once"], 44, C["text"], 58, "bold")}
    {multiline(90, 365, ["A deployable runtime must manage streaming, cancellation, long context,", "prompt reuse, concurrent tasks, backend fallback, and health metrics."], 24, C["sub"], 38)}
  </g>
  <g id="runtime-needs">
    {rect(790, 128, 360, 440, C["panel"], C["border"], 12)}
    {text(830, 178, "Runtime questions", 22, C["text"], "bold")}
    {multiline(830, 232, ["1. Is first token fast enough?", "2. Can KV and prompt be reused?", "3. Can small concurrency be controlled?", "4. Can backend failure degrade safely?", "5. Can runtime state be explained?"], 18, C["sub"], 38)}
    {rect(830, 472, 280, 52, C["accent"], C["accent"], 8, "0.16")}
    {text(970, 505, "OSH26 focuses here", 18, C["text"], "bold", "middle")}
  </g>
  <g id="footer">{text(1196, 700, "03", 11, C["muted"], anchor="end", family=MONO)}</g>
""")))

slides.append(("04_why_not_plain_llamacpp.svg", base(header("llama.cpp is the right base, but not the complete Agent framework", "OSH26 adds the endpoint runtime layer instead of rewriting model inference.", 4) + f"""
  <g id="comparison">
    {rect(70, 160, 520, 400, C["panel"], C["border"])}
    {rect(690, 160, 520, 400, C["panel"], C["border"])}
    {text(110, 210, "llama.cpp provides", 26, C["primary2"], "bold")}
    {text(730, 210, "OSH26 Runtime adds", 26, C["accent"], "bold")}
    {multiline(110, 268, ["GGUF model ecosystem", "Tokenizer and sampling base", "CPU backend and C++ portability", "Basic server and CLI", "Model loading and core inference"], 20, C["sub"], 42)}
    {multiline(730, 268, ["Android JNI and HTTP service", "Stream, cancel, release, health", "Prefix KV Cache", "Scheduler Lite", "Adreno-targeted Q8 Vulkan path"], 20, C["sub"], 42)}
  </g>
  <g id="takeaway">
    {rect(190, 602, 900, 54, C["primary"], C["primary"], 8, "0.18")}
    {text(640, 636, "Not rebuilding the wheel: we put the wheel into an Android Agent vehicle.", 20, C["text"], "bold", "middle")}
  </g>
""")))

slides.append(("05_architecture.svg", base(header("OSH26 adds a thin Agent-oriented layer above llama.cpp", "The design keeps llama.cpp as model base and places runtime capabilities around it.", 5) + f"""
  <g id="architecture-flow">
    {rect(58, 210, 155, 90, C["panel"], C["border"])}
    {rect(248, 210, 155, 90, C["panel"], C["border"])}
    {rect(438, 210, 155, 90, C["panel"], C["border"])}
    {rect(628, 210, 155, 90, C["panel"], C["border"])}
    {rect(818, 210, 155, 90, C["panel"], C["border"])}
    {rect(1008, 210, 155, 90, C["panel"], C["border"])}
    {text(135, 246, "Agent / App", 18, C["text"], "bold", "middle")}
    {text(325, 246, "JNI / HTTP", 18, C["text"], "bold", "middle")}
    {text(515, 246, "Scheduler", 18, C["text"], "bold", "middle")}
    {text(705, 246, "Prefix KV", 18, C["text"], "bold", "middle")}
    {text(895, 246, "Backend", 18, C["text"], "bold", "middle")}
    {text(1085, 246, "Stream", 18, C["text"], "bold", "middle")}
    <polygon points="218,255 238,243 238,267" fill="{C["primary2"]}"/>
    <polygon points="408,255 428,243 428,267" fill="{C["primary2"]}"/>
    <polygon points="598,255 618,243 618,267" fill="{C["primary2"]}"/>
    <polygon points="788,255 808,243 808,267" fill="{C["primary2"]}"/>
    <polygon points="978,255 998,243 998,267" fill="{C["primary2"]}"/>
  </g>
  <g id="layers">
    {rect(120, 386, 1040, 84, C["panel2"], C["border"], 8)}
    {text(150, 422, "vLLM-inspired mechanisms", 18, C["purple"], "bold")}
    {text(150, 452, "Prefix cache / KV reuse / chunked prefill / Scheduler Lite", 16, C["sub"])}
    {rect(120, 498, 1040, 84, C["panel"], C["border"], 8)}
    {text(150, 534, "llama.cpp base", 18, C["accent"], "bold")}
    {text(150, 564, "GGUF / tokenizer / model metadata / CPU fallback / sampling primitives", 16, C["sub"])}
  </g>
""")))

slides.append(("06_android_integration.svg", base(header("The Android loop is closed from UI to native generation", "The runtime can be called by real apps or local agents, not only benchmark scripts.", 6) + f"""
  <g id="api-grid">
    {rect(70, 165, 360, 128, C["panel"], C["border"])}
    {rect(470, 165, 360, 128, C["panel"], C["border"])}
    {rect(870, 165, 360, 128, C["panel"], C["border"])}
    {rect(70, 330, 360, 128, C["panel"], C["border"])}
    {rect(470, 330, 360, 128, C["panel"], C["border"])}
    {rect(870, 330, 360, 128, C["panel"], C["border"])}
    {text(100, 210, "loadModel()", 22, C["primary2"], "bold", family=MONO)}
    {text(500, 210, "generateStream()", 22, C["primary2"], "bold", family=MONO)}
    {text(900, 210, "cancel()", 22, C["primary2"], "bold", family=MONO)}
    {text(100, 375, "release()", 22, C["accent"], "bold", family=MONO)}
    {text(500, 375, "getEngineStats()", 22, C["accent"], "bold", family=MONO)}
    {text(900, 375, "/v1/chat", 22, C["purple"], "bold", family=MONO)}
    {text(100, 246, "Load GGUF and return state", 16, C["sub"])}
    {text(500, 246, "Token-by-token callback", 16, C["sub"])}
    {text(900, 246, "Interrupt active generation", 16, C["sub"])}
    {text(100, 411, "Release native resources", 16, C["sub"])}
    {text(500, 411, "Expose TTFT / TPS / memory", 16, C["sub"])}
    {text(900, 411, "OpenAI-style local endpoint", 16, C["sub"])}
  </g>
  <g id="takeaway">{rect(170, 545, 940, 70, C["primary"], C["primary"], 10, "0.18")}{text(640, 588, "Displayable, callable, cancellable, observable: the minimum Agent Runtime loop.", 22, C["text"], "bold", "middle")}</g>
""")))

slides.append(("07_q8_prefill.svg", base(header("Q8 W8A8 prefill cuts first-token latency by 9.34x on device", "F16 prefill is replaced by dynamic Q8 activation quantization and packed Q8 projection GEMM.", 7) + bar_chart(90, 150, 760, 470, "TTFT comparison", ["F16 prefill", "Q8 prefill"], [7361.61, 788.56], [C["warn"], C["accent"]], 8000) + f"""
  <g id="insight">
    {rect(900, 175, 280, 155, C["accent"], C["accent"], 12, "0.16")}
    {text(1040, 245, "9.34x", 54, C["accent"], "bold", "middle", MONO)}
    {text(1040, 285, "TTFT improvement", 18, C["text"], "bold", "middle")}
    {rect(900, 370, 280, 160, C["panel"], C["border"], 12)}
    {text(930, 415, "Why it matters", 18, C["text"], "bold")}
    {multiline(930, 455, ["Interactive agents are", "first-token sensitive.", "This directly improves UX."], 16, C["sub"], 28)}
  </g>
""")))

slides.append(("08_native_q8.svg", base(f"""
  <g id="hero">
    {text(74, 92, "DEPLOYMENT RESULT", 18, C["primary2"], "bold")}
    {text(74, 170, "Native Q8 keeps sub-second TTFT", 44, C["text"], "bold")}
    {text(74, 220, "It also avoids repeated F16-to-Q8 conversion at startup.", 24, C["sub"])}
  </g>
  <g id="kpi">
    {text(110, 405, "717.259 ms", 54, C["accent"], "bold", family=MONO)}
    {text(116, 445, "Average repeated TTFT on native Q8_0 GGUF", 20, C["text"], "bold")}
    {text(116, 485, "Preserves the 0.7 to 0.84 second TTFT range in a deployable model format.", 18, C["sub"])}
  </g>
  {bar_chart(760, 260, 390, 260, "Repeated Native Q8 TTFT", ["avg"], [717], [C["accent"]], 900)}
  <g id="takeaway">{rect(74, 570, 1070, 58, C["primary"], C["primary"], 8, "0.18")}{text(609, 606, "The optimization lands in model format, loading behavior, and runtime execution.", 20, C["text"], "bold", "middle")}</g>
  <g id="footer">{text(1196, 700, "08", 11, C["muted"], anchor="end", family=MONO)}</g>
""")))

slides.append(("09_prefix_cache.svg", base(header("Prefix KV Cache converts repeated Agent prompts into immediate TTFT gains", "System prompts and tool templates repeat frequently in endpoint agents.", 9) + bar_chart(70, 158, 540, 420, "TTFT", ["Cold A", "Repeat A"], [1902.63, 613.66], [C["warn"], C["accent"]], 2000) + bar_chart(670, 158, 540, 420, "TPS", ["Cold A", "Repeat A"], [3.383, 4.734], [C["primary2"], C["accent"]], 5, "TPS") + f"""
  <g id="takeaway">{text(640, 636, "3.10x TTFT gain, 39.9% TPS increase", 30, C["accent"], "bold", "middle", MONO)}</g>
""")))

slides.append(("10_8k_context.svg", base(header("8K context validation extends the runtime toward real Agent memory", "Long context keeps tool instructions, task history, and user preference on device.", 10) + f"""
  <g id="metrics">
    {rect(70, 165, 350, 165, C["panel"], C["border"])}
    {rect(465, 165, 350, 165, C["panel"], C["border"])}
    {rect(860, 165, 350, 165, C["panel"], C["border"])}
    {text(105, 230, "8192", 54, C["primary2"], "bold", family=MONO)}
    {text(500, 230, "460.893", 54, C["accent"], "bold", family=MONO)}
    {text(895, 230, "5488", 54, C["purple"], "bold", family=MONO)}
    {text(105, 275, "validated context tokens", 18, C["sub"])}
    {text(500, 275, "ms exact-repeat TTFT", 18, C["sub"])}
    {text(895, 275, "user tokens prefetched", 18, C["sub"])}
  </g>
  <g id="chunk-flow">
    {rect(95, 440, 1090, 86, C["panel2"], C["border"])}
    {text(130, 475, "Long prompt execution", 18, C["text"], "bold")}
    {multiline(130, 508, ["5488 user tokens are split into 43 chunks of 128 tokens, then valid output is generated.", "This proves the runtime operates beyond the old 1024-token boundary."], 17, C["sub"], 26)}
    {rect(170, 570, 140, 20, C["primary"], rx=4)}
    {rect(320, 570, 140, 20, C["primary"], rx=4, opacity="0.78")}
    {rect(470, 570, 140, 20, C["primary"], rx=4, opacity="0.58")}
    {text(640, 587, "... 43 chunks ...", 16, C["sub"], anchor="middle")}
    {rect(815, 570, 140, 20, C["accent"], rx=4)}
    {rect(965, 570, 100, 20, C["accent"], rx=4, opacity="0.72")}
  </g>
""")))

slides.append(("11_lm_head_descriptor.svg", base(header("Latest LM-head path reaches zero hot descriptor allocation", "Descriptor set cache proves the optimization reaches the Vulkan hot path.", 11) + bar_chart(70, 155, 720, 440, "Median TTFT", ["0.6B-Q8", "1.7B-Q8"], [392.802, 841.991], [C["accent"], C["primary2"]], 900) + f"""
  <g id="kpis">
    {rect(850, 180, 300, 102, C["panel"], C["border"])}
    {rect(850, 314, 300, 102, C["panel"], C["border"])}
    {rect(850, 448, 300, 102, C["panel"], C["border"])}
    {text(890, 222, "0", 36, C["accent"], "bold", family=MONO)}
    {text(940, 222, "descriptor alloc median", 18, C["text"], "bold")}
    {text(890, 356, "2", 36, C["accent"], "bold", family=MONO)}
    {text(940, 356, "submit median", 18, C["text"], "bold")}
    {text(890, 490, "2.2154", 36, C["primary2"], "bold", family=MONO)}
    {text(1018, 490, "TPS on 1.7B", 18, C["text"], "bold")}
  </g>
""")))

slides.append(("12_why_not_opencl.svg", base(header("OpenCL is blocked by Android deployment constraints, not by lack of effort", "The goal is a normal deployable Android App, not a one-off rooted phone demo.", 12) + f"""
  <g id="failure-chain">
    {rect(80, 180, 260, 120, C["panel"], C["border"])}
    {rect(410, 180, 260, 120, C["panel"], C["border"])}
    {rect(740, 180, 260, 120, C["panel"], C["border"])}
    <polygon points="360,240 392,222 392,258" fill="{C["warn"]}"/>
    <polygon points="690,240 722,222 722,258" fill="{C["warn"]}"/>
    {text(110, 225, "libOpenCL.so", 20, C["primary2"], "bold", family=MONO)}
    {text(440, 225, "vendor namespace", 20, C["text"], "bold")}
    {text(770, 225, "dependency chain fails", 20, C["text"], "bold")}
    {multiline(110, 262, ["System vendor library,", "not public app path"], 16, C["sub"], 24)}
    {multiline(440, 262, ["Android linker namespace", "isolates app loading"], 16, C["sub"], 24)}
    {multiline(770, 262, ["libcutils and VNDK", "dependencies still fail"], 16, C["sub"], 24)}
  </g>
  <g id="evidence">
    {rect(120, 390, 1000, 138, C["warn"], C["warn"], 10, "0.14")}
    {text(160, 438, "Measured conclusion", 22, C["text"], "bold")}
    {multiline(160, 480, ["Copying vendor libraries, preloading dependencies, wrapping OpenCL symbols, and android_dlopen_ext did not produce a deployable path.", "Root or Magisk may work, but that violates the course project's deployability target."], 18, C["sub"], 32)}
  </g>
  <g id="takeaway">{text(640, 620, "OpenCL is a documented risk path, not the main deployment route.", 22, C["accent"], "bold", "middle")}</g>
""")))

slides.append(("13_why_not_native_vulkan.svg", base(header("Native ggml Vulkan is too fragile for the target Adreno stack", "Vulkan availability does not guarantee shader and buffer correctness on each Android GPU.", 13) + f"""
  <g id="timeline">
    {rect(70, 170, 1140, 350, C["panel"], C["border"])}
    <line x1="170" y1="350" x2="1070" y2="350" stroke="{C["grid"]}" stroke-width="3"/>
    <circle cx="210" cy="350" r="13" fill="{C["primary2"]}"/>
    <circle cx="470" cy="350" r="13" fill="{C["primary2"]}"/>
    <circle cx="730" cy="350" r="13" fill="{C["warn"]}"/>
    <circle cx="990" cy="350" r="13" fill="{C["warn"]}"/>
    {text(165, 290, "Build passes", 19, C["text"], "bold")}
    {text(405, 290, "Init succeeds", 19, C["text"], "bold")}
    {text(665, 290, "Version gap", 19, C["text"], "bold")}
    {text(920, 290, "Corrupted output", 19, C["text"], "bold")}
    {text(135, 405, "vendored headers", 15, C["sub"])}
    {text(390, 405, "Adreno 650 detected", 15, C["sub"])}
    {text(650, 405, "ggml-vulkan leans Vulkan 1.2", 15, C["sub"])}
    {text(895, 405, "shader / buffer correctness fails", 15, C["sub"])}
  </g>
  <g id="decision">{rect(160, 560, 960, 62, C["primary"], C["primary"], 8, "0.18")}{text(640, 599, "OSH26 chooses controlled Vulkan 1.1 kernels, correctness gates, and CPU fallback.", 21, C["text"], "bold", "middle")}</g>
""")))

slides.append(("14_alternatives_matrix.svg", base(header("OSH26 fills the gap between model libraries, server runtimes, and mobile graph engines", "The comparison criterion is endpoint Agent deployment, not popularity.", 14) + f"""
  <g id="chartArea">
    {rect(70, 155, 1140, 430, C["panel"], C["border"])}
    <!-- chart-plot-area: 70,155,1210,585 -->
    <line x1="70" y1="225" x2="1210" y2="225" stroke="{C["grid"]}" stroke-width="2"/>
    <line x1="70" y1="305" x2="1210" y2="305" stroke="{C["border"]}" stroke-width="1"/>
    <line x1="70" y1="385" x2="1210" y2="385" stroke="{C["border"]}" stroke-width="1"/>
    <line x1="70" y1="465" x2="1210" y2="465" stroke="{C["border"]}" stroke-width="1"/>
    <line x1="300" y1="155" x2="300" y2="585" stroke="{C["grid"]}" stroke-width="2"/>
    <line x1="520" y1="155" x2="520" y2="585" stroke="{C["border"]}" stroke-width="1"/>
    <line x1="740" y1="155" x2="740" y2="585" stroke="{C["border"]}" stroke-width="1"/>
    <line x1="960" y1="155" x2="960" y2="585" stroke="{C["border"]}" stroke-width="1"/>
    {text(155, 200, "Framework", 18, C["text"], "bold", "middle")}
    {text(410, 200, "Strength", 18, C["text"], "bold", "middle")}
    {text(630, 200, "Gap", 18, C["text"], "bold", "middle")}
    {text(850, 200, "OSH26 use", 18, C["text"], "bold", "middle")}
    {text(1080, 200, "Agent fit", 18, C["text"], "bold", "middle")}
    {text(155, 272, "llama.cpp", 18, C["primary2"], "bold", "middle", MONO)}
    {text(155, 352, "vLLM", 18, C["purple"], "bold", "middle", MONO)}
    {text(155, 432, "MNN", 18, C["accent"], "bold", "middle", MONO)}
    {text(155, 512, "OSH26", 18, C["text"], "bold", "middle", MONO)}
    {text(410, 272, "GGUF / C++ / CPU", 15, C["sub"], anchor="middle")}
    {text(630, 272, "Agent lifecycle", 15, C["sub"], anchor="middle")}
    {text(850, 272, "Model base", 15, C["sub"], anchor="middle")}
    {text(1080, 272, "Medium", 20, C["accent"], "bold", "middle")}
    {text(410, 352, "Paged / batching", 15, C["sub"], anchor="middle")}
    {text(630, 352, "CUDA / Python / server", 15, C["sub"], anchor="middle")}
    {text(850, 352, "Borrow ideas", 15, C["sub"], anchor="middle")}
    {text(1080, 352, "Medium", 20, C["accent"], "bold", "middle")}
    {text(410, 432, "Mobile operators", 15, C["sub"], anchor="middle")}
    {text(630, 432, "Not Agent-centered", 15, C["sub"], anchor="middle")}
    {text(850, 432, "Operator reference", 15, C["sub"], anchor="middle")}
    {text(1080, 432, "Low-Mid", 20, C["primary2"], "bold", "middle")}
    {text(410, 512, "Deploy + cache + schedule", 15, C["sub"], anchor="middle")}
    {text(630, 512, "Prototype, extensible", 15, C["sub"], anchor="middle")}
    {text(850, 512, "Endpoint Agent runtime", 15, C["sub"], anchor="middle")}
    {text(1080, 512, "High", 20, C["accent"], "bold", "middle")}
  </g>
""")))

slides.append(("15_novelty.svg", base(header("Novelty comes from system integration plus endpoint-specific optimization", "The work is in real links, real devices, and real metrics.", 15) + cards([
    ("Android service layer", ["JNI / HTTP", "stream / cancel", "health telemetry"], C["primary2"]),
    ("vLLM-style memory", ["Prefix KV Cache", "paged-style reuse", "chunked prefill"], C["purple"]),
    ("Q8 Vulkan path", ["Q8 W8A8 prefill", "LM-head timing", "descriptor cache"], C["accent"]),
], 155) + cards([
    ("8K context", ["8192-token context", "5488-token prefill", "valid generation"], C["primary2"]),
    ("Benchmark gates", ["TTFT / TPS / submit", "descriptor allocation", "correctness checks"], C["accent"]),
    ("Failure analysis", ["OpenCL namespace", "Adreno Vulkan correctness", "fallback strategy"], C["warn"]),
], 385))))

slides.append(("16_closing.svg", base(f"""
  <g id="closing-message">
    {text(90, 150, "FINAL ANSWER", 18, C["primary2"], "bold")}
    {multiline(90, 230, ["OSH26 Runtime makes", "local Android Agents deployable"], 44, C["text"], 58, "bold")}
    {multiline(90, 365, ["Faster first token, reusable memory, small-concurrency readiness,", "and explainable runtime state make it more than a llama.cpp wrapper."], 24, C["sub"], 38)}
  </g>
  <g id="proof-row">
    {rect(92, 498, 250, 90, C["panel"], C["border"])}
    {rect(372, 498, 250, 90, C["panel"], C["border"])}
    {rect(652, 498, 250, 90, C["panel"], C["border"])}
    {rect(932, 498, 250, 90, C["panel"], C["border"])}
    {text(122, 540, "9.34x", 30, C["accent"], "bold", family=MONO)}
    {text(402, 540, "3.10x", 30, C["accent"], "bold", family=MONO)}
    {text(682, 540, "8K", 30, C["primary2"], "bold", family=MONO)}
    {text(962, 540, "1.7B", 30, C["primary2"], "bold", family=MONO)}
    {text(122, 568, "Q8 prefill", 14, C["sub"])}
    {text(402, 568, "Prefix cache", 14, C["sub"])}
    {text(682, 568, "Context", 14, C["sub"])}
    {text(962, 568, "Q8 model validation", 14, C["sub"])}
  </g>
  <g id="footer">
    {text(90, 700, "Thank you", 11, C["muted"])}
    {text(1196, 700, "16", 11, C["muted"], anchor="end", family=MONO)}
  </g>
""")))

for name, content in slides:
    (OUT / name).write_text(content, encoding="utf-8")

notes = ROOT / "notes" / "total.md"
notes.write_text("\n\n".join(f"# {name[:-4]}\n\n{idx+1}. Speaker note: explain the slide conclusion first, then connect it to the defense claim that OSH26 is an endpoint Agent Runtime rather than a plain llama.cpp wrapper." for idx, (name, _) in enumerate(slides)), encoding="utf-8")
