from pathlib import Path
from html import escape
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "svg_output"
FINAL = ROOT / "svg_final"
NOTES = ROOT / "notes"
for d in (OUT, FINAL, NOTES):
    d.mkdir(exist_ok=True)

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
MONO = "Consolas, Courier New, monospace"


def t(x, y, s, size=18, color=None, weight=None, anchor=None, family=None):
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'font-family="{family or FONT}"',
        f'font-size="{size}"',
        f'fill="{color or C["sub"]}"',
    ]
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    return f"<text {' '.join(attrs)}>{escape(str(s))}</text>"


def ml(x, y, lines, size=18, color=None, gap=28, weight=None, family=None):
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'font-family="{family or FONT}"',
        f'font-size="{size}"',
        f'fill="{color or C["sub"]}"',
    ]
    if weight:
        attrs.append(f'font-weight="{weight}"')
    spans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else gap
        spans.append(f'<tspan x="{x}" dy="{dy}">{escape(str(line))}</tspan>')
    return f"<text {' '.join(attrs)}>{''.join(spans)}</text>"


def rect(x, y, w, h, fill, stroke=None, rx=8, opacity=None):
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'width="{w}"',
        f'height="{h}"',
        f'rx="{rx}"',
        f'fill="{fill}"',
    ]
    if stroke:
        attrs.append(f'stroke="{stroke}"')
    if opacity:
        attrs.append(f'fill-opacity="{opacity}"')
    return f"<rect {' '.join(attrs)}/>"


def line(x1, y1, x2, y2, color=None, width=2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or C["grid"]}" stroke-width="{width}"/>'


def arrow(x1, y1, x2, y2, color=None):
    color = color or C["primary2"]
    mid = (x1 + x2) / 2
    return (
        line(x1, y1, x2, y2, color, 3)
        + f'<polygon points="{x2},{y2} {x2-12},{y2-7} {x2-12},{y2+7}" fill="{color}"/>'
        + f'<circle cx="{mid}" cy="{y1}" r="3" fill="{color}"/>'
    )


def base(content):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <rect x="0" y="0" width="1280" height="720" fill="{C["bg"]}"/>
  <rect x="0" y="0" width="1280" height="8" fill="{C["primary"]}"/>
  {content}
</svg>
'''


def header(title, subtitle, page):
    return f"""
  {t(56, 54, title, 31, C["text"], "bold")}
  {rect(56, 76, 1168, 42, C["panel2"], C["border"], 8)}
  {t(78, 104, subtitle, 17, C["sub"])}
  {t(1196, 700, f"{page:02d}", 11, C["muted"], anchor="end", family=MONO)}
"""


def bullet_card(x, y, w, h, title, bullets, color, title_size=22, body_size=17):
    out = [rect(x, y, w, h, C["panel"], C["border"], 8)]
    out.append(t(x + 24, y + 42, title, title_size, color, "bold"))
    out.append(ml(x + 24, y + 82, bullets, body_size, C["sub"], 28))
    return "\n".join(out)


def kpi(x, y, value, label, color, w=250):
    return (
        rect(x, y, w, 92, C["panel"], C["border"], 8)
        + t(x + 24, y + 43, value, 31, color, "bold", family=MONO)
        + t(x + 24, y + 72, label, 14, C["sub"])
    )


def mini_bar(x, y, w, h, labels, values, colors, maxv, unit="ms"):
    plot_x, plot_y = x + 52, y + 60
    plot_w, plot_h = w - 92, h - 108
    gap = plot_w / max(1, len(values))
    out = [rect(x, y, w, h, C["panel"], C["border"], 8)]
    out.append(line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, C["grid"], 2))
    out.append(line(plot_x, plot_y, plot_x, plot_y + plot_h, C["grid"], 2))
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        bw = min(104, gap * 0.46)
        bx = plot_x + i * gap + gap * 0.27
        bh = max(4, plot_h * val / maxv)
        by = plot_y + plot_h - bh
        out.append(rect(round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1), color, rx=4))
        out.append(t(round(bx + bw / 2, 1), round(by - 12, 1), f"{val:g}", 15, C["text"], "bold", "middle", MONO))
        out.append(t(round(bx + bw / 2, 1), y + h - 28, label, 13, C["sub"], anchor="middle"))
    out.append(t(x + w - 24, y + 34, unit, 12, C["muted"], anchor="end", family=MONO))
    return "\n".join(out)


slides = []
notes = []


def add(name, page, title, svg, note):
    slides.append((name, base(svg)))
    notes.append((name[:-4], page, title, note))


add("01_cover.svg", 1, "OSH26 Runtime",
    f"""
  {t(72, 148, "ANDROID AGENT RUNTIME", 18, C["primary2"], "bold")}
  {t(72, 232, "OSH26 Runtime", 58, C["text"], "bold")}
  {t(72, 292, "端侧 AI 框架：把本地 LLM 变成 Android 上可部署的系统服务", 30, C["sub"], "bold")}
  {ml(72, 356, ["不是简单套壳 llama.cpp，而是在 Android 资源、驱动、调度、内存约束下重建端侧 Agent Runtime。", "我们用实机数据、失败路径和系统机制证明：端侧部署本身就是 OS 级工程。"], 21, C["sub"], 34)}
  {rect(72, 474, 1088, 86, C["panel2"], C["border"], 10)}
  {t(108, 512, "核心结论", 18, C["text"], "bold")}
  {t(108, 544, "端侧 Agent 的瓶颈不是“有没有模型”，而是 first token、KV 管理、并发调度和驱动可控性。", 22, C["text"], "bold")}
  {kpi(78, 610, "9.34x", "Q8 prefill TTFT", C["accent"], 210)}
  {kpi(330, 610, "77.36%", "1.7B stateless cache", C["accent"], 250)}
  {kpi(622, 610, "8K", "validated context", C["primary2"], 190)}
  {kpi(854, 610, "3.06", "TPS on 1.7B Q8", C["primary2"], 230)}
""", "开场直接讲：我们的贡献不是模型算法，而是把本地模型做成 Android 上可部署、可观测、可调度的端侧系统。")

add("02_endpoint_importance.svg", 2, "为什么端侧部署重要",
    header("端侧 Agent 的价值：离线、隐私、低时延、可控执行", "端侧部署不是云端推理的降级版，而是移动智能体真正进入系统能力层的前提。", 2) + f"""
  {bullet_card(70, 158, 350, 220, "用户体验", ["弱网/断网仍可响应", "首 token 决定交互手感", "工具执行不被云端往返拖慢"], C["primary2"])}
  {bullet_card(465, 158, 350, 220, "隐私与权限", ["通讯录、短信、日历留在设备", "系统权限由本地 Runtime 管控", "高风险操作可本地确认"], C["purple"])}
  {bullet_card(860, 158, 350, 220, "系统可控性", ["可观测 health 指标", "可取消、可释放、可限流", "失败路径不依赖云端重试"], C["accent"])}
  {rect(118, 456, 1044, 108, C["panel2"], C["border"], 8)}
  {t(160, 498, "OS 视角", 22, C["text"], "bold")}
  {ml(160, 536, ["端侧 AI 不是一个函数调用，而是一个常驻服务：它要管理内存工作集、异步请求、驱动差异、队列背压和错误恢复。"], 21, C["sub"], 32)}
""", "把端侧部署的价值讲清楚：不是为了炫技，而是 Agent 要接触本地系统能力，云端不能替代。")

add("03_runtime_problem.svg", 3, "端侧推理是 Runtime 问题",
    header("端侧推理是 Runtime 问题，不是单次模型调用问题", "Agent 会反复调用模型、工具和 subagent；系统必须管理长期状态。", 3) + f"""
  {rect(74, 156, 1132, 390, C["panel"], C["border"], 8)}
  {t(120, 210, "一次 chat completion", 24, C["muted"], "bold")}
  {t(120, 340, "端侧 Agent Runtime", 30, C["text"], "bold")}
  {arrow(360, 204, 515, 204, C["muted"])}
  {arrow(360, 334, 515, 334, C["accent"])}
  {bullet_card(545, 168, 250, 110, "单请求", ["tokenize", "prefill", "decode"], C["muted"], 18, 15)}
  {bullet_card(545, 298, 250, 190, "系统服务", ["流式输出与取消", "KV cache 生命周期", "请求队列与调度", "驱动 fallback 与 telemetry"], C["accent"], 18, 15)}
  {bullet_card(835, 168, 300, 320, "如果没有 Runtime", ["重复 prefill 固定 prompt", "一次长 prompt 占满内存", "GPU/CPU 复制不可控", "driver fail 后无诊断", "并发 subagent 互相污染上下文", "UI 卡死或无法取消"], C["warn"], 20, 16)}
  {t(640, 622, "我们做的是 Android 上的“小型 LLM 操作系统层”。", 28, C["accent"], "bold", "middle")}
""", "这一页把问题从模型拉回系统：端侧部署要有调度、内存、可观测和恢复机制。")

add("04_llamacpp_boundary.svg", 4, "与 llama.cpp 的边界",
    header("llama.cpp 是模型底座，OSH26 是端侧 Agent Runtime 层", "我们复用 GGUF/tokenizer/sampler；重写 Android 端最脆弱的系统边界。", 4) + f"""
  {rect(78, 164, 510, 350, C["panel"], C["border"], 8)}
  {rect(692, 164, 510, 350, C["panel"], C["border"], 8)}
  {t(120, 214, "llama.cpp 给我们什么", 26, C["primary2"], "bold")}
  {ml(120, 274, ["GGUF 模型生态", "Tokenizer / Sampler", "CPU fallback 与基础推理", "跨平台 C/C++ 工程基础"], 21, C["sub"], 42)}
  {t(734, 214, "OSH26 补上什么", 26, C["accent"], "bold")}
  {ml(734, 274, ["Android JNI / HTTP service", "stream / cancel / release / health", "Prefix KV Cache 与请求调度", "Adreno 定制 Q8 Vulkan 后端", "性能门控与实机验证脚本"], 21, C["sub"], 42)}
  {rect(180, 580, 920, 58, C["primary"], C["primary"], 8, "0.18")}
  {t(640, 616, "区分度：不是替代 llama.cpp，而是把它变成可部署的端侧 Agent 系统。", 22, C["text"], "bold", "middle")}
""", "强调边界：我们不是重复造 tokenizer 或模型格式，而是围绕 Android Agent 场景做系统层增强。")

add("05_git_roadmap.svg", 5, "优化路线图",
    header("git 历史显示：这是连续系统优化，不是一页包装", "从 CPU demo 到 Agent cache，每一步都针对一个实际端侧瓶颈。", 5) + f"""
  {rect(72, 158, 1136, 410, C["panel"], C["border"], 8)}
  {line(126, 355, 1140, 355, C["grid"], 4)}
  {''.join(f'<circle cx="{x}" cy="355" r="10" fill="{color}"/>' for x, color in [(150,C["muted"]),(270,C["warn"]),(390,C["primary2"]),(510,C["purple"]),(630,C["accent"]),(750,C["accent"]),(870,C["primary2"]),(990,C["primary2"]),(1110,C["accent"])])}
  {t(118, 305, "CPU", 18, C["sub"], "bold")}
  {t(238, 305, "Vulkan 乱码", 18, C["warn"], "bold")}
  {t(345, 305, "GPU runtime", 18, C["primary2"], "bold")}
  {t(474, 305, "KV cache", 18, C["purple"], "bold")}
  {t(590, 305, "Q8 prefill", 18, C["accent"], "bold")}
  {t(708, 305, "原生 Q8", 18, C["accent"], "bold")}
  {t(835, 305, "8K context", 18, C["primary2"], "bold")}
  {t(940, 305, "1.7B/单份驻留", 18, C["primary2"], "bold")}
  {t(1062, 305, "Agent cache", 18, C["accent"], "bold")}
  {ml(102, 415, ["Android Studio/CPU 推理", "OpenAI API", "JNI 闭环"], 14, C["sub"], 22)}
  {ml(222, 415, ["原生 Vulkan 可加载", "但输出乱码", "驱动正确性失败"], 14, C["sub"], 22)}
  {ml(342, 415, ["手写 shader", "matmul 并行", "GPU 预填充"], 14, C["sub"], 22)}
  {ml(462, 415, ["LRU 管理", "prefix 匹配", "paged restore"], 14, C["sub"], 22)}
  {ml(582, 415, ["W8A8", "LM-head", "submit 降低"], 14, C["sub"], 22)}
  {ml(702, 415, ["Q8_0 GGUF", "避免转换", "decode Q8"], 14, C["sub"], 22)}
  {ml(822, 415, ["8192 ctx", "5488 tokens", "分块 prefill"], 14, C["sub"], 22)}
  {ml(942, 415, ["消除 CPU 重复驻留", "启用 Qwen3-1.7B"], 14, C["sub"], 22)}
  {ml(1062, 415, ["stateless subagent", "pinned prefix", "chat 隔离"], 14, C["sub"], 22)}
  {t(640, 626, "这条路线对应真实 commit：每一步都修掉一个 Android 端侧系统问题。", 22, C["text"], "bold", "middle")}
""", "这一页直接引用 git 历史脉络，告诉老师我们做了很多系统工程，不是只做页面包装。")

add("06_architecture.svg", 6, "总体架构",
    header("总体架构：模型底座 + Android 服务 + OSH26 GPU Runtime", "上层给 Agent 用，下层对 Android 驱动和资源负责。", 6) + f"""
  {bullet_card(74, 170, 230, 126, "Agent / App", ["OpenAI-style API", "streaming UI"], C["primary2"], 19, 15)}
  {bullet_card(344, 170, 230, 126, "JNI / HTTP", ["load/generate/cancel", "health stats"], C["primary2"], 19, 15)}
  {bullet_card(614, 170, 230, 126, "Scheduler Lite", ["queue/backpressure", "prefix-aware aging"], C["purple"], 19, 15)}
  {bullet_card(884, 170, 230, 126, "Prefix KV Cache", ["pinned/dynamic", "paged reuse"], C["purple"], 19, 15)}
  {arrow(304, 233, 337, 233)}{arrow(574, 233, 607, 233)}{arrow(844, 233, 877, 233)}
  {bullet_card(190, 390, 330, 132, "llama.cpp base", ["GGUF / tokenizer / sampler", "CPU fallback / correctness gate"], C["accent"], 21, 16)}
  {bullet_card(590, 390, 500, 132, "OSH26 Vulkan backend", ["Q8 W8A8 prefill / decode GEMV / LM head top-k", "descriptor cache / submit gate / Adreno Vulkan 1.1"], C["accent"], 21, 16)}
  {arrow(520, 456, 583, 456, C["accent"])}
  {rect(150, 580, 980, 54, C["panel2"], C["border"], 8)}
  {t(640, 615, "服务层、内存层、驱动层三者一起优化，才有端侧 Agent 的可部署性。", 20, C["text"], "bold", "middle")}
""", "把系统拆开：上层是 Agent 可调用的服务，下层是可控的 GPU/CPU 执行和 KV 内存。")

add("07_android_lifecycle.svg", 7, "Android 生命周期闭环",
    header("Android 闭环：从 UI 输入到 native 流式输出已经打通", "端侧框架必须处理生命周期，而不是只跑 benchmark。", 7) + f"""
  {bullet_card(72, 160, 330, 145, "模型生命周期", ["loadModel(path, backend)", "release() 清理 native/GPU 资源", "reset_cache() 隔离测试和会话"], C["primary2"], 20, 16)}
  {bullet_card(474, 160, 330, 145, "生成生命周期", ["generateStream()", "token callback 实时回 UI", "cancel() 中断 active/pending 请求"], C["accent"], 20, 16)}
  {bullet_card(876, 160, 330, 145, "服务生命周期", ["/v1/chat/completions", "/health telemetry", "队列、错误、最后 token ids 可见"], C["purple"], 20, 16)}
  {rect(112, 388, 1056, 130, C["panel"], C["border"], 8)}
  {t(152, 430, "OS 知识嵌入", 23, C["text"], "bold")}
  {ml(152, 470, ["请求状态是共享内核对象：active request、pending queue、cancel flag、shutdown flag 都需要锁和条件变量保护。", "UI 不应该直接等待 native 长任务；流式回调和 cancel 是端侧交互的基本系统能力。"], 19, C["sub"], 31)}
  {t(640, 622, "我们交付的是可交互 Runtime，不是离线脚本。", 27, C["accent"], "bold", "middle")}
""", "强调 Android 集成很实际：UI、JNI、native、HTTP、取消、释放、健康状态都在同一个闭环。")

add("08_opencl_failure.svg", 8, "为什么不用 OpenCL",
    header("OpenCL 不是我们没试，而是 Android 部署边界不允许", "正常 App 不能依赖 root/Magisk 级别的 vendor namespace 绕过。", 8) + f"""
  {bullet_card(80, 170, 300, 140, "尝试路径", ["查找 vendor libOpenCL.so", "复制/预加载依赖", "android_dlopen_ext 包装"], C["primary2"], 20, 16)}
  {bullet_card(490, 170, 300, 140, "失败原因", ["Android linker namespace 隔离", "DT_NEEDED 依赖仍不可见", "VNDK/vendor/system 边界复杂"], C["warn"], 20, 16)}
  {bullet_card(900, 170, 300, 140, "工程判断", ["不能要求 root", "不能靠机型私有路径", "不能把 demo 建在不可部署假设上"], C["accent"], 20, 16)}
  {arrow(385, 240, 482, 240, C["warn"])}{arrow(795, 240, 892, 240, C["accent"])}
  {rect(120, 414, 1040, 110, C["warn"], C["warn"], 8, "0.14")}
  {t(160, 456, "结论", 23, C["text"], "bold")}
  {t(160, 494, "OpenCL 是移动端驱动碎片化和 Android 动态链接策略的系统失败，不是一个 kernel 写没写好的问题。", 20, C["sub"])}
  {t(640, 620, "所以我们走 Vulkan 1.1 可控路径。", 28, C["accent"], "bold", "middle")}
""", "讲 OpenCL 的失败时要讲系统边界：linker namespace、vendor 依赖、部署性，不是单纯性能选择。")

add("09_native_vulkan_failure.svg", 9, "为什么不用原生 ggml Vulkan",
    header("原生 ggml Vulkan 在目标 Adreno 栈上太脆弱", "能初始化不等于能稳定生成；驱动语义和 shader 正确性才是关键。", 9) + f"""
  {rect(90, 170, 1100, 320, C["panel"], C["border"], 8)}
  {line(160, 330, 1100, 330, C["grid"], 4)}
  {''.join(f'<circle cx="{x}" cy="330" r="12" fill="{color}"/>' for x, color in [(190,C["primary2"]),(430,C["primary2"]),(670,C["warn"]),(910,C["warn"])])}
  {t(140, 282, "编译接入", 19, C["primary2"], "bold")}
  {t(375, 282, "设备初始化", 19, C["primary2"], "bold")}
  {t(600, 282, "能力假设冲突", 19, C["warn"], "bold")}
  {t(840, 282, "输出乱码/错误", 19, C["warn"], "bold")}
  {ml(130, 370, ["适配 Android", "链接 Vulkan"], 15, C["sub"], 22)}
  {ml(365, 370, ["Adreno 650", "Vulkan 1.1"], 15, C["sub"], 22)}
  {ml(590, 370, ["ggml-vulkan 偏桌面/新 API", "移动端同步和 buffer 语义不同"], 15, C["sub"], 22)}
  {ml(830, 370, ["降低版本检查后仍不可靠", "不能作为 Agent Runtime 基座"], 15, C["sub"], 22)}
  {rect(160, 555, 960, 62, C["primary"], C["primary"], 8, "0.18")}
  {t(640, 594, "我们重写的是可控执行边界：Vulkan 1.1 kernel + correctness gate + health telemetry。", 21, C["text"], "bold", "middle")}
""", "强调原生 Vulkan 的风险：不是跑不起来那么简单，而是跑起来后输出不可信。")

add("10_custom_gpu_runtime.svg", 10, "自研 GPU Runtime",
    header("自研 OSH26 GPU Runtime：面向 Adreno 的最小可控后端", "保留 llama.cpp 生态，替换 Android 上不可控的 GPU 执行边界。", 10) + f"""
  {bullet_card(70, 160, 350, 210, "算子路径", ["Q8 prefill GEMM", "decode Q8 GEMV", "GPU LM head top-k", "RMS / projection 辅助 kernel"], C["accent"])}
  {bullet_card(465, 160, 350, 210, "驱动路径", ["Vulkan 1.1", "手写 descriptor / pipeline", "timestamp query", "submit count gate"], C["primary2"])}
  {bullet_card(860, 160, 350, 210, "正确性路径", ["debug correctness 开关", "top-k overlap 检查", "logits sanity", "CPU fallback 隔离"], C["purple"])}
  {rect(108, 458, 1064, 98, C["panel2"], C["border"], 8)}
  {t(148, 500, "区分度", 23, C["text"], "bold")}
  {t(148, 536, "llama.cpp 给模型生态；OSH26 给 Android 端稳定、可测、可调优的 GPU Runtime。", 22, C["sub"])}
  {t(640, 632, "这就是我们“重写后端”的原因。", 28, C["accent"], "bold", "middle")}
""", "这页正面讲重写后端：不是为了炫耀，而是原生路径在目标设备上不可控。")

add("11_memory_architecture.svg", 11, "统一内存架构",
    header("统一内存架构：CPU 管控制面，GPU 承接数据面", "移动端不是无限显存服务器；需要把工作集、复制、驻留和恢复都显式管理。", 11) + f"""
  {rect(84, 170, 480, 330, C["panel"], C["border"], 8)}
  {rect(716, 170, 480, 330, C["panel"], C["border"], 8)}
  {t(125, 218, "CPU 控制面", 26, C["primary2"], "bold")}
  {ml(125, 278, ["tokenize / sampler", "请求队列与状态机", "cache policy / LRU", "debug correctness", "HTTP / JNI 生命周期"], 20, C["sub"], 39)}
  {t(756, 218, "GPU 数据面", 26, C["accent"], "bold")}
  {ml(756, 278, ["Q8 权重常驻", "prefill / decode 投影", "packed FP16 KV", "LM head top-k", "timestamp 计时"], 20, C["sub"], 39)}
  {arrow(570, 335, 704, 335, C["accent"])}
  {rect(190, 570, 900, 58, C["panel2"], C["border"], 8)}
  {t(640, 606, "OS 类比：CPU 是调度器，GPU 是加速设备，KV/权重是长期驻留工作集。", 20, C["text"], "bold", "middle")}
""", "把内存架构讲成 OS 课知识：控制面/数据面，工作集，设备驻留，减少复制。")

add("12_kv_cache_os.svg", 12, "KV Cache 管理",
    header("KV Cache 管理：把上下文当成虚拟内存页缓存", "受 vLLM 启发，但实现目标是 Android 小并发、低 TTFT、可恢复。", 12) + f"""
  {rect(82, 158, 1116, 380, C["panel"], C["border"], 8)}
  {t(128, 210, "Prompt tokens", 18, C["primary2"], "bold", family=MONO)}
  {''.join(rect(128 + i*42, 245, 32, 32, C["primary2"] if i < 9 else C["muted"], rx=4, opacity="0.8" if i < 9 else "0.35") for i in range(22))}
  {t(128, 335, "KV pages", 18, C["purple"], "bold", family=MONO)}
  {''.join(rect(128 + i*86, 370, 70, 44, C["purple"] if i < 5 else C["panel2"], C["border"], 5, "0.76" if i < 5 else None) for i in range(11))}
  {t(128, 486, "Policy", 18, C["accent"], "bold", family=MONO)}
  {t(230, 486, "page pool + reusable prefix tokens + pinned/dynamic entries + eviction/fragmentation stats", 19, C["sub"])}
  {bullet_card(760, 205, 350, 130, "OS 对应", ["页缓存：复用 prefix", "页表：token->KV page", "LRU：淘汰动态 entry"], C["accent"], 20, 15)}
  {bullet_card(760, 365, 350, 130, "工程收益", ["跳过重复 prefill", "恢复 KV page", "减少 TTFT 和功耗"], C["primary2"], 20, 15)}
  {t(640, 622, "KV Cache 不只是大数组，而是 Runtime 管理的内存对象。", 26, C["accent"], "bold", "middle")}
""", "这一页把 KV cache 和 OS 里的虚拟内存/页缓存类比起来，是课程知识的自然融入。")

add("13_stateless_subagent.svg", 13, "Agent 专用 Prefix Cache",
    header("stateless subagent：固定协议 prompt 进入 pinned prefix", "Agent 工作负载和普通聊天不同：协议重复、任务短、并发多、上下文不应串扰。", 13) + f"""
  {mini_bar(70, 160, 520, 340, ["cold", "warm"], [5787.45, 1310.29], [C["warn"], C["accent"]], 6000, "ms")}
  {rect(640, 160, 540, 340, C["panel"], C["border"], 8)}
  {t(680, 208, "1.7B-Q8_0 stateless subagent", 24, C["text"], "bold")}
  {ml(680, 262, ["固定 YAML action flow / tool:return 协议", "112 tokens pinned prefix reused", "7/7 warm cache hits", "warm TTFT reduction: 77.36%", "普通 chat cache key 分离，避免污染"], 20, C["sub"], 39)}
  {rect(130, 545, 1020, 66, C["primary"], C["primary"], 8, "0.18")}
  {t(640, 586, "OS 类比：共享只读代码段 + 私有进程栈；协议共享，任务隔离。", 22, C["text"], "bold", "middle")}
""", "这页体现针对 Agent 做了专门优化，不是通用聊天缓存。")

add("14_scheduler.svg", 14, "请求调度",
    header("请求调度：cache-aware aging，比 FIFO 更适合端侧 Agent", "小并发下，调度目标是缩短 TTFT 同时避免 cache miss 请求饥饿。", 14) + f"""
  {bullet_card(78, 162, 350, 220, "队列对象", ["active request", "pending queue", "cancel flag", "shutdown flag", "queue depth peak"], C["primary2"])}
  {bullet_card(465, 162, 350, 220, "调度策略", ["prefix-aware-aging-v2", "cache hit 低服务时间", "aging 防止饥饿", "max pending 背压"], C["purple"])}
  {bullet_card(852, 162, 350, 220, "系统效果", ["UI 可取消", "请求不会无限堆积", "健康状态可解释", "Agent 小并发更稳"], C["accent"])}
  {rect(145, 470, 990, 96, C["panel2"], C["border"], 8)}
  {t(185, 512, "OS 知识嵌入", 23, C["text"], "bold")}
  {t(185, 548, "短作业优先、aging、防饥饿、背压、临界区同步都落在 native runtime 的请求状态管理里。", 20, C["sub"])}
  {t(640, 636, "模型服务也需要调度器。", 28, C["accent"], "bold", "middle")}
""", "把调度讲实：不是只有 KV cache，还有队列、取消、背压、aging。")

add("15_q8_prefill.svg", 15, "Q8 Prefill",
    header("Q8 W8A8 prefill：把 first-token 临界路径打下来", "Prefill 是 TTFT 大头；端侧 Agent 首 token 慢，用户就会认为系统卡住。", 15) + f"""
  {mini_bar(80, 152, 700, 410, ["F16 prefill", "Q8 prefill"], [7361.61, 788.56], [C["warn"], C["accent"]], 8000, "ms")}
  {rect(840, 190, 300, 130, C["accent"], C["accent"], 10, "0.16")}
  {t(990, 255, "9.34x", 56, C["accent"], "bold", "middle", MONO)}
  {t(990, 296, "TTFT improvement", 18, C["text"], "bold", "middle")}
  {bullet_card(840, 365, 300, 146, "做了什么", ["activation dynamic Q8", "packed int8 dot", "F32 accumulation", "性能 gate 验证"], C["primary2"], 20, 15)}
  {t(640, 632, "优化的是 first-token 临界路径，不只是 kernel microbenchmark。", 25, C["accent"], "bold", "middle")}
""", "用实测数据讲 Q8 prefill。注意强调 first-token，而不是泛泛讲吞吐。")

add("16_decode_gemv.svg", 16, "Decode Q8 GEMV",
    header("decode Q8 GEMV：让生成阶段也吃到量化收益", "prefill 解决 TTFT，decode 决定持续流式输出的 TPS。", 16) + f"""
  {mini_bar(70, 160, 530, 360, ["baseline", "GEMV"], [4.78, 5.7498], [C["primary2"], C["accent"]], 6.2, "TPS")}
  {mini_bar(680, 160, 530, 360, ["baseline", "GEMV"], [2.2154, 3.0408], [C["primary2"], C["accent"]], 3.3, "TPS")}
  {t(335, 560, "0.6B-Q8_0: +20.3%", 23, C["accent"], "bold", "middle")}
  {t(945, 560, "1.7B-Q8_0: +37.3%", 23, C["accent"], "bold", "middle")}
  {rect(170, 612, 940, 46, C["panel2"], C["border"], 8)}
  {t(640, 642, "nt == 1 decode projection 走 subgroup GEMV_Q8；prefill 仍保留 MMQ8。", 18, C["sub"], "bold", "middle")}
""", "git 历史里 decode Q8 GEMV 是明显增益点，要单独给一页。")

add("17_lm_head_descriptor.svg", 17, "LM head 与 descriptor cache",
    header("LM head + descriptor cache：优化穿透到 Vulkan hot path", "我们不仅写 kernel，还把提交、等待、descriptor 分配这些系统开销纳入 gate。", 17) + f"""
  {mini_bar(70, 158, 560, 360, ["0.6B", "1.7B"], [388.606, 841.068], [C["accent"], C["primary2"]], 900, "TTFT ms")}
  {rect(700, 176, 430, 92, C["panel"], C["border"], 8)}
  {t(735, 232, "0", 38, C["accent"], "bold", family=MONO)}
  {t(800, 232, "hot descriptor allocation", 20, C["text"], "bold")}
  {rect(700, 300, 430, 92, C["panel"], C["border"], 8)}
  {t(735, 356, "40-50 ms", 38, C["primary2"], "bold", family=MONO)}
  {t(910, 356, "LM-head GPU time", 20, C["text"], "bold")}
  {rect(700, 424, 430, 92, C["panel"], C["border"], 8)}
  {t(735, 480, "top-k", 38, C["purple"], "bold", family=MONO)}
  {t(850, 480, "candidate set returned to CPU", 20, C["text"], "bold")}
  {t(640, 622, "系统开销进入指标体系，优化才不会停在“理论 kernel 更快”。", 24, C["accent"], "bold", "middle")}
""", "讲系统指标：submit、descriptor、GPU timestamp，这些都是 OS/驱动边界上的优化。")

add("18_context_8k.svg", 18, "8K Context",
    header("8K context：端侧 Agent 需要长期任务记忆", "长上下文不是炫参数，而是工具说明、历史摘要和用户偏好的工作集。", 18) + f"""
  {kpi(80, 170, "8192", "context tokens", C["primary2"], 260)}
  {kpi(390, 170, "5488", "user tokens prefetched", C["purple"], 300)}
  {kpi(740, 170, "43", "chunks of 128 tokens", C["accent"], 260)}
  {rect(90, 340, 1100, 145, C["panel"], C["border"], 8)}
  {t(130, 386, "长上下文执行机制", 24, C["text"], "bold")}
  {ml(130, 428, ["超过 256 tokens 使用 128-token chunks；只有最后 chunk 保留 logits，前面 chunk 跳过不必要 LM head。", "这相当于把长任务切成可调度的内存批次，避免一次 prefill 把端侧设备拖死。"], 20, C["sub"], 33)}
  {rect(160, 555, 960, 58, C["primary"], C["primary"], 8, "0.18")}
  {t(640, 592, "OS 类比：长 prompt 是大工作集，chunking 是端侧内存压力下的批处理策略。", 20, C["text"], "bold", "middle")}
""", "8K context 要讲成 Agent memory 需求，结合 chunking 和内存工作集。")

add("19_model_residency.svg", 19, "权重驻留与 1.7B",
    header("单份权重驻留 + 1.7B 支持：部署能力继续上探", "端侧优化不能只服务最小模型；要能支撑更强的小模型。", 19) + f"""
  {bullet_card(76, 168, 350, 230, "Step 1", ["Q8_0 单份权重驻留", "减少重复展开/复制", "保留 device-local LM head"], C["accent"])}
  {bullet_card(466, 168, 350, 230, "Step 2", ["参数化模型路径", "启用 Qwen3-1.7B", "健康状态输出模型级指标"], C["primary2"])}
  {bullet_card(856, 168, 350, 230, "Step 3", ["消除 llama.cpp CPU 重复驻留", "降低内存压力", "给 8K / 1.7B 留空间"], C["purple"])}
  {rect(150, 488, 980, 92, C["panel2"], C["border"], 8)}
  {t(190, 530, "为什么重要", 23, C["text"], "bold")}
  {t(190, 566, "移动端部署的瓶颈经常是 resident memory，而不是 FLOPS；少一份驻留就多一档可用模型。", 20, C["sub"])}
  {t(640, 638, "这是端侧框架和桌面 demo 的关键差别。", 26, C["accent"], "bold", "middle")}
""", "结合 git 的 Step 1/2/3 讲内存驻留，体现不是只做 0.6B demo。")

add("20_benchmark_gates.svg", 20, "Benchmark Gates",
    header("我们用 verifier gate 管优化，而不是只挑好看的样例", "每次优化都要同时过性能、正确性和系统状态门槛。", 20) + f"""
  {bullet_card(74, 158, 350, 230, "性能 gate", ["TTFT / TPS median", "LM head / decode ms", "submit count", "descriptor alloc"], C["accent"])}
  {bullet_card(466, 158, 350, 230, "正确性 gate", ["token ids reproducibility", "logits sanity", "top-k overlap", "attention fallback == 0"], C["primary2"])}
  {bullet_card(858, 158, 350, 230, "部署 gate", ["APK install/run", "model readable by app", "/health ready", "no active request leak"], C["purple"])}
  {rect(120, 468, 1040, 104, C["panel"], C["border"], 8)}
  {t(160, 512, "工程意义", 24, C["text"], "bold")}
  {ml(160, 550, ["CPU correctness checks 被隔离到 debug 模式；正常 TTFT 不背对拍成本。", "这让实验数据接近真实部署路径，而不是 debug-only 路径。"], 19, C["sub"], 31)}
  {t(640, 638, "系统优化必须被 gate 约束，否则很容易“快但不可信”。", 24, C["accent"], "bold", "middle")}
""", "这一页证明我们有工程闭环，而不是只跑了一次样例。")

add("21_comparison.svg", 21, "框架对比",
    header("OSH26 的位置：模型库、服务器 runtime、移动算子框架之间的端侧 Agent 层", "比较标准是 Android Agent 部署，而不是谁覆盖的场景最多。", 21) + f"""
  {rect(64, 150, 1152, 432, C["panel"], C["border"], 8)}
  {line(64, 220, 1216, 220, C["grid"], 2)}
  {line(64, 310, 1216, 310, C["border"], 1)}
  {line(64, 400, 1216, 400, C["border"], 1)}
  {line(64, 490, 1216, 490, C["border"], 1)}
  {''.join(line(x, 150, x, 582, C["border"], 1) for x in [280, 520, 760, 990])}
  {t(170, 195, "系统", 18, C["text"], "bold", "middle")}
  {t(400, 195, "优势", 18, C["text"], "bold", "middle")}
  {t(640, 195, "缺口", 18, C["text"], "bold", "middle")}
  {t(875, 195, "我们如何使用", 18, C["text"], "bold", "middle")}
  {t(1100, 195, "端侧 Agent 适配", 18, C["text"], "bold", "middle")}
  {t(170, 268, "llama.cpp", 18, C["primary2"], "bold", "middle", MONO)}
  {t(400, 268, "GGUF / tokenizer / CPU", 15, C["sub"], anchor="middle")}
  {t(640, 268, "不是 Android Agent Runtime", 15, C["sub"], anchor="middle")}
  {t(875, 268, "作为模型底座", 15, C["sub"], anchor="middle")}
  {t(1100, 268, "中", 22, C["primary2"], "bold", "middle")}
  {t(170, 358, "vLLM", 18, C["purple"], "bold", "middle", MONO)}
  {t(400, 358, "PagedAttention / scheduling", 15, C["sub"], anchor="middle")}
  {t(640, 358, "CUDA / Python / server 假设", 15, C["sub"], anchor="middle")}
  {t(875, 358, "借鉴内存思想", 15, C["sub"], anchor="middle")}
  {t(1100, 358, "中", 22, C["purple"], "bold", "middle")}
  {t(170, 448, "MNN", 18, C["accent"], "bold", "middle", MONO)}
  {t(400, 448, "移动算子和图优化", 15, C["sub"], anchor="middle")}
  {t(640, 448, "不围绕 GGUF Agent 生命周期", 15, C["sub"], anchor="middle")}
  {t(875, 448, "参考 Vulkan 工程", 15, C["sub"], anchor="middle")}
  {t(1100, 448, "中低", 22, C["primary2"], "bold", "middle")}
  {t(170, 538, "OSH26", 18, C["text"], "bold", "middle", MONO)}
  {t(400, 538, "Android + cache + Q8 backend", 15, C["sub"], anchor="middle")}
  {t(640, 538, "项目原型，仍可扩展", 15, C["sub"], anchor="middle")}
  {t(875, 538, "端侧 Agent Runtime", 15, C["sub"], anchor="middle")}
  {t(1100, 538, "高", 22, C["accent"], "bold", "middle")}
""", "对比要讲清楚我们的定位：不是全能框架，而是端侧 Agent 这条链路打穿。")

add("22_action_fabric_link.svg", 22, "与 Action Fabric 合流",
    header("与 Action Fabric 合流：上层任务图需要下层推理服务支撑", "Action Graph 减少模型轮次；OSH26 Runtime 让每次本地模型调用更快、更稳。", 22) + f"""
  {bullet_card(80, 170, 330, 230, "Action Fabric 需要", ["一次规划 Workflow", "大量短 subagent", "确定性工具节点", "失败后有限修正"], C["primary2"])}
  {bullet_card(475, 170, 330, 230, "LLM Runtime 提供", ["低 TTFT 本地服务", "stateless pinned prefix", "OpenAI-style API", "stream/cancel/health"], C["accent"])}
  {bullet_card(870, 170, 330, 230, "系统合成效果", ["少模型往返", "少上下文重复", "小并发可控", "端侧闭环可部署"], C["purple"])}
  {arrow(414, 280, 468, 280, C["accent"])}{arrow(809, 280, 863, 280, C["accent"])}
  {rect(160, 500, 960, 76, C["panel2"], C["border"], 8)}
  {t(640, 548, "上层把 Agent 变成可调度 DAG；下层把本地模型变成可调度服务。", 23, C["text"], "bold", "middle")}
""", "和前面 Action Fabric 结合，说明端侧部署不是孤立部分。")

add("23_teacher_takeaway.svg", 23, "老师应该看到什么",
    header("老师应该看到：这是系统课项目，不是单点模型 demo", "每个贡献点都对应一个操作系统问题和一个实机验证结果。", 23) + f"""
  {bullet_card(80, 160, 340, 300, "内存管理", ["KV page pool", "pinned/dynamic prefix", "LRU / fragmentation", "单份权重驻留"], C["purple"])}
  {bullet_card(470, 160, 340, 300, "调度与并发", ["prefix-aware-aging", "队列背压", "cancel/release", "stream callback"], C["primary2"])}
  {bullet_card(860, 160, 340, 300, "硬件抽象", ["OpenCL namespace 失败分析", "Vulkan 1.1 可控 kernel", "descriptor / submit gate", "CPU fallback 隔离"], C["accent"])}
  {rect(130, 535, 1020, 62, C["primary"], C["primary"], 8, "0.18")}
  {t(640, 574, "最终贡献：让端侧 Android Agent 从“能跑”走向“可部署、可解释、可持续优化”。", 22, C["text"], "bold", "middle")}
""", "这页可以作为答辩前的总结：课程知识点都自然嵌在系统实现里。")

add("24_closing.svg", 24, "结尾",
    f"""
  {t(90, 150, "FINAL ANSWER", 18, C["primary2"], "bold")}
  {ml(90, 236, ["OSH26 Runtime 让", "本地 Android Agent 可部署"], 48, C["text"], 64, "bold")}
  {ml(90, 390, ["我们复用 llama.cpp 的模型生态，补上 Android 端缺失的 Runtime：", "驱动可控、KV 可管理、请求可调度、指标可观测。"], 24, C["sub"], 38)}
  {kpi(92, 520, "9.34x", "Q8 prefill", C["accent"], 240)}
  {kpi(372, 520, "77.36%", "subagent cache", C["accent"], 260)}
  {kpi(672, 520, "8K", "context", C["primary2"], 200)}
  {kpi(912, 520, "0", "hot descriptor alloc", C["primary2"], 250)}
  {t(90, 700, "Thank you", 11, C["muted"])}
  {t(1196, 700, "24", 11, C["muted"], anchor="end", family=MONO)}
""", "结尾回到主张：端侧 AI 框架的重点是部署系统能力，而不是单次推理。")


for target in (OUT, FINAL):
    for old in target.glob("*.svg"):
        old.unlink()

for name, content in slides:
    (OUT / name).write_text(content, encoding="utf-8")
    shutil.copyfile(OUT / name, FINAL / name)

for old in NOTES.glob("*.md"):
    old.unlink()

total_parts = []
for title_id, page, title, note in notes:
    md = f"# {title_id}\n\n{page}. {title}\n\n{note}\n"
    (NOTES / f"{page:02d}_{title_id[3:]}.md").write_text(md, encoding="utf-8")
    total_parts.append(md)
(NOTES / "total.md").write_text("\n\n".join(total_parts), encoding="utf-8")

print(f"Generated {len(slides)} SVG slides in {OUT} and {FINAL}")
