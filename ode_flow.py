import ast
import ctypes
import io
import operator
import re
import sys
import time
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QSize, QSignalBlocker, QEvent
from PySide6.QtGui import QImage, QSurfaceFormat, QVector4D
from PySide6.QtOpenGL import (
    QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
    QOpenGLFunctions_3_3_Core, QOpenGLShader, QOpenGLShaderProgram,
    QOpenGLVersionFunctionsFactory, QOpenGLVersionProfile,
    QOpenGLTexture, QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QSplitter, QAbstractSpinBox, QComboBox, QStackedWidget
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _upload_rgba8(pixels, width, x_offset=0, y_offset=0):
    """Upload packed float bytes without narrowing their native memory address."""
    if not hasattr(_upload_rgba8, "gl_call"):
        if sys.platform != "win32":
            raise RuntimeError("Native OpenGL texture upload is currently Windows-only")
        call = ctypes.windll.opengl32.glTexSubImage2D
        call.argtypes = [
            ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p,
        ]
        call.restype = None
        _upload_rgba8.gl_call = call
    _upload_rgba8.gl_call(
        0x0DE1, 0, int(x_offset), int(y_offset), int(width), 1, 0x1908, 0x1401,
        ctypes.c_void_p(int(pixels.ctypes.data)),
    )


# ============================================================
# Safe expression parser: no eval(), no attribute access/imports
# ============================================================

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
    "sign": np.sign, "minimum": np.minimum, "maximum": np.maximum,
}

LATEX_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ",
    r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\omega": "ω",
    r"\zeta": "ζ",
}


def normalize_latex_symbols(text):
    for latex, symbol in LATEX_SYMBOLS.items():
        text = text.replace(latex, symbol)
    return text


SUPERSCRIPT_TO_ASCII = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾", "0123456789+-=()"))
SUPERSCRIPT_TO_ASCII.update({
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f",
    "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j", "ᵏ": "k", "ˡ": "l",
    "ᵐ": "m", "ⁿ": "n", "ᵒ": "o", "ᵖ": "p", "ʳ": "r", "ˢ": "s",
    "ᵗ": "t", "ᵘ": "u", "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
})
ASCII_TO_SUPERSCRIPT = {value: key for key, value in SUPERSCRIPT_TO_ASCII.items()}
SUBSCRIPT_TO_ASCII = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
ASCII_TO_SUBSCRIPT = {value: key for key, value in SUBSCRIPT_TO_ASCII.items()}


def normalize_display_expression(text):
    """Convert the editor's typographic symbols back into parser syntax."""
    text = normalize_latex_symbols(text.strip())
    text = text.replace("π", "pi").replace("·", "*").replace("×", "*")
    text = text.replace("÷", "/").replace("−", "-")
    text = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+", lambda m: "".join(SUBSCRIPT_TO_ASCII[c] for c in m.group()), text)

    # Parenthesized superscripts (for example x⁽ᵃ⁺¹⁾) must stay grouped.
    text = re.sub(
        r"⁽([^⁽⁾]*)⁾",
        lambda m: "**(" + "".join(SUPERSCRIPT_TO_ASCII.get(c, c) for c in m.group(1)) + ")",
        text,
    )
    output = []
    index = 0
    superscript_chars = set(SUPERSCRIPT_TO_ASCII) - {"⁽", "⁾"}
    while index < len(text):
        if text[index] in superscript_chars:
            end = index + 1
            while end < len(text) and text[end] in superscript_chars:
                end += 1
            output.append("**" + "".join(SUPERSCRIPT_TO_ASCII[c] for c in text[index:end]))
            index = end
        else:
            output.append(text[index])
            index += 1
    return "".join(output)


def format_formula_input(text):
    """Render common calculator notation as typographic symbols in-place."""
    text = normalize_latex_symbols(text).replace("**", "^")

    def format_power(match):
        exponent = match.group("exponent")
        if exponent.startswith("(") and exponent.endswith(")"):
            value = exponent[1:-1]
            if not value or any(char not in ASCII_TO_SUPERSCRIPT for char in value):
                return match.group(0)
            return match.group("base") + "⁽" + "".join(ASCII_TO_SUPERSCRIPT[c] for c in value) + "⁾"
        if any(char not in ASCII_TO_SUPERSCRIPT for char in exponent):
            return match.group(0)
        return match.group("base") + "".join(ASCII_TO_SUPERSCRIPT[c] for c in exponent)

    power_pattern = re.compile(
        r"(?P<base>(?:[A-Za-z_Α-Ωα-ωπ][A-Za-z_0-9Α-Ωα-ωπ]*|\([^()]*\)))"
        r"\s*\^\s*(?P<exponent>\([^()]*\)|[+-]?(?:\d+(?:\.\d*)?|\.\d+)|[A-Za-z_Α-Ωα-ω][A-Za-z_0-9Α-Ωα-ω]*)"
    )
    for _ in range(4):
        updated = power_pattern.sub(format_power, text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"\bpi\b", "π", text)
    text = re.sub(
        r"\b([A-Za-z]+)(\d+)\b",
        lambda m: m.group(1) + "".join(ASCII_TO_SUBSCRIPT.get(c, c) for c in m.group(2)),
        text,
    )
    return text.replace("*", "·").replace("/", "÷").replace("-", "−")
BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def eval_node(node, variables):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numerical constants are allowed.")

    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id == "pi":
            return np.pi
        if node.id == "e":
            return np.e
        if node.id in FUNCTIONS:
            return FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable or function: {node.id}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in BINARY_OPS:
            raise ValueError("Unsupported operator.")
        return BINARY_OPS[op_type](
            eval_node(node.left, variables),
            eval_node(node.right, variables)
        )

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ValueError("Unsupported unary operator.")
        return UNARY_OPS[op_type](eval_node(node.operand, variables))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ValueError("Only supported math functions may be called.")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")
        return FUNCTIONS[node.func.id](
            *[eval_node(arg, variables) for arg in node.args]
        )

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def compile_formula(text):
    text = normalize_display_expression(text).replace("^", "**")
    if not text:
        raise ValueError("Formula cannot be empty.")
    return ast.parse(text, mode="eval").body


def evaluate(tree, x, y, params):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        result = eval_node(tree, {"x": x, "y": y, **params})
    return np.asarray(result, dtype=float)


def evaluate_3d(tree, x, y, z, params):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        result = eval_node(tree, {"x": x, "y": y, "z": z, **params})
    return np.asarray(result, dtype=float)


def parse_parameters(text):
    params = {}
    text = normalize_display_expression(text)
    if not text.strip():
        return params
    for item in text.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid parameter '{item}'. Use symbol=value.")
        name, value = (part.strip() for part in item.split("=", 1))
        if not name.isidentifier():
            raise ValueError(f"Invalid parameter symbol: {name}")
        if name in {"x", "y", "z", "pi", "e"} or name in FUNCTIONS:
            raise ValueError(f"Reserved parameter symbol: {name}")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Parameters must be finite numbers.")
        params[name] = number
    return params


def glsl_number(value):
    text = format(float(value), ".9g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text


def compile_glsl(node, params):
    """Translate the already-validated expression AST into safe GLSL."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return glsl_number(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return "3.141592653589793"
        if node.id == "e":
            return "2.718281828459045"
        if node.id in {"x", "y"}:
            return node.id
        if node.id in params:
            return glsl_number(params[node.id])
        raise ValueError(f"Unknown variable: {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        symbol = "+" if isinstance(node.op, ast.UAdd) else "-"
        return f"({symbol}{compile_glsl(node.operand, params)})"
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
        left = compile_glsl(node.left, params)
        right = compile_glsl(node.right, params)
        if isinstance(node.op, ast.Pow):
            return f"pow({left}, {right})"
        if isinstance(node.op, ast.Mod):
            return f"mod({left}, {right})"
        symbol = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"
        }[type(node.op)]
        return f"({left} {symbol} {right})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [compile_glsl(arg, params) for arg in node.args]
        if name == "log10":
            return f"(log({args[0]}) / log(10.0))"
        name = {"ln": "log", "minimum": "min", "maximum": "max"}.get(name, name)
        return f"{name}({', '.join(args)})"
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


class NoWheelSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event):
        event.accept()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event):
        event.accept()


class FlowCanvas(QOpenGLWidget):
    gpuStatus = Signal(str)
    canvasResized = Signal()
    viewChanged = Signal(float, float, float, float)
    TEX2D, RGBA8, RGBA, UBYTE = 0x0DE1, 0x8058, 0x1908, 0x1401
    MIN_FILTER, MAG_FILTER, WRAP_S, WRAP_T = 0x2801, 0x2800, 0x2802, 0x2803
    CLAMP, NEAREST, LINEAR = 0x812F, 0x2600, 0x2601
    COLOR_BIT, FRAMEBUFFER, TEX0 = 0x4000, 0x8D40, 0x84C0
    TRIANGLES, POINTS = 0x0004, 0x0000
    SCISSOR_TEST = 0x0C11
    BLEND, DEPTH, ONE, ONE_MINUS_SRC_ALPHA = 0x0BE2, 0x0B71, 1, 0x0303
    PROGRAM_POINT_SIZE = 0x8642

    QUAD_VS = """#version 330 core
    out vec2 v_uv;
    void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
      v_uv=p;gl_Position=vec4(p*2.0-1.0,0.0,1.0);}
    """
    SIM_FS = """#version 330 core
    uniform sampler2D u_x;uniform sampler2D u_y;uniform float u_count,u_dt,u_coordinate;
    uniform int u_normalize;out vec4 out_color;
    float decodeF(vec4 c){uvec4 b=uvec4(round(c*255.0));uint n=(b.r<<24u)|(b.g<<16u)|(b.b<<8u)|b.a;return uintBitsToFloat(n);}
    vec4 encodeF(float v){uint n=floatBitsToUint(v);return vec4(float((n>>24u)&255u),float((n>>16u)&255u),float((n>>8u)&255u),float(n&255u))/255.0;}
    vec2 velocity(vec2 p){float x=p.x;float y=p.y;return vec2(__VX__,__VY__);}
    vec2 field(vec2 p){vec2 v=velocity(p);if(u_normalize==1){float m=length(v);if(m>1e-12)v/=m;}return v;}
    void main(){vec2 uv=vec2(gl_FragCoord.x/u_count,0.5);
      vec2 p=vec2(decodeF(texture(u_x,uv)),decodeF(texture(u_y,uv)));
      vec2 k1=field(p),k2=field(p+0.5*u_dt*k1),k3=field(p+0.5*u_dt*k2),k4=field(p+u_dt*k3);
      vec2 q=p+(u_dt/6.0)*(k1+2.0*k2+2.0*k3+k4);
      if(any(isnan(q))||any(isinf(q)))q=p;
      out_color=encodeF(u_coordinate<0.5?q.x:q.y);}
    """
    TRAIL_VS = """#version 330 core
    uniform sampler2D u_history_x,u_history_y,u_history_birth;
    uniform float u_count,u_history_length,u_history_latest,u_speed_max,u_line_width,u_frame;
    uniform vec4 u_bounds,u_plot,u_viewport;out float v_speed;out float v_edge;out float v_age;out float v_birth_fade;
    float decodeF(vec4 c){uvec4 b=uvec4(round(c*255.0));uint n=(b.r<<24u)|(b.g<<16u)|(b.b<<8u)|b.a;return uintBitsToFloat(n);}
    vec2 velocity(vec2 p){float x=p.x;float y=p.y;return vec2(__VX__,__VY__);}
    vec2 sample_position(int particle,int age){
      int row=(int(u_history_latest)-age+int(u_history_length)*int(u_history_length))%int(u_history_length);
      vec2 uv=vec2((float(particle)+0.5)/u_count,(float(row)+0.5)/u_history_length);
      return vec2(decodeF(texture(u_history_x,uv)),decodeF(texture(u_history_y,uv)));}
    float sample_birth(int particle,int age){
      int row=(int(u_history_latest)-age+int(u_history_length)*int(u_history_length))%int(u_history_length);
      vec2 uv=vec2((float(particle)+0.5)/u_count,(float(row)+0.5)/u_history_length);
      return decodeF(texture(u_history_birth,uv));}
    vec2 project(vec2 p){
      vec2 lo=vec2(u_bounds.x,u_bounds.y),hi=vec2(u_bounds.z,u_bounds.w),unit=(p-lo)/(hi-lo);
      vec2 lower=2.0*vec2(u_plot.x,u_plot.z)-1.0;
      vec2 upper=2.0*vec2(u_plot.y,u_plot.w)-1.0;
      return mix(lower,upper,unit);}
    void main(){int corner=gl_VertexID%6;int segment=(gl_VertexID/6)%int(u_history_length-1.0);
      int particle=gl_VertexID/(6*int(u_history_length-1.0));
      float newer_birth=sample_birth(particle,segment);
      float older_birth=sample_birth(particle,segment+1);
      if(segment>=int(min(u_frame,u_history_length-1.0)) || abs(newer_birth-older_birth)>0.25){
        gl_Position=vec4(2.0,2.0,0.0,1.0);v_speed=0.0;v_edge=0.0;v_age=1.0;v_birth_fade=0.0;return;}
      vec2 older=sample_position(particle,segment+1),newer=sample_position(particle,segment);
      vec2 a=project(older),b=project(newer);
      vec2 delta=(b-a)*u_viewport.xy*0.5;
      vec2 normal=vec2(-delta.y,delta.x)/max(length(delta),1e-5);
      vec2 offset=normal*(u_line_width/u_viewport.xy);
      bool at_a=(corner==0||corner==1||corner==3);
      bool positive=(corner==0||corner==3||corner==4);
      vec2 p=at_a?older:newer;float s=length(velocity(p));
      v_speed=(isnan(s)||isinf(s))?0.0:s;v_edge=positive?1.0:-1.0;
      v_age=float(segment)/max(1.0,u_history_length-1.0);
      v_birth_fade=smoothstep(0.0,12.0,u_frame-newer_birth);
      vec2 base=at_a?a:b;gl_Position=vec4(base+(positive?offset:-offset),0.0,1.0);}
    """
    TRAIL_FS = """#version 330 core
    in float v_speed;in float v_edge;in float v_age;in float v_birth_fade;uniform int u_speed_color,u_dark_mode;
    uniform float u_speed_max,u_density_scale;out vec4 out_color;
    void main(){float a=0.94*u_density_scale*v_birth_fade*(1.0-smoothstep(0.65,1.0,abs(v_edge)))*(1.0-v_age);
      float t=clamp(log(1.0+max(v_speed,0.0))/max(log(1.0+u_speed_max),1e-6),0.0,1.0);
      vec3 spectrum=clamp(1.5-abs(4.0*t-vec3(3.0,2.0,1.0)),0.0,1.0);
      vec3 base=u_speed_color==1?spectrum:(u_dark_mode==1?vec3(0.38,0.76,1.0):vec3(0.02,0.12,0.38));
      out_color=vec4(base*a,a);}
    """
    STORE_FS = """#version 330 core
    uniform sampler2D u_x,u_y,u_birth_frame;uniform float u_count,u_coordinate;out vec4 out_color;
    float decodeF(vec4 c){uvec4 b=uvec4(round(c*255.0));uint n=(b.r<<24u)|(b.g<<16u)|(b.b<<8u)|b.a;return uintBitsToFloat(n);}
    vec4 encodeF(float v){uint n=floatBitsToUint(v);return vec4(float((n>>24u)&255u),float((n>>16u)&255u),float((n>>8u)&255u),float(n&255u))/255.0;}
    void main(){vec2 uv=vec2(gl_FragCoord.x/u_count,0.5);
      if(u_coordinate>1.5){out_color=texture(u_birth_frame,uv);return;}
      float value=u_coordinate<0.5?decodeF(texture(u_x,uv)):decodeF(texture(u_y,uv));
      out_color=encodeF(value);}
    """
    BACKGROUND_FS = """#version 330 core
    in vec2 v_uv;uniform sampler2D u_image;out vec4 out_color;
    void main(){out_color=texture(u_image,vec2(v_uv.x,1.0-v_uv.y));}
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self.gl = None
        self.vao = None
        self.ready = self.running = False
        self.formulas = ("0.0", "0.0")
        self.bounds = (-5.0, 5.0, -5.0, 5.0)
        self.count, self.trail_length, self.dt, self.normalize = 1800, 100, 0.01, True
        self.speed, self.simulation_substeps = 1.0, 1
        self.respawn_period = 8.0
        self.speed_color = False
        self.dark_mode = False
        self.speed_max = 1.0
        self._rng = np.random.default_rng()
        self.history_frame = 0
        self._last_frame_time = time.monotonic()
        self.background_image = QImage()
        self.background_texture = None
        self.plot_rect = (0.10, 0.98, 0.10, 0.94)
        self.position_fbos, self.history_fbos = [[], []], []
        self.position_ping = 0
        self.history_cursor = self.history_latest = 0
        self._pan_origin = None
        self._pan_bounds = None
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._advance)
        self.timer.start()

    def set_model(self, vx, vy, bounds, count, trail, dt, speed, normalize, respawn_period,
                  speed_color, dark_mode, speed_max, background_png, plot_rect):
        self.formulas = (vx, vy)
        self.bounds = tuple(float(v) for v in bounds)
        self.count, self.trail_length = int(count), int(trail)
        self._set_simulation_rate(dt, speed)
        self.respawn_period = float(respawn_period)
        self.speed_color = bool(speed_color)
        self.dark_mode = bool(dark_mode)
        self.speed_max = max(float(speed_max), 1e-9)
        self._rng = np.random.default_rng()
        self.normalize = bool(normalize)
        self.set_background(background_png, plot_rect)
        self.update()
        if self.ready:
            self.makeCurrent()
            try:
                self._build_resources()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()

    def set_background(self, background_png, plot_rect):
        self.background_image = QImage.fromData(background_png, "PNG")
        self.plot_rect = tuple(float(v) for v in plot_rect)
        if self.ready:
            self.makeCurrent()
            try:
                self._create_background_texture()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()
        self.update()

    def _set_simulation_rate(self, dt, speed):
        self.speed = max(float(speed), 0.01)
        self.simulation_substeps = max(1, int(np.ceil(self.speed)))
        self.dt = float(dt) * self.speed / self.simulation_substeps

    def update_runtime(self, count, trail, dt, speed, normalize, respawn_period):
        count = int(count)
        trail = int(trail)
        trail_changed = trail != self.trail_length
        self.trail_length, self.normalize = trail, bool(normalize)
        self._set_simulation_rate(dt, speed)
        self.respawn_period = float(respawn_period)
        if count == self.count:
            if trail_changed and self.ready:
                self.makeCurrent()
                self._create_history_fbos()
                self._seed_particles()
                self._clear_trails()
                self.doneCurrent()
            return
        self.count = count
        if self.ready:
            self.makeCurrent()
            try:
                self._create_position_fbos()
                self._create_history_fbos()
                self._seed_particles()
                self._clear_trails()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()

    def set_view_bounds(self, bounds):
        self.bounds = tuple(float(v) for v in bounds)
        if self.ready:
            self.makeCurrent()
            try:
                self._seed_particles()
                self._clear_trails()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()
        self.update()

    def set_visual_options(self, speed_color, dark_mode, speed_max=None):
        speed_color = bool(speed_color)
        speed_color_changed = speed_color != self.speed_color
        self.speed_color = speed_color
        self.dark_mode = bool(dark_mode)
        if speed_max is not None:
            self.speed_max = max(float(speed_max), 1e-9)
        if self.ready:
            self.makeCurrent()
            try:
                self._clear_trails()
                if speed_color_changed:
                    self._seed_particles()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()
        self.update()

    def set_running(self, running):
        self.running = bool(running)

    def clear_trails(self):
        if self.ready:
            self.makeCurrent()
            self._clear_trails()
            self.doneCurrent()

    def _fail(self, error):
        self.ready = False
        self.gpuStatus.emit(f"OpenGL setup failed: {error}")

    def initializeGL(self):
        try:
            profile = QOpenGLVersionProfile()
            profile.setVersion(3, 3)
            profile.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            self.gl = QOpenGLVersionFunctionsFactory.get(profile, self.context())
            if self.gl is None or not self.gl.initializeOpenGLFunctions():
                raise RuntimeError("OpenGL 3.3 core functions unavailable")
            self.gl.glDisable(self.DEPTH)
            self.gl.glEnable(self.PROGRAM_POINT_SIZE)
            self.vao = QOpenGLVertexArrayObject(self)
            if not self.vao.create():
                raise RuntimeError("Could not create OpenGL vertex array")
            self.vao.bind()
            self._build_resources()
        except Exception as exc:
            self._fail(exc)

    def resizeGL(self, width, height):
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvasResized.emit()

    def _screen_to_view(self, position):
        left, right, bottom, top = self.plot_rect
        x_fraction = (position.x() / max(1, self.width()) - left) / max(1e-9, right-left)
        y_fraction = ((1.0-position.y() / max(1, self.height())) - bottom) / max(1e-9, top-bottom)
        if not (0.0 <= x_fraction <= 1.0 and 0.0 <= y_fraction <= 1.0):
            return None
        xmin, xmax, ymin, ymax = self.bounds
        return (xmin + x_fraction*(xmax-xmin), ymin + y_fraction*(ymax-ymin))

    def wheelEvent(self, event):
        anchor = self._screen_to_view(event.position())
        delta = event.angleDelta().y()
        if anchor is None or delta == 0:
            event.ignore()
            return
        factor = 0.8 if delta > 0 else 1.25
        xmin, xmax, ymin, ymax = self.bounds
        ax, ay = anchor
        self.viewChanged.emit(
            ax + (xmin-ax)*factor, ax + (xmax-ax)*factor,
            ay + (ymin-ay)*factor, ay + (ymax-ay)*factor,
        )
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._screen_to_view(event.position()) is not None:
            self._pan_origin = event.position()
            self._pan_bounds = self.bounds
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            dx = event.position().x() - self._pan_origin.x()
            dy = event.position().y() - self._pan_origin.y()
            xmin, xmax, ymin, ymax = self._pan_bounds
            left, right, bottom, top = self.plot_rect
            plot_width = max(1.0, self.width()*(right-left))
            plot_height = max(1.0, self.height()*(top-bottom))
            shift_x = -dx/plot_width*(xmax-xmin)
            shift_y = dy/plot_height*(ymax-ymin)
            self.viewChanged.emit(xmin+shift_x, xmax+shift_x, ymin+shift_y, ymax+shift_y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._pan_origin is not None:
            self._pan_origin = None
            self._pan_bounds = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintGL(self):
        gl = self.gl
        if gl is None:
            return
        gl.glBindFramebuffer(self.FRAMEBUFFER, self.defaultFramebufferObject())
        ratio = self.devicePixelRatioF()
        gl.glViewport(0, 0, int(self.width()*ratio), int(self.height()*ratio))
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(self.COLOR_BIT)
        self.vao.bind()
        if self.background_texture is not None and self.background_texture.isCreated():
            gl.glDisable(self.BLEND)
            self.background_texture.bind(0)
            self.background_program.bind()
            self._set_uniform(self.background_program, "u_image", 0)
            gl.glDrawArrays(self.TRIANGLES, 0, 3)
            self.background_program.release()
        if self.ready:
            self._draw_trail_history(int(self.width()*ratio), int(self.height()*ratio))
        self.vao.release()

    def _make_program(self, vs, fs):
        p = QOpenGLShaderProgram(self)
        if not p.addShaderFromSourceCode(QOpenGLShader.Vertex, vs):
            raise RuntimeError(p.log())
        if not p.addShaderFromSourceCode(QOpenGLShader.Fragment, fs):
            raise RuntimeError(p.log())
        if not p.link():
            raise RuntimeError(p.log())
        return p

    def _build_resources(self):
        sim = self.SIM_FS.replace("__VX__", self.formulas[0]).replace("__VY__", self.formulas[1])
        trail_vs = self.TRAIL_VS.replace("__VX__", self.formulas[0]).replace("__VY__", self.formulas[1])
        self.sim_program = self._make_program(self.QUAD_VS, sim)
        self.trail_program = self._make_program(trail_vs, self.TRAIL_FS)
        self.store_program = self._make_program(self.QUAD_VS, self.STORE_FS)
        self.background_program = self._make_program(self.QUAD_VS, self.BACKGROUND_FS)
        self._create_position_fbos()
        self._create_history_fbos()
        self._seed_particles()
        self._clear_trails()
        self._create_background_texture()
        self.ready = True
        self.gpuStatus.emit("GPU particle flow ready.")

    def _create_background_texture(self):
        if self.background_texture is not None:
            self.background_texture.destroy()
            self.background_texture = None
        if self.background_image.isNull():
            return
        texture = QOpenGLTexture(
            self.background_image,
            QOpenGLTexture.MipMapGeneration.DontGenerateMipMaps,
        )
        texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
        texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionS,
                             QOpenGLTexture.WrapMode.ClampToEdge)
        texture.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionT,
                             QOpenGLTexture.WrapMode.ClampToEdge)
        self.background_texture = texture

    def _new_fbo(self, width, height, linear=False):
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.NoAttachment)
        fmt.setInternalTextureFormat(self.RGBA8)
        fmt.setTextureTarget(self.TEX2D)
        fbo = QOpenGLFramebufferObject(QSize(max(1, width), max(1, height)), fmt)
        if not fbo.isValid():
            raise RuntimeError(f"RGBA8 framebuffer allocation failed ({width}x{height})")
        self._bind_texture(0, fbo.texture())
        filt = self.LINEAR if linear else self.NEAREST
        for pname, value in ((self.MIN_FILTER, filt), (self.MAG_FILTER, filt),
                             (self.WRAP_S, self.CLAMP), (self.WRAP_T, self.CLAMP)):
            self.gl.glTexParameteri(self.TEX2D, pname, value)
        return fbo

    def _create_position_fbos(self):
        self.position_fbos = [[self._new_fbo(self.count, 1), self._new_fbo(self.count, 1)]
                              for _ in range(2)]

    def _create_history_fbos(self):
        self.history_fbos = [
            self._new_fbo(self.count, self.trail_length),
            self._new_fbo(self.count, self.trail_length),
        ]
        self.birth_fbo = self._new_fbo(self.count, 1)
        self.history_birth_fbo = self._new_fbo(self.count, self.trail_length)
        self.history_cursor = self.history_latest = 0

    def _seed_particles(self):
        xmin, xmax, ymin, ymax = self.bounds
        cols = max(1, int(np.ceil(np.sqrt(self.count*(xmax-xmin)/max(ymax-ymin, 1e-12)))))
        rows = int(np.ceil(self.count/cols))
        i = np.arange(self.count)
        jitter = np.random.default_rng().uniform(-0.22, 0.22, self.count)
        xs = xmin + np.clip((i % cols + 0.5 + jitter)/cols, 0, 1)*(xmax-xmin)
        ys = ymin + np.clip((i // cols + 0.5 + jitter)/rows, 0, 1)*(ymax-ymin)
        for pair in self.position_fbos:
            for fbo, values in zip(pair, (xs, ys)):
                raw = np.ascontiguousarray(np.asarray(values, dtype=">f4").view(np.uint8))
                self._bind_texture(0, fbo.texture())
                _upload_rgba8(raw, self.count)
        self.position_ping = 0
        self.history_cursor = self.history_latest = 0
        self.history_frame = 0
        self._rng = np.random.default_rng()
        self._upload_birth_frames(np.zeros(self.count, dtype=np.float32))
        self._seed_history(xs, ys)

    def _upload_birth_frames(self, frames, start=0):
        raw = np.ascontiguousarray(np.asarray(frames, dtype=">f4").view(np.uint8))
        self._bind_texture(0, self.birth_fbo.texture())
        _upload_rgba8(raw, len(frames), start)

    def _seed_history(self, xs, ys, start=0):
        if len(self.history_fbos) != 2:
            return
        for fbo, values in zip(self.history_fbos, (xs, ys)):
            raw = np.ascontiguousarray(np.asarray(values, dtype=">f4").view(np.uint8))
            self._bind_texture(0, fbo.texture())
            for row in range(self.trail_length):
                _upload_rgba8(raw, len(values), start, row)
        birth_raw = np.zeros((len(xs), 4), dtype=np.uint8)
        self._bind_texture(0, self.history_birth_fbo.texture())
        for row in range(self.trail_length):
            _upload_rgba8(birth_raw, len(xs), start, row)

    def _respawn_indices(self, indices, positions):
        if len(indices) == 0:
            return
        xs, ys = positions[:, 0], positions[:, 1]
        for pair in self.position_fbos:
            for fbo, values in zip(pair, (xs, ys)):
                self._bind_texture(0, fbo.texture())
                for index, value in zip(indices, values):
                    raw = np.ascontiguousarray(np.asarray([value], dtype=">f4").view(np.uint8))
                    _upload_rgba8(raw, 1, int(index))
        self._bind_texture(0, self.birth_fbo.texture())
        born = np.full(len(indices), self.history_frame + 1, dtype=np.float32)
        for index, value in zip(indices, born):
            raw = np.ascontiguousarray(np.asarray([value], dtype=">f4").view(np.uint8))
            _upload_rgba8(raw, 1, int(index))

    def _respawn_due_particles(self, elapsed):
        if elapsed <= 0.0:
            return
        # Independent exponential lifetimes avoid a synchronized population
        # restart, following Fieldplay's per-frame probabilistic particle drop.
        probability = 1.0 - np.exp(-elapsed / max(0.1, self.respawn_period))
        indices = np.flatnonzero(self._rng.random(self.count) < probability)
        if not len(indices):
            return
        xmin, xmax, ymin, ymax = self.bounds
        positions = np.column_stack((
            self._rng.uniform(xmin, xmax, len(indices)),
            self._rng.uniform(ymin, ymax, len(indices)),
        )).astype(np.float32)
        self._respawn_indices(indices, positions)

    def _clear_trails(self):
        if len(self.history_fbos) == 2:
            self._seed_particles()

    def _bind_texture(self, unit, texture):
        self.gl.glActiveTexture(self.TEX0 + unit)
        self.gl.glBindTexture(self.TEX2D, texture)

    def _draw_quad(self):
        self.gl.glDrawArrays(self.TRIANGLES, 0, 3)

    def _set_uniform(self, program, name, value):
        location = program.uniformLocation(name)
        if location < 0:
            raise RuntimeError(f"OpenGL uniform is unavailable: {name}")
        if isinstance(value, QVector4D):
            self.gl.glUniform4f(location, value.x(), value.y(), value.z(), value.w())
        elif isinstance(value, float):
            self.gl.glUniform1f(location, value)
        else:
            self.gl.glUniform1i(location, int(value))

    def _simulate(self):
        gl = self.gl
        next_ping = 1-self.position_ping
        self.sim_program.bind()
        self._bind_texture(0, self.position_fbos[self.position_ping][0].texture())
        self._bind_texture(1, self.position_fbos[self.position_ping][1].texture())
        self._set_uniform(self.sim_program, "u_x", 0)
        self._set_uniform(self.sim_program, "u_y", 1)
        self._set_uniform(self.sim_program, "u_count", float(self.count))
        self._set_uniform(self.sim_program, "u_dt", float(self.dt))
        self._set_uniform(self.sim_program, "u_normalize", 1 if self.normalize else 0)
        gl.glDisable(self.BLEND)
        for c in (0, 1):
            self.position_fbos[next_ping][c].bind()
            gl.glViewport(0, 0, self.count, 1)
            self._set_uniform(self.sim_program, "u_coordinate", float(c))
            self._draw_quad()
            self.position_fbos[next_ping][c].release()
        gl.glBindFramebuffer(self.FRAMEBUFFER, self.defaultFramebufferObject())
        self.sim_program.release()
        self.position_ping = next_ping

    def _record_history(self):
        gl = self.gl
        self.store_program.bind()
        self._bind_texture(0, self.position_fbos[self.position_ping][0].texture())
        self._bind_texture(1, self.position_fbos[self.position_ping][1].texture())
        self._set_uniform(self.store_program, "u_x", 0)
        self._set_uniform(self.store_program, "u_y", 1)
        self._bind_texture(2, self.birth_fbo.texture())
        self._set_uniform(self.store_program, "u_birth_frame", 2)
        self._set_uniform(self.store_program, "u_count", float(self.count))
        gl.glDisable(self.BLEND)
        for coordinate, fbo in enumerate((*self.history_fbos, self.history_birth_fbo)):
            fbo.bind()
            gl.glViewport(0, self.history_cursor, self.count, 1)
            self._set_uniform(self.store_program, "u_coordinate", float(coordinate))
            self._draw_quad()
            fbo.release()
        self.store_program.release()
        gl.glBindFramebuffer(self.FRAMEBUFFER, self.defaultFramebufferObject())
        self.history_latest = self.history_cursor
        self.history_cursor = (self.history_cursor + 1) % self.trail_length
        self.history_frame += 1

    def _draw_trail_history(self, width, height):
        gl = self.gl
        gl.glEnable(self.BLEND)
        gl.glBlendFunc(self.ONE, self.ONE_MINUS_SRC_ALPHA)
        self.trail_program.bind()
        self._bind_texture(0, self.history_fbos[0].texture())
        self._bind_texture(1, self.history_fbos[1].texture())
        self._bind_texture(2, self.history_birth_fbo.texture())
        self._set_uniform(self.trail_program, "u_history_x", 0)
        self._set_uniform(self.trail_program, "u_history_y", 1)
        self._set_uniform(self.trail_program, "u_history_birth", 2)
        self._set_uniform(self.trail_program, "u_count", float(self.count))
        self._set_uniform(self.trail_program, "u_history_length", float(self.trail_length))
        self._set_uniform(self.trail_program, "u_history_latest", float(self.history_latest))
        self._set_uniform(self.trail_program, "u_frame", float(self.history_frame))
        xmin, xmax, ymin, ymax = self.bounds
        self._set_uniform(self.trail_program, "u_bounds", QVector4D(xmin, ymin, xmax, ymax))
        self._set_uniform(self.trail_program, "u_plot", QVector4D(*self.plot_rect))
        self._set_uniform(self.trail_program, "u_viewport", QVector4D(float(width), float(height), 0.0, 0.0))
        self._set_uniform(self.trail_program, "u_speed_max", float(self.speed_max))
        self._set_uniform(self.trail_program, "u_density_scale", min(1.0, 1800.0/self.count))
        self._set_uniform(self.trail_program, "u_speed_color", 1 if self.speed_color else 0)
        self._set_uniform(self.trail_program, "u_dark_mode", 1 if self.dark_mode else 0)
        self._set_uniform(self.trail_program, "u_line_width", float(2.2 * self.devicePixelRatioF()))
        left, right, bottom, top = self.plot_rect
        gl.glEnable(self.SCISSOR_TEST)
        gl.glScissor(int(left*width), int(bottom*height),
                     max(1, int((right-left)*width)), max(1, int((top-bottom)*height)))
        gl.glDrawArrays(self.TRIANGLES, 0, self.count*(self.trail_length-1)*6)
        gl.glDisable(self.SCISSOR_TEST)
        self.trail_program.release()
        gl.glDisable(self.BLEND)

    def _advance(self):
        if not self.ready or not self.running:
            return
        self.makeCurrent()
        try:
            self.vao.bind()
            now = time.monotonic()
            elapsed = min(max(now - self._last_frame_time, 0.0), 0.1)
            self._last_frame_time = now
            self._respawn_due_particles(elapsed * self.speed)
            for _ in range(self.simulation_substeps):
                self._simulate()
                self._record_history()
            self.vao.release()
        except Exception as exc:
            self._fail(exc)
        finally:
            self.doneCurrent()
        self.update()


class Flow3DCanvas(QWidget):
    """Interactive, rotatable phase-space view for a 3D autonomous system."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.trajectory = np.empty((0, 3))
        self.index = 0
        self.playback_speed = 1.0
        self._playback_remainder = 0.0
        self.running = False
        self._dark = False
        self._path_artist = None
        self._marker, = self.ax.plot([], [], [], marker="o", markersize=5, linestyle="None")
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._advance)
        self._draw_empty()

    @staticmethod
    def _integrate(fields, params, initial, dt, steps=9000):
        def f(point):
            x, y, z = point
            return np.array([evaluate_3d(tree, x, y, z, params).item() for tree in fields], dtype=float)
        point = np.asarray(initial, dtype=float)
        points, speeds = [], []
        for i in range(steps + 350):
            try:
                k1 = f(point)
                k2 = f(point + 0.5 * dt * k1)
                k3 = f(point + 0.5 * dt * k2)
                k4 = f(point + dt * k3)
                velocity = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                next_point = point + dt * velocity
            except (ValueError, TypeError, OverflowError, FloatingPointError):
                break
            if not np.all(np.isfinite(next_point)) or np.max(np.abs(next_point)) > 1e5:
                break
            point = next_point
            if i >= 350:
                points.append(point.copy())
                speeds.append(float(np.linalg.norm(velocity)))
        if len(points) < 20:
            raise ValueError("This 3D model did not produce a finite trajectory.")
        return np.asarray(points), np.asarray(speeds)

    def set_model(self, fields, params, initial, dt, dark, speed_color, playback_speed=1.0):
        trajectory, speeds = self._integrate(fields, params, initial, dt)
        self.trajectory = trajectory
        self.index = 0
        self.set_playback_speed(playback_speed)
        self._dark = bool(dark)
        self.ax.clear()
        bg = "#000000" if dark else "#ffffff"
        fg = "#dedede" if dark else "#222222"
        self.figure.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0, 0, 0, 0) if dark else (1, 1, 1, 0))
            axis.label.set_color(fg)
            axis.set_tick_params(colors=fg, labelsize=8)
        path_color = "#80cfff" if dark else "#123d78"
        if speed_color:
            segments = np.stack((trajectory[:-1], trajectory[1:]), axis=1)
            self._path_artist = Line3DCollection(
                segments, cmap="turbo", linewidth=0.75, alpha=0.9,
            )
            self._path_artist.set_array(speeds[:-1])
            self.ax.add_collection3d(self._path_artist)
        else:
            self._path_artist, = self.ax.plot(
                *trajectory.T, color=path_color, linewidth=0.7,
                alpha=0.88, rasterized=True,
            )
        mins, maxs = trajectory.min(axis=0), trajectory.max(axis=0)
        center = (mins + maxs) / 2
        span = max(float(np.max(maxs - mins)), 1e-3) * 0.58
        self.ax.set_xlim(center[0] - span, center[0] + span)
        self.ax.set_ylim(center[1] - span, center[1] + span)
        self.ax.set_zlim(center[2] - span, center[2] + span)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("x", labelpad=4)
        self.ax.set_ylabel("y", labelpad=4)
        self.ax.set_zlabel("z", labelpad=4)
        self.ax.view_init(elev=25, azim=-55)
        self.ax.set_title("3D PHASE SPACE  ·  DRAG TO ROTATE  ·  SCROLL TO ZOOM", color=fg,
                          fontsize=9, pad=12, fontweight="bold")
        self._marker, = self.ax.plot([], [], [], marker="o", markersize=5,
                                     color="#ff7b54" if dark else "#e84b2c", linestyle="None")
        self.figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
        self.canvas.draw_idle()

    def _draw_empty(self):
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def _advance(self):
        if not self.running or len(self.trajectory) == 0:
            return
        self._playback_remainder += 4.0 * self.playback_speed
        advance = int(self._playback_remainder)
        self._playback_remainder -= advance
        if advance == 0:
            return
        self.index = (self.index + advance) % len(self.trajectory)
        x, y, z = self.trajectory[self.index]
        self._marker.set_data([x], [y])
        self._marker.set_3d_properties([z])
        self.canvas.draw_idle()

    def set_running(self, running):
        self.running = bool(running)
        if self.running:
            self.timer.start()
        else:
            self.timer.stop()

    def set_playback_speed(self, speed):
        self.playback_speed = max(float(speed), 0.01)
        self._playback_remainder = 0.0

    def reset(self):
        self.index = 0
        if len(self.trajectory):
            x, y, z = self.trajectory[0]
            self._marker.set_data([x], [y])
            self._marker.set_3d_properties([z])
            self.canvas.draw_idle()

    def clear_trails(self):
        self.set_running(False)
        self.trajectory = np.empty((0, 3))
        if self._path_artist is not None:
            self._path_artist.remove()
            self._path_artist = None
        self._marker.set_data([], [])
        self._marker.set_3d_properties([])
        self.canvas.draw_idle()

    def reset_view(self):
        self.ax.view_init(elev=25, azim=-55)
        self.canvas.draw_idle()


ODE_MODEL_PRESETS = {
    "Harmonic oscillator": ("y", "-x", ""),
    "Damped oscillator": ("y", "-2*ζ*ω*y - ω^2*x", "ζ=0.1, ω=1"),
    "Van der Pol oscillator": ("y", "μ*(1-x^2)*y - x", "μ=1"),
    "Lotka–Volterra": (
        "α*x - β*x*y", "δ*x*y - γ*y",
        "α=1.1, β=0.4, δ=0.1, γ=0.4",
    ),
    "Brusselator": ("a - (b+1)*x + x^2*y", "b*x - x^2*y", "a=1, b=3"),
    "Linear saddle": ("x", "-y", ""),
}

ODE_3D_MODEL_PRESETS = {
    "Lorenz attractor": (
        "σ*(y-x)", "x*(ρ-z)-y", "x*y-β*z", "σ=10, ρ=28, β=2.6666666667", (0.1, 0.0, 0.0)
    ),
    "Rössler attractor": (
        "-y-z", "x+a*y", "b+(x-c)*z", "a=0.2, b=0.2, c=5.7", (1.0, 1.0, 1.0)
    ),
    "Chen attractor": (
        "a*(y-x)", "(c-a)*x-x*z+c*y", "x*y-b*z",
        "a=35, b=3, c=28", (-10.0, 0.0, 37.0),
    ),
    "Lü attractor": (
        "a*(y-x)", "-x*z+c*y", "x*y-b*z",
        "a=36, b=3, c=20", (1.0, 1.0, 1.0),
    ),
    "Thomas cyclic attractor": (
        "sin(y)-b*x", "sin(z)-b*y", "sin(x)-b*z",
        "b=0.208186", (0.1, 0.0, 0.0),
    ),
    "Chua double scroll": (
        "α*(y-(m1*x+0.5*(m0-m1)*(abs(x+1)-abs(x-1))))",
        "x-y+z", "-β*y",
        "α=9, β=14.286, m0=-0.1428571429, m1=0.2857142857",
        (0.1, 0.0, 0.0),
    ),
}


class ODEFlowApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODE Flow Lab — 2D / 3D Phase Portrait")
        self.resize(1360, 820)

        self.running = False
        self.fx = self.fy = None
        self.params = {}
        self._speed_max = 1.0
        self._syncing_view = False
        self._gpu_ready = False
        self._gpu_error = ""
        self._model_applied = False
        self._updating_model_inputs = False
        self._restoring_formula_history = False
        self._formula_history = {}
        self._previous_model_name = "Custom"
        self._custom_model_states = {
            "2D": ("x^3 - 2*x*y^2", "2*x^2*y - y^3", "", ""),
            "3D": ODE_3D_MODEL_PRESETS["Lorenz attractor"][:4],
        }
        self._dimension = "2D"
        self._mode_states = {
            "2D": ("Custom", "x^3 - 2*x*y^2", "2*x^2*y - y^3", "", ""),
            "3D": ("Lorenz attractor", *ODE_3D_MODEL_PRESETS["Lorenz attractor"][:4]),
        }
        self._auto_apply_timer = QTimer(self)
        self._auto_apply_timer.setSingleShot(True)
        self._auto_apply_timer.setInterval(450)
        self._auto_apply_timer.timeout.connect(self._apply_model_after_edit)
        self._static_refresh_timer = QTimer(self)
        self._static_refresh_timer.setSingleShot(True)
        self._static_refresh_timer.setInterval(140)
        self._static_refresh_timer.timeout.connect(self._refresh_static_background)

        self.build_ui()
        QTimer.singleShot(0, self.apply_model)

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(390)
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(6)

        title = QLabel("ODE Flow Lab")
        self.title_label = title
        title.setStyleSheet("font-size: 21px; font-weight: 700; color: #111111;")
        controls.addWidget(title)

        eq_box = QGroupBox("Differential equations")
        eq_form = QFormLayout(eq_box)
        self.dimension_selector = QComboBox()
        self.dimension_selector.addItems(["2D", "3D"])
        self.dimension_selector.currentTextChanged.connect(self._on_dimension_changed)
        eq_form.addRow("Dimension", self.dimension_selector)
        self.model_selector = QComboBox()
        self.model_selector.addItem("Custom")
        self.model_selector.addItems(ODE_MODEL_PRESETS)
        self.model_selector.currentTextChanged.connect(self._on_model_selected)
        eq_form.addRow("Model", self.model_selector)
        self.fx_input = QLineEdit(format_formula_input("x^3 - 2*x*y^2"))
        self.fy_input = QLineEdit(format_formula_input("2*x^2*y - y^3"))
        self.fz_input = QLineEdit()
        self.fx_input.setPlaceholderText("e.g. x^2 - y")
        self.fy_input.setPlaceholderText("e.g. x + y^2")
        for editor in (self.fx_input, self.fy_input, self.fz_input):
            self._formula_history[editor] = {"undo": [], "redo": [], "last": editor.text()}
            editor.installEventFilter(self)
            editor.textChanged.connect(self._on_formula_edited)
        eq_form.addRow("dx/dt =", self.fx_input)
        eq_form.addRow("dy/dt =", self.fy_input)
        self._fz_label = QLabel("dz/dt =")
        eq_form.addRow(self._fz_label, self.fz_input)
        for widget in (self._fz_label, self.fz_input):
            widget.hide()
        controls.addWidget(eq_box)

        keypad = QGroupBox("Calculator")
        keypad_layout = QGridLayout(keypad)
        keypad_layout.setHorizontalSpacing(4)
        keypad_layout.setVerticalSpacing(3)
        symbols = [
            ("x", "x"), ("y", "y"), ("z", "z"), ("a", "a"), ("b", "b"),
            ("+", "+"), ("−", "-"), ("×", "*"), ("÷", "/"),
            ("^", "^"), ("(", "("), (")", ")"), ("π", "pi"),
            ("sin", "sin("), ("cos", "cos("), ("tan", "tan("),
            ("exp", "exp("), ("sqrt", "sqrt("), ("log", "log("),
        ]
        for i, (label, value) in enumerate(symbols):
            btn = QPushButton(label)
            btn.setMaximumHeight(29)
            btn.clicked.connect(lambda checked=False, v=value: self.insert_symbol(v))
            keypad_layout.addWidget(btn, i // 6, i % 6)
        controls.addWidget(keypad)

        param_row = QHBoxLayout()
        param_row.setSpacing(8)
        param_label = QLabel("Params")
        param_label.setMinimumWidth(48)
        param_row.addWidget(param_label)
        self.param_input = QLineEdit()
        self.param_input.textChanged.connect(self._on_parameters_edited)
        param_row.addWidget(self.param_input, 1)
        controls.addLayout(param_row)

        settings = QGroupBox("View and motion")
        form = QGridLayout(settings)
        self._two_d_only_widgets = []
        form.setHorizontalSpacing(5)
        form.setVerticalSpacing(5)
        self.xmin = self.make_spin(-5, -1e9, 1e9, 0.5)
        self.xmax = self.make_spin(5, -1e9, 1e9, 0.5)
        self.ymin = self.make_spin(-5, -1e9, 1e9, 0.5)
        self.ymax = self.make_spin(5, -1e9, 1e9, 0.5)
        self._add_2d_setting(form, "x min", 0, 0)
        form.addWidget(self.xmin, 0, 1)
        self._two_d_only_widgets.append(self.xmin)
        self._add_2d_setting(form, "x max", 0, 2)
        form.addWidget(self.xmax, 0, 3)
        self._two_d_only_widgets.append(self.xmax)
        self._add_2d_setting(form, "y min", 1, 0)
        form.addWidget(self.ymin, 1, 1)
        self._two_d_only_widgets.append(self.ymin)
        self._add_2d_setting(form, "y max", 1, 2)
        form.addWidget(self.ymax, 1, 3)
        self._two_d_only_widgets.append(self.ymax)

        self.particle_count = NoWheelSpinBox()
        self.particle_count.setRange(20, 3000)
        self.particle_count.setValue(1800)
        self._add_2d_setting(form, "Particles", 2, 0)
        form.addWidget(self.particle_count, 2, 1)
        self._two_d_only_widgets.append(self.particle_count)

        self.trail_length = NoWheelSpinBox()
        self.trail_length.setRange(3, 300)
        self.trail_length.setValue(100)
        self._add_2d_setting(form, "Trail frames", 2, 2)
        form.addWidget(self.trail_length, 2, 3)
        self._two_d_only_widgets.append(self.trail_length)

        self.respawn_period = self.make_spin(8.0, 1.0, 60.0, 1.0)
        self.respawn_period.setSuffix(" s")
        self._add_2d_setting(form, "Respawn cycle", 4, 0, 1, 2)
        form.addWidget(self.respawn_period, 4, 2, 1, 2)
        self._two_d_only_widgets.append(self.respawn_period)

        self.dt = self.make_spin(0.01, 0.0001, 0.5, 0.005)
        form.addWidget(QLabel("Step"), 3, 0)
        form.addWidget(self.dt, 3, 1)
        self.speed = self.make_spin(1.0, 0.01, 10.0, 0.1)
        form.addWidget(QLabel("Speed"), 3, 2)
        form.addWidget(self.speed, 3, 3)
        controls.addWidget(settings)

        self.show_field = QCheckBox("Vectors")
        self.normalize = QCheckBox("Unit speed")
        self._two_d_only_widgets.extend((self.show_field, self.normalize))
        self.normalize.setChecked(True)
        self.speed_color = QCheckBox("Speed colors")
        self.speed_color.setToolTip(
            "Colors use the vector-field magnitude |F| on a log scale. "
            "Unit speed changes particle motion, not this magnitude."
        )
        self.dark_mode_check = QCheckBox("Dark mode")
        self.particle_count.valueChanged.connect(lambda _value: self._update_flow_runtime())
        self.trail_length.valueChanged.connect(lambda _value: self._update_flow_runtime())
        self.respawn_period.valueChanged.connect(lambda _value: self._update_flow_runtime())
        self.dt.valueChanged.connect(lambda _value: self._on_timestep_changed())
        self.speed.valueChanged.connect(lambda _value: self._on_speed_changed())
        self.normalize.toggled.connect(lambda _value: self._update_flow_runtime())
        self.speed_color.toggled.connect(self._on_visual_options_changed)
        self.dark_mode_check.toggled.connect(self._on_visual_options_changed)
        for spin in (self.xmin, self.xmax, self.ymin, self.ymax):
            spin.valueChanged.connect(self._bounds_changed_from_controls)
        self.show_field.toggled.connect(lambda _value: self._static_refresh_timer.start(120))
        view_options = QHBoxLayout()
        view_options.setSpacing(12)
        view_options.addWidget(self.show_field)
        view_options.addWidget(self.normalize)
        controls.addLayout(view_options)
        visual_options = QHBoxLayout()
        visual_options.setSpacing(12)
        visual_options.addWidget(self.speed_color)
        visual_options.addWidget(self.dark_mode_check)
        controls.addLayout(visual_options)

        buttons = QGridLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_animation)
        buttons.addWidget(self.start_btn, 0, 0, 1, 2)
        self.clear_btn = QPushButton("Clear trails")
        self.clear_btn.clicked.connect(self.clear_trails)
        buttons.addWidget(self.clear_btn, 1, 0, 1, 2)
        self.reset_view_btn = QPushButton("Reset view · 10 × 10")
        self.reset_view_btn.clicked.connect(self.reset_view)
        buttons.addWidget(self.reset_view_btn, 2, 0, 1, 2)
        controls.addLayout(buttons)

        self.status = QLabel("Ready · equations update as you edit")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #666666; padding: 5px 2px;")
        controls.addWidget(self.status)
        controls.addStretch()

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.flow_canvas = FlowCanvas()
        self.flow_canvas.gpuStatus.connect(self._on_gpu_status)
        self.flow_canvas.viewChanged.connect(self._on_canvas_view_changed)
        self.flow_canvas.canvasResized.connect(lambda: self._static_refresh_timer.start())
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.flow_canvas)
        self.canvas3d = Flow3DCanvas()
        self.view_stack.addWidget(self.canvas3d)
        plot_layout.addWidget(self.view_stack)

        splitter.addWidget(panel)
        splitter.addWidget(plot_panel)
        splitter.setSizes([360, 1000])

        self._mode_states["2D"] = self._read_editor_state()
        self._refresh_model_items("2D", "Custom")

        self._apply_theme(False)

    @staticmethod
    def make_spin(value, minimum, maximum, step):
        spin = NoWheelDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setKeyboardTracking(False)
        spin.setMinimumWidth(82)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _add_2d_setting(self, layout, text, row, column, row_span=1, column_span=1):
        label = QLabel(text)
        layout.addWidget(label, row, column, row_span, column_span)
        self._two_d_only_widgets.append(label)

    def _apply_theme(self, dark):
        background = "#000000" if dark else "#ffffff"
        foreground = "#f2f2f2" if dark else "#111111"
        secondary = "#d0d0d0" if dark else "#222222"
        border = "#454545" if dark else "#dedede"
        field_border = "#555555" if dark else "#d2d2d2"
        hover = "#252525" if dark else "#f1f1f1"
        split = "#333333" if dark else "#e7e7e7"
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {background}; color: {foreground}; }}
            QGroupBox {{ background: {background}; border: 1px solid {border}; border-radius: 6px;
                margin-top: 8px; padding: 8px 7px 6px; font-weight: 600; color: {foreground}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; }}
            QLabel {{ color: {secondary}; }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {background}; border: 1px solid {field_border};
                border-radius: 4px; padding: 4px 6px; color: {foreground}; selection-background-color: #777777; }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {foreground}; }}
            QPushButton {{ background: {background}; border: 1px solid {field_border}; border-radius: 4px;
                padding: 5px; color: {foreground}; font-weight: 600; }}
            QPushButton:hover {{ background: {hover}; border-color: #888888; }}
            QCheckBox {{ spacing: 8px; color: {secondary}; }}
            QCheckBox::indicator {{ width: 15px; height: 15px; }}
            QSplitter::handle {{ background: {split}; width: 1px; }}
        """)
        self.title_label.setStyleSheet(f"font-size: 21px; font-weight: 700; color: {foreground};")
        self.status.setStyleSheet(f"color: {secondary}; padding: 5px 2px;")

    def _on_visual_options_changed(self, _checked=False):
        dark = self.dark_mode_check.isChecked()
        self._apply_theme(dark)
        if hasattr(self, "flow_canvas"):
            self.flow_canvas.set_visual_options(self.speed_color.isChecked(), dark, self._speed_max)
        if self._dimension == "3D" and self._model_applied:
            self._auto_apply_timer.start(80)
        else:
            self._static_refresh_timer.start(80)

    def _set_view_bounds(self, bounds):
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        if xmin >= xmax or ymin >= ymax:
            return
        self._syncing_view = True
        blockers = [QSignalBlocker(spin) for spin in (self.xmin, self.xmax, self.ymin, self.ymax)]
        self.xmin.setValue(xmin)
        self.xmax.setValue(xmax)
        self.ymin.setValue(ymin)
        self.ymax.setValue(ymax)
        del blockers
        self._syncing_view = False
        self._active_bounds = (xmin, xmax, ymin, ymax)
        if hasattr(self, "flow_canvas"):
            self.flow_canvas.set_view_bounds(self._active_bounds)
        self._static_refresh_timer.start(130)

    def _on_canvas_view_changed(self, xmin, xmax, ymin, ymax):
        self._set_view_bounds((xmin, xmax, ymin, ymax))

    def _bounds_changed_from_controls(self, _value):
        if self._syncing_view or not hasattr(self, "flow_canvas"):
            return
        try:
            self._set_view_bounds(self.get_bounds())
        except ValueError:
            return

    def _read_editor_state(self):
        return (self.model_selector.currentText(), self.fx_input.text(), self.fy_input.text(),
                self.fz_input.text(), self.param_input.text())

    def _refresh_model_items(self, dimension, selected):
        blocker = QSignalBlocker(self.model_selector)
        self.model_selector.clear()
        self.model_selector.addItem("Custom")
        self.model_selector.addItems(ODE_MODEL_PRESETS if dimension == "2D" else ODE_3D_MODEL_PRESETS)
        self.model_selector.setCurrentText(selected)
        del blocker

    def _on_dimension_changed(self, dimension):
        if dimension == self._dimension or not hasattr(self, "model_selector"):
            return
        self._mode_states[self._dimension] = self._read_editor_state()
        self._dimension = dimension
        state = self._mode_states[dimension]
        self._refresh_model_items(dimension, state[0])
        self.view_stack.setCurrentIndex(0 if dimension == "2D" else 1)
        is_2d = dimension == "2D"
        for widget in self._two_d_only_widgets:
            widget.setVisible(is_2d)
        self.clear_btn.setText("Clear trails" if is_2d else "Clear path")
        self.reset_view_btn.setText("Reset view · 10 × 10" if is_2d else "Reset 3D view")
        for widget in (self._fz_label, self.fz_input):
            widget.setVisible(dimension == "3D")
        self._updating_model_inputs = True
        try:
            _, fx, fy, fz, params = state
            self.fx_input.setText(fx)
            self.fy_input.setText(fy)
            self.fz_input.setText(fz)
            self.param_input.setText(params)
            self._previous_model_name = state[0]
        finally:
            self._updating_model_inputs = False
        self._auto_apply_timer.start()

    def _on_model_selected(self, name):
        if self._updating_model_inputs:
            return
        presets = ODE_MODEL_PRESETS if self._dimension == "2D" else ODE_3D_MODEL_PRESETS
        if self._previous_model_name == "Custom" and name != "Custom":
            self._custom_model_states[self._dimension] = (
                self.fx_input.text(), self.fy_input.text(), self.fz_input.text(), self.param_input.text())
        self._updating_model_inputs = True
        try:
            if name == "Custom":
                values = self._custom_model_states[self._dimension]
            else:
                values = presets[name][:4] if self._dimension == "3D" else (*presets[name][:2], "", presets[name][2])
            self.fx_input.setText(values[0])
            self.fy_input.setText(values[1])
            self.fz_input.setText(values[2])
            self.param_input.setText(values[3])
        finally:
            self._updating_model_inputs = False
        self._previous_model_name = name
        self._mode_states[self._dimension] = self._read_editor_state()
        self._auto_apply_timer.start()

    def _on_formula_edited(self, text):
        sender = self.sender()
        history = self._formula_history[sender]
        if self._restoring_formula_history:
            history["last"] = text
        elif self._updating_model_inputs:
            history["undo"].clear()
            history["redo"].clear()
            history["last"] = text
        elif history["last"] != text:
            history["undo"].append(history["last"])
            del history["undo"][:-100]
            history["redo"].clear()
            history["last"] = text
        formatted = format_formula_input(text)
        if formatted != text:
            cursor = sender.cursorPosition()
            formatted_cursor = len(format_formula_input(text[:cursor]))
            blocker = QSignalBlocker(sender)
            sender.setText(formatted)
            sender.setCursorPosition(min(formatted_cursor, len(formatted)))
            del blocker
            text = formatted
        history["last"] = text
        if self._updating_model_inputs:
            return
        self._remember_custom_equations()

    def eventFilter(self, watched, event):
        if watched in self._formula_history and event.type() == QEvent.Type.KeyPress:
            modifiers = event.modifiers()
            is_control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            is_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            history = self._formula_history[watched]
            if is_control and event.key() == Qt.Key.Key_Z:
                source, destination = ("redo", "undo") if is_shift else ("undo", "redo")
            elif is_control and event.key() == Qt.Key.Key_Y:
                source, destination = "redo", "undo"
            else:
                source = None
            if source and history[source]:
                history[destination].append(watched.text())
                target = history[source].pop()
                self._restoring_formula_history = True
                try:
                    watched.setText(target)
                    watched.setCursorPosition(len(watched.text()))
                    history["last"] = watched.text()
                finally:
                    self._restoring_formula_history = False
                return True
        return super().eventFilter(watched, event)

    def _on_parameters_edited(self, _text):
        if self._updating_model_inputs:
            return
        self._remember_custom_equations()

    def _remember_custom_equations(self):
        self._custom_model_states[self._dimension] = (
            self.fx_input.text(), self.fy_input.text(), self.fz_input.text(), self.param_input.text())
        if self.model_selector.currentText() != "Custom":
            self.model_selector.setCurrentText("Custom")
        else:
            self._mode_states[self._dimension] = self._read_editor_state()
            self._auto_apply_timer.start()

    def _apply_model_after_edit(self):
        self.apply_model()

    def insert_symbol(self, value):
        widget = QApplication.focusWidget()
        if widget not in (self.fx_input, self.fy_input, self.fz_input):
            widget = self.fx_input
        pos = widget.cursorPosition()
        text = widget.text()
        widget.setText(text[:pos] + value + text[pos:])
        widget.setCursorPosition(min(len(format_formula_input(text[:pos] + value)), len(widget.text())))
        widget.setFocus()

    def read_model(self):
        self.fx = compile_formula(self.fx_input.text())
        self.fy = compile_formula(self.fy_input.text())
        self.params = parse_parameters(self.param_input.text())
        # Check formula at a vector point; catches most input mistakes.
        test_x = np.array([0.2, 0.5])
        test_y = np.array([0.3, -0.2])
        if self._dimension == "2D":
            evaluate(self.fx, test_x, test_y, self.params)
            evaluate(self.fy, test_x, test_y, self.params)
        else:
            self.fz = compile_formula(self.fz_input.text())
            evaluate_3d(self.fx, 0.2, 0.3, 0.4, self.params)
            evaluate_3d(self.fy, 0.2, 0.3, 0.4, self.params)
            evaluate_3d(self.fz, 0.2, 0.3, 0.4, self.params)

    def get_bounds(self):
        bounds = (self.xmin.value(), self.xmax.value(),
                  self.ymin.value(), self.ymax.value())
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("Each minimum must be smaller than its maximum.")
        return bounds

    def velocity(self, x, y):
        u = np.broadcast_to(evaluate(self.fx, x, y, self.params), np.shape(x)).copy()
        v = np.broadcast_to(evaluate(self.fy, x, y, self.params), np.shape(y)).copy()
        # Clip pathological velocities to keep the animation numerically bounded.
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(u, -1e6, 1e6), np.clip(v, -1e6, 1e6)

    def apply_model(self):
        was_running = self.running
        try:
            self.read_model()
            if was_running:
                self.stop_animation()
            if self._dimension == "3D":
                preset = ODE_3D_MODEL_PRESETS.get(self.model_selector.currentText())
                initial = preset[4] if preset else (0.1, 0.1, 0.1)
                self.canvas3d.set_model(
                    (self.fx, self.fy, self.fz), self.params, initial,
                    self.dt.value(), self.dark_mode_check.isChecked(),
                    self.speed_color.isChecked(), self.speed.value(),
                )
                self._model_applied = True
                self._gpu_ready = True
                self.status.setText("3D trajectory ready · drag to rotate, scroll to zoom.")
                if was_running:
                    self.toggle_animation()
                return
            config = {
                "vx": compile_glsl(self.fx, self.params),
                "vy": compile_glsl(self.fy, self.params),
                "bounds": self.get_bounds(),
                "particleCount": self.particle_count.value(),
                "trailLength": self.trail_length.value(),
                "respawnPeriod": self.respawn_period.value(),
                "dt": self.dt.value(),
                "speed": self.speed.value(),
                "normalize": self.normalize.isChecked(),
                "speedColor": self.speed_color.isChecked(),
                "darkMode": self.dark_mode_check.isChecked(),
            }
            self._active_bounds = config["bounds"]
            static_image, plot_rect = self.render_static_background(self._active_bounds)
            self.flow_canvas.set_model(
                config["vx"], config["vy"], config["bounds"],
                config["particleCount"], config["trailLength"], config["dt"], config["speed"],
                config["normalize"], config["respawnPeriod"], config["speedColor"],
                config["darkMode"], self._speed_max, static_image, plot_rect,
            )
            self._model_applied = True
            if self.flow_canvas.ready:
                self.status.setText("GPU particle flow ready.")
                if was_running:
                    self.toggle_animation()
            elif not self._gpu_error:
                self.status.setText("Preparing OpenGL flow…")
        except Exception as exc:
            self.status.setText(f"Input error · {exc}")

    def render_static_background(self, bounds):
        xmin, xmax, ymin, ymax = bounds
        dark = self.dark_mode_check.isChecked()
        background = "#000000" if dark else "#ffffff"
        primary = "#ededed" if dark else "#222222"
        secondary = "#888888" if dark else "#666666"
        grid_color = "#777777" if dark else "#aaaaaa"
        spine_color = "#393939" if dark else "#c8c8c8"
        ratio = self.flow_canvas.devicePixelRatioF()
        pixel_width = max(1, int(self.flow_canvas.width() * ratio))
        pixel_height = max(1, int(self.flow_canvas.height() * ratio))
        dpi = 144
        figure = Figure(
            figsize=(pixel_width / dpi, pixel_height / dpi),
            facecolor=background, dpi=dpi,
        )
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        ax.set_facecolor(background)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors=secondary, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(True, color=grid_color, alpha=0.10, linestyle="-")
        ax.axhline(0, color=secondary, linewidth=0.55, alpha=0.28)
        ax.axvline(0, color=secondary, linewidth=0.55, alpha=0.28)
        ax.set_xlabel("x", color=primary, labelpad=8)
        ax.set_ylabel("y", color=primary, labelpad=8)
        ax.set_title("PHASE FLOW", color=primary, pad=14, fontsize=10, fontweight="bold")
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.94)

        xx = np.linspace(xmin, xmax, 32)
        yy = np.linspace(ymin, ymax, 32)
        X, Y = np.meshgrid(xx, yy)
        U, V = self.velocity(X, Y)
        speeds = np.hypot(U, V)
        finite_speeds = speeds[np.isfinite(speeds)]
        self._speed_max = float(np.percentile(finite_speeds, 95)) if finite_speeds.size else 1.0
        if self._speed_max <= 1e-9:
            self._speed_max = 1.0

        if self.show_field.isChecked():
            mag = np.hypot(U, V)
            ax.quiver(
                X, Y, U / np.maximum(mag, 1e-10), V / np.maximum(mag, 1e-10),
                color=primary, alpha=0.42, pivot="mid", scale=38,
                width=0.002
            )
        canvas.draw()
        left, bottom, width, height = ax.get_position().bounds
        plot_rect = (left, left + width, bottom, bottom + height)
        output = io.BytesIO()
        canvas.print_png(output)
        figure.clear()
        return output.getvalue(), plot_rect

    def _refresh_static_background(self):
        if not self._model_applied or self._dimension != "2D":
            return
        try:
            background, plot_rect = self.render_static_background(self._active_bounds)
            self.flow_canvas.set_background(background, plot_rect)
            self.flow_canvas.speed_max = max(self._speed_max, 1e-9)
            self.flow_canvas.update()
        except Exception as exc:
            self.status.setText(f"Plot refresh failed: {exc}")

    def _on_gpu_status(self, message):
        self._gpu_ready = message == "GPU particle flow ready."
        self._gpu_error = "" if self._gpu_ready else message
        self.status.setText(message)

    def _update_flow_runtime(self):
        if self._dimension == "2D" and hasattr(self, "flow_canvas"):
            self.flow_canvas.update_runtime(
                self.particle_count.value(), self.trail_length.value(),
                self.dt.value(), self.speed.value(), self.normalize.isChecked(),
                self.respawn_period.value(),
            )

    def _on_timestep_changed(self):
        if self._dimension == "2D":
            self._update_flow_runtime()
        elif self._model_applied:
            self._auto_apply_timer.start()

    def _on_speed_changed(self):
        if self._dimension == "2D":
            self._update_flow_runtime()
        elif hasattr(self, "canvas3d"):
            self.canvas3d.set_playback_speed(self.speed.value())

    def reset_view(self):
        if self._dimension == "2D":
            self._set_view_bounds((-5.0, 5.0, -5.0, 5.0))
        else:
            self.canvas3d.reset_view()

    def clear_trails(self):
        if self._dimension == "2D":
            self.flow_canvas.clear_trails()
        elif self._model_applied:
            self.canvas3d.clear_trails()

    def stop_animation(self):
        self.running = False
        if hasattr(self, "flow_canvas"):
            self.flow_canvas.set_running(False)
        if hasattr(self, "canvas3d"):
            self.canvas3d.set_running(False)
        if hasattr(self, "start_btn"):
            self.start_btn.setText("Start")

    def toggle_animation(self):
        if self.running:
            self.stop_animation()
            self.status.setText("Paused.")
            return
        if self._dimension == "2D" and not self.flow_canvas.ready:
            self.status.setText("OpenGL is still starting.")
            return
        self.running = True
        self.start_btn.setText("Pause")
        self.status.setText("Flowing..." if self._dimension == "2D" else "Tracing 3D trajectory...")
        if self._dimension == "2D":
            self.flow_canvas.set_running(True)
        else:
            self.canvas3d.set_running(True)


def main():
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setVersion(3, 3)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface_format.setDepthBufferSize(0)
    surface_format.setStencilBufferSize(0)
    surface_format.setSamples(0)
    QSurfaceFormat.setDefaultFormat(surface_format)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ODEFlowApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
