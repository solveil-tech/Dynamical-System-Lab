import ast
import copy
import ctypes
import html
import io
import json
import itertools
import math
import operator
import re
import sys
import time
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QSize, QRect, QSignalBlocker, QEvent
from PySide6.QtGui import QFontDatabase, QImage, QPixmap, QPainter, QSurfaceFormat, QVector4D
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
    QSplitter, QAbstractSpinBox, QComboBox, QStackedWidget, QSizePolicy, QFrame, QScrollArea,
    QTabWidget, QFileDialog,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from flow_tools import (
    integrate_trajectory, trajectory_curve_points, speed_colormap,
    speed_normalization, validate_experiment,
)


def _upload_rgba8(pixels, width, x_offset=0, y_offset=0, height=1):
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
        0x0DE1, 0, int(x_offset), int(y_offset), int(width), int(height), 0x1908, 0x1401,
        ctypes.c_void_p(int(pixels.ctypes.data)),
    )


# ============================================================
# Safe expression parser: no eval(), no attribute access/imports
# ============================================================

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
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


_FORMULA_TOKEN_PATTERN = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|(?:[^\W\d]|_)\w*|\*\*|[()+\-*/%^,]",
    re.UNICODE,
)


def _split_implicit_identifier(value, parameter_names, followed_by_call=False):
    known = set(parameter_names) | {"x", "y", "z", "r", "θ", "theta", "pi", "e"} | set(FUNCTIONS)
    if value in known:
        return None
    atoms = {"x", "y", "z", "r", "θ", "theta", "pi", "e"} | set(parameter_names)
    atoms.update(symbol for symbol in LATEX_SYMBOLS.values() if len(symbol) == 1)
    if followed_by_call:
        atoms.update(FUNCTIONS)
    atoms = sorted(atoms, key=len, reverse=True)

    def split_from(index):
        if index == len(value):
            return []
        candidates = [atom for atom in atoms if value.startswith(atom, index)]
        if followed_by_call and index == 0:
            candidates = [atom for atom in candidates if atom not in FUNCTIONS]
        for atom in candidates:
            rest = split_from(index + len(atom))
            if rest is not None and (atom not in FUNCTIONS or index + len(atom) == len(value)):
                return [atom, *rest]
        if index > 0:
            number = re.match(r"\d+(?:\.\d*)?|\.\d+", value[index:])
            if number:
                rest = split_from(index + len(number.group()))
                if rest is not None:
                    return [number.group(), *rest]
        return None

    parts = split_from(0)
    return parts if parts and len(parts) > 1 else None


def _insert_implicit_multiplication(text, parameter_names=(), preserve_scientific_prefix=False):
    """Turn adjacent math atoms into explicit products for safe AST parsing."""
    raw_tokens = []
    cursor = 0
    for match in _FORMULA_TOKEN_PATTERN.finditer(text):
        if text[cursor:match.start()].strip():
            return text
        value = match.group()
        kind = "number" if value[0].isdigit() or value[0] == "." else "operator"
        if value[0].isalpha() or value[0] == "_" or value[0] in LATEX_SYMBOLS.values():
            kind = "name"
        if value == "(":
            kind = "open"
        elif value == ")":
            kind = "close"
        raw_tokens.append((kind, value, match.start(), match.end()))
        cursor = match.end()
    if text[cursor:].strip():
        return text

    tokens = []
    for index, (kind, value, start, end) in enumerate(raw_tokens):
        if kind != "name":
            tokens.append((kind, value, start, end))
            continue
        next_is_call = index + 1 < len(raw_tokens) and raw_tokens[index + 1][1] == "("
        parts = _split_implicit_identifier(value, parameter_names, next_is_call)
        if parts is None:
            tokens.append((kind, value, start, end))
        else:
            offset = start
            for part in parts:
                part_kind = "number" if part[0].isdigit() or part[0] == "." else "name"
                tokens.append((part_kind, part, offset, offset + len(part)))
                offset += len(part)

    insert_positions = []
    for previous, current in zip(tokens, tokens[1:]):
        left_kind, left_value, _, _ = previous
        right_kind, _, right_start, _ = current
        left_can_end = left_kind in {"number", "name", "close"}
        right_can_start = right_kind in {"number", "name", "open"}
        is_function_call = left_kind == "name" and left_value in FUNCTIONS and right_kind == "open"
        is_scientific_prefix = (
            preserve_scientific_prefix
            and left_kind == "number"
            and right_kind == "name"
            and current[1] in {"e", "E"}
            and previous[3] == right_start
        )
        if left_can_end and right_can_start and not is_function_call and not is_scientific_prefix:
            insert_positions.append(right_start)

    if not insert_positions:
        return text
    output = []
    cursor = 0
    for position in insert_positions:
        output.append(text[cursor:position])
        output.append("*")
        cursor = position
    output.append(text[cursor:])
    return "".join(output)


def format_formula_input(text, parameter_names=()):
    """Render powers and implicit products as typographic math in-place."""
    text = normalize_display_expression(text)
    text = _insert_implicit_multiplication(
        text, parameter_names, preserve_scientific_prefix=True
    ).replace("**", "^")

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
    # The editor may receive the two asterisks of x**2 in separate keystrokes.
    # Normalize the completed numeric power once more so it always displays as a superscript.
    text = re.sub(
        r"(?P<base>(?:[^\W\d_]\w*|\([^()]*\)))\^(?P<exponent>[+-]?\d+)",
        lambda match: match.group("base") + "".join(
            ASCII_TO_SUPERSCRIPT.get(char, char) for char in match.group("exponent")
        ),
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r"\bpi\b", "π", text)
    text = re.sub(
        r"\b([A-Za-z]+)(\d+)\b",
        lambda m: m.group(1) + "".join(ASCII_TO_SUBSCRIPT.get(c, c) for c in m.group(2)),
        text,
    )
    return text.replace("*", "·").replace("-", "−")
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


def compile_formula(text, parameter_names=()):
    text = normalize_display_expression(text)
    text = _insert_implicit_multiplication(text, parameter_names).replace("^", "**")
    if not text:
        raise ValueError("Formula cannot be empty.")
    return ast.parse(text, mode="eval").body


def _simplify_coordinate_expression(node):
    """Simplify arithmetic identities without changing undefined divisions."""
    if isinstance(node, ast.UnaryOp):
        operand = _simplify_coordinate_expression(node.operand)
        if isinstance(operand, ast.Constant):
            return ast.Constant(UNARY_OPS[type(node.op)](operand.value))
        return ast.UnaryOp(op=node.op, operand=operand)
    if not isinstance(node, ast.BinOp):
        return node
    left = _simplify_coordinate_expression(node.left)
    right = _simplify_coordinate_expression(node.right)
    is_number = lambda value, number: isinstance(value, ast.Constant) and value.value == number
    if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
        try:
            value = BINARY_OPS[type(node.op)](left.value, right.value)
            if isinstance(value, (int, float)) and math.isfinite(value):
                return ast.Constant(value)
        except (ArithmeticError, KeyError, ValueError):
            pass
    if isinstance(node.op, ast.Mult):
        if is_number(left, 0) or is_number(right, 0): return ast.Constant(0)
        if is_number(left, 1): return right
        if is_number(right, 1): return left
    if isinstance(node.op, (ast.Add, ast.Sub)):
        if is_number(right, 0): return left
        if is_number(left, 0):
            return right if isinstance(node.op, ast.Add) else ast.UnaryOp(op=ast.USub(), operand=right)
    if isinstance(node.op, ast.Div) and is_number(right, 1): return left
    if isinstance(node.op, ast.Pow):
        if is_number(right, 0): return ast.Constant(1)
        if is_number(right, 1): return left
    return ast.BinOp(left=left, op=node.op, right=right)


def _remove_radial_factor(node):
    """Return node/r when an explicit radial factor can be cancelled."""
    if isinstance(node, ast.Constant) and node.value == 0:
        return node
    if isinstance(node, ast.Name) and node.id == "r":
        return ast.Constant(1)
    if isinstance(node, ast.UnaryOp):
        inner = _remove_radial_factor(node.operand)
        return None if inner is None else ast.UnaryOp(op=node.op, operand=inner)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow) and isinstance(node.left, ast.Name) and node.left.id == "r":
            if isinstance(node.right, ast.Constant) and node.right.value >= 1:
                return ast.BinOp(left=node.left, op=ast.Pow(), right=ast.Constant(node.right.value-1))
        left = _remove_radial_factor(node.left)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            right = _remove_radial_factor(node.right)
            if left is not None and right is not None:
                return ast.BinOp(left=left, op=node.op, right=right)
        elif isinstance(node.op, ast.Mult):
            if left is not None:
                return ast.BinOp(left=left, op=ast.Mult(), right=node.right)
            right = _remove_radial_factor(node.right)
            if right is not None:
                return ast.BinOp(left=node.left, op=ast.Mult(), right=right)
        elif isinstance(node.op, ast.Div) and left is not None:
            return ast.BinOp(left=left, op=ast.Div(), right=node.right)
    return None


def polar_to_cartesian(radial, angular, params):
    """Convert r'=F(r,theta), theta'=G(r,theta) into an x,y vector field.

    For r>0: x'=(F/r)x-Gy, y'=(F/r)y+Gx. Explicit radial
    factors are cancelled before substitution so common smooth systems
    (rotation, spiral sinks, Hopf) remain well-defined at the origin.
    """
    allowed = {"r", "θ", "pi", "e"} | set(params) | set(FUNCTIONS)
    for field in (radial, angular):
        unknown = {node.id for node in ast.walk(field) if isinstance(node, ast.Name)} - allowed
        if unknown:
            raise ValueError("Polar equations use r and θ; unknown symbol: " + ", ".join(sorted(unknown)))
    quotient = _remove_radial_factor(radial)
    if quotient is None:
        quotient = ast.BinOp(left=radial, op=ast.Div(), right=ast.Name(id="r", ctx=ast.Load()))
    square = compile_formula("x^2+y^2")

    class SubstitutePolar(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "r":
                return ast.Call(func=ast.Name(id="sqrt", ctx=ast.Load()), args=[copy.deepcopy(square)], keywords=[])
            if node.id == "θ":
                return ast.Call(func=ast.Name(id="atan2", ctx=ast.Load()),
                                args=[ast.Name(id="y", ctx=ast.Load()), ast.Name(id="x", ctx=ast.Load())], keywords=[])
            return node

        def visit_BinOp(self, node):
            if (isinstance(node.op, ast.Pow) and isinstance(node.left, ast.Name)
                    and node.left.id == "r" and isinstance(node.right, ast.Constant)
                    and isinstance(node.right.value, (int, float))
                    and node.right.value >= 0 and float(node.right.value).is_integer()
                    and int(node.right.value) % 2 == 0):
                return _simplify_coordinate_expression(ast.BinOp(
                    left=copy.deepcopy(square), op=ast.Pow(), right=ast.Constant(int(node.right.value)//2)))
            return self.generic_visit(node)

    quotient = SubstitutePolar().visit(copy.deepcopy(_simplify_coordinate_expression(quotient)))
    angular = SubstitutePolar().visit(copy.deepcopy(angular))
    x, y = ast.Name(id="x", ctx=ast.Load()), ast.Name(id="y", ctx=ast.Load())
    fx = ast.BinOp(left=ast.BinOp(left=x, op=ast.Mult(), right=quotient), op=ast.Sub(),
                   right=ast.BinOp(left=y, op=ast.Mult(), right=angular))
    fy = ast.BinOp(left=ast.BinOp(left=y, op=ast.Mult(), right=quotient), op=ast.Add(),
                   right=ast.BinOp(left=x, op=ast.Mult(), right=angular))
    return tuple(ast.fix_missing_locations(_simplify_coordinate_expression(field)) for field in (fx, fy))


def coordinate_formula_for_editor(field, parameter_names=()):
    """Export an expression while keeping compound exponents grouped."""
    marker = "__ode_power_group__"

    class GroupPowers(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.BinOp):
                node.right = ast.Call(func=ast.Name(id=marker, ctx=ast.Load()),
                                      args=[node.right], keywords=[])
            return node

    grouped = GroupPowers().visit(copy.deepcopy(field))
    text = ast.unparse(ast.fix_missing_locations(grouped)).replace(marker + "(", "(")
    return format_formula_input(text, parameter_names)


def evaluate(tree, x, y, params):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        result = eval_node(tree, {"x": x, "y": y, **params})
    return np.asarray(result, dtype=float)


def evaluate_3d(tree, x, y, z, params):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        result = eval_node(tree, {"x": x, "y": y, "z": z, **params})
    return np.asarray(result, dtype=float)


def _field_vector(fields, point, params):
    variables = dict(zip(("x", "y", "z"), point))
    return np.asarray([eval_node(field, variables | params) for field in fields], dtype=float)


def _value_and_gradient(node, variables, params):
    """Evaluate an expression and its analytic gradient from its AST."""
    dimension = len(variables)
    zero = np.zeros(dimension, dtype=float)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value), zero
    if isinstance(node, ast.Name):
        if node.id in variables:
            gradient = zero.copy()
            gradient[tuple(variables).index(node.id)] = 1.0
            return float(variables[node.id]), gradient
        constants = {"pi": math.pi, "e": math.e}
        if node.id in constants:
            return constants[node.id], zero
        if node.id in params:
            return float(params[node.id]), zero
        return None
    if isinstance(node, ast.UnaryOp):
        result = _value_and_gradient(node.operand, variables, params)
        if result is None:
            return None
        value, gradient = result
        if isinstance(node.op, ast.UAdd):
            return value, gradient
        if isinstance(node.op, ast.USub):
            return -value, -gradient
        return None
    if isinstance(node, ast.BinOp):
        left = _value_and_gradient(node.left, variables, params)
        right = _value_and_gradient(node.right, variables, params)
        if left is None or right is None:
            return None
        a, da = left
        b, db = right
        try:
            if isinstance(node.op, ast.Add):
                return a + b, da + db
            if isinstance(node.op, ast.Sub):
                return a - b, da - db
            if isinstance(node.op, ast.Mult):
                return a * b, da * b + db * a
            if isinstance(node.op, ast.Div) and b != 0:
                return a / b, (da * b - db * a) / (b * b)
            if isinstance(node.op, ast.Pow):
                value = a ** b
                if not np.any(db):
                    if b == 0:
                        return value, zero.copy()
                    return value, b * (a ** (b - 1)) * da
                if a > 0:
                    return value, value * (db * math.log(a) + b * da / a)
        except (ArithmeticError, FloatingPointError, TypeError, ValueError, OverflowError):
            return None
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [_value_and_gradient(arg, variables, params) for arg in node.args]
        if any(arg is None for arg in args):
            return None
        values = [arg[0] for arg in args]
        gradients = [arg[1] for arg in args]
        try:
            value = float(FUNCTIONS[name](*values))
            if name in ("sin",): derivative = math.cos(values[0])
            elif name in ("cos",): derivative = -math.sin(values[0])
            elif name == "tan": derivative = 1 / math.cos(values[0]) ** 2
            elif name in ("asin",): derivative = 1 / math.sqrt(1 - values[0] ** 2)
            elif name == "acos": derivative = -1 / math.sqrt(1 - values[0] ** 2)
            elif name == "atan": derivative = 1 / (1 + values[0] ** 2)
            elif name == "atan2" and len(values) == 2:
                denominator = values[0] ** 2 + values[1] ** 2
                if denominator == 0: return None
                return value, (values[1]*gradients[0] - values[0]*gradients[1])/denominator
            elif name == "sinh": derivative = math.cosh(values[0])
            elif name == "cosh": derivative = math.sinh(values[0])
            elif name == "tanh": derivative = 1 - value ** 2
            elif name == "exp": derivative = value
            elif name in ("log", "ln"): derivative = 1 / values[0]
            elif name == "log10": derivative = 1 / (values[0] * math.log(10))
            elif name == "sqrt": derivative = 0.5 / value
            elif name == "abs":
                if values[0] == 0: return None
                derivative = 1.0 if values[0] > 0 else -1.0
            elif name in ("floor", "ceil", "sign"):
                if name == "sign" and values[0] == 0: return None
                if name in ("floor", "ceil") and values[0] == round(values[0]): return None
                derivative = 0.0
            elif name in ("minimum", "maximum") and len(values) == 2:
                if values[0] == values[1]: return None
                index = (0 if values[0] < values[1] else 1) if name == "minimum" else (0 if values[0] > values[1] else 1)
                return value, gradients[index]
            else:
                return None
            return value, derivative * gradients[0]
        except (ArithmeticError, FloatingPointError, TypeError, ValueError, OverflowError):
            return None
    return None


def _field_jacobian(fields, point, params):
    point = np.asarray(point, dtype=float)
    variables = dict(zip(("x", "y", "z"), point))
    rows = [_value_and_gradient(field, variables, params) for field in fields]
    if any(row is None for row in rows):
        return None
    return np.asarray([row[1] for row in rows], dtype=float)


def find_equilibria(fields, params, bounds, grid_size=None, x_axis_only=False):
    """Find isolated equilibria from a grid of Newton seeds within a search box."""
    dimension = len(fields)
    bounds = np.asarray(bounds, dtype=float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    spans = upper - lower
    if np.any(spans <= 0):
        return []
    if grid_size is None:
        grid_size = 9 if dimension == 2 else 5
    axes = [np.linspace(lo, hi, grid_size) for lo, hi in bounds]
    seeds = list(itertools.product(*axes))
    if x_axis_only and dimension == 2:
        seeds = [(x, 0.0) for x in axes[0]]
    midpoint = (lower + upper) / 2.0
    if x_axis_only and dimension == 2:
        if not lower[1] <= 0 <= upper[1]:
            return []
        midpoint[1] = 0.0
    seeds.extend([np.zeros(dimension), midpoint])
    scale = max(1.0, float(np.max(spans)))
    field_scale = np.finfo(float).tiny
    for seed in seeds:
        try:
            seed_value = _field_vector(fields, seed, params)
            if np.all(np.isfinite(seed_value)):
                field_scale = max(field_scale, float(np.linalg.norm(seed_value)))
        except (ArithmeticError, FloatingPointError, TypeError, ValueError):
            pass
    residual_tolerance = 1e-12 * field_scale
    point_tolerance = 1e-7 * scale
    accepted = []

    for seed in seeds:
        point = np.asarray(seed, dtype=float)
        converged = False
        for _ in range(65):
            try:
                value = _field_vector(fields, point, params)
                if not np.all(np.isfinite(value)):
                    break
                residual = float(np.linalg.norm(value, ord=2))
                if residual == 0.0:
                    converged = True
                    break
                jacobian = _field_jacobian(fields, point, params)
                if jacobian is None:
                    break
                if x_axis_only and dimension == 2:
                    reduced = np.linalg.lstsq(jacobian[:, :1], value, rcond=None)[0]
                    step = np.array([reduced[0], 0.0])
                else:
                    step = np.linalg.lstsq(jacobian, value, rcond=None)[0]
            except (ArithmeticError, FloatingPointError, TypeError, ValueError, np.linalg.LinAlgError):
                break
            if not np.all(np.isfinite(step)):
                break
            step_norm = float(np.linalg.norm(step))
            if residual <= residual_tolerance and step_norm <= point_tolerance:
                converged = True
                break
            if residual == 0.0:
                converged = True
                break
            if step_norm > 2.0 * scale:
                step *= (2.0 * scale / step_norm)
            accepted_step = False
            for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                candidate = point - damping * step
                if x_axis_only and dimension == 2:
                    candidate[1] = 0.0
                if np.any(candidate < lower - 5.0 * spans) or np.any(candidate > upper + 5.0 * spans):
                    continue
                try:
                    candidate_residual = float(np.linalg.norm(_field_vector(fields, candidate, params)))
                except (ArithmeticError, FloatingPointError, TypeError, ValueError):
                    continue
                if np.isfinite(candidate_residual) and candidate_residual < residual:
                    point = candidate
                    accepted_step = True
                    break
            if not accepted_step:
                break

        if not converged:
            try:
                final_value = _field_vector(fields, point, params)
                final_jacobian = _field_jacobian(fields, point, params)
                if final_jacobian is None:
                    continue
                final_step = np.linalg.lstsq(
                    final_jacobian[:, :1] if x_axis_only and dimension == 2 else final_jacobian,
                    final_value, rcond=None,
                )[0]
                converged = (
                    np.all(np.isfinite(final_value))
                    and np.linalg.norm(final_value) <= residual_tolerance
                    and np.linalg.norm(final_step) <= point_tolerance
                )
            except (ArithmeticError, FloatingPointError, TypeError, ValueError, np.linalg.LinAlgError):
                converged = False
        if not converged or np.any(point < lower - point_tolerance) or np.any(point > upper + point_tolerance):
            continue
        duplicate = next(
            (index for index, existing in enumerate(accepted)
             if np.linalg.norm(point - existing) <= 1e-5 * scale),
            None,
        )
        if duplicate is None:
            accepted.append(point)
        else:
            # Prefer the candidate with the smaller residual when several
            # seeds converge into the same numerical root cluster.
            existing = accepted[duplicate]
            if np.linalg.norm(_field_vector(fields, point, params)) < np.linalg.norm(
                _field_vector(fields, existing, params)
            ):
                accepted[duplicate] = point

    accepted.sort(key=lambda point: tuple(point))
    return [(point, classify_equilibrium(fields, point, params)) for point in accepted]


def _expression_degree(node, variables):
    if isinstance(node, ast.Constant):
        return 0
    if isinstance(node, ast.Name):
        return 1 if node.id in variables else 0
    if isinstance(node, ast.UnaryOp):
        return _expression_degree(node.operand, variables)
    if isinstance(node, ast.BinOp):
        left = _expression_degree(node.left, variables)
        right = _expression_degree(node.right, variables)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return max(left, right)
        if isinstance(node.op, ast.Mult):
            return left + right
        if isinstance(node.op, ast.Div) and right == 0:
            return left
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            exponent = node.right.value
            if isinstance(exponent, (int, float)) and exponent >= 0:
                return int(left * exponent)
        return math.inf
    if isinstance(node, ast.Call):
        return 0 if all(_expression_degree(arg, variables) == 0 for arg in node.args) else math.inf
    return math.inf


def _is_affine_field(fields):
    return all(_expression_degree(field, {"x", "y", "z"}) <= 1 for field in fields)


def classify_equilibrium(fields, point, params):
    """Return the local type and whether its linearization is hyperbolic."""
    dimension = len(fields)
    jacobian = _field_jacobian(fields, point, params)
    if jacobian is None or not np.all(np.isfinite(jacobian)):
        return {"type": "Jacobian unavailable", "hyperbolic": None,
                "linearization": "Inconclusive (Jacobian unavailable).",
                "eigenvalues": "unavailable", "jacobian": jacobian}
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    # Componentwise backward-error scales avoid erasing slow eigenvalues just
    # because another direction is very fast, e.g. diag(-1, -1e9).
    action = np.linalg.norm(np.abs(jacobian) @ np.abs(eigenvectors), axis=0)
    residual = np.linalg.norm(jacobian @ eigenvectors - eigenvectors * eigenvalues, axis=0)
    tolerance = residual + 32 * np.finfo(float).eps * np.maximum(np.finfo(float).tiny, action)
    real_parts = eigenvalues.real
    positive = int(np.count_nonzero(real_parts > tolerance))
    negative = int(np.count_nonzero(real_parts < -tolerance))
    neutral = dimension - positive - negative
    hyperbolic = neutral == 0
    has_complex = bool(np.any(np.abs(eigenvalues.imag) > tolerance))
    if neutral:
        zero_modes = int(np.count_nonzero(np.abs(eigenvalues) <= tolerance))
        if positive and negative:
            kind = "Saddle-type with neutral mode(s)"
        elif zero_modes:
            if negative:
                kind = f"Degenerate type ({zero_modes} zero, {negative} stable mode(s))"
            elif positive:
                kind = f"Degenerate type ({zero_modes} zero, {positive} unstable mode(s))"
            else:
                kind = "Degenerate equilibrium"
        elif positive:
            kind = "Unstable type with neutral mode(s)"
        elif has_complex and zero_modes == 0:
            if negative:
                kind = "Center-stable type (imaginary and stable modes)"
            elif positive:
                kind = "Center-unstable type (imaginary and unstable modes)"
            else:
                kind = "Center-type (purely imaginary linear modes)"
        else:
            kind = "Non-hyperbolic type (stable and neutral modes)"
        linearization = (
            "Unstable (positive-real-part eigenvalue)." if positive else
            "Inconclusive for nonlinear stability (non-hyperbolic)."
        )
    else:
        if positive and negative:
            kind = "Saddle-focus" if has_complex else "Saddle"
            linearization = "Unstable (hyperbolic saddle)."
        elif negative == dimension:
            kind = "Stable focus" if has_complex else "Stable node"
            linearization = "Locally asymptotically stable (hyperbolic sink)."
        elif positive == dimension:
            kind = "Unstable focus" if has_complex else "Unstable node"
            linearization = "Unstable (hyperbolic source)."
        else:
            kind = "Hyperbolic equilibrium"
            linearization = "Unstable (hyperbolic, with an expanding direction)."
    def display_eigenvalue(value, uncertainty):
        real = 0.0 if abs(value.real) <= uncertainty else float(value.real)
        imaginary = 0.0 if abs(value.imag) <= uncertainty else float(value.imag)
        if imaginary == 0.0:
            return f"{real:.4g}"
        return f"{real:.4g}{imaginary:+.4g}i"

    eigen_text = ", ".join(display_eigenvalue(value, tol) for value, tol in zip(eigenvalues, tolerance))
    return {
        "type": kind,
        "hyperbolic": hyperbolic,
        "linearization": linearization,
        "eigenvalues": eigen_text,
        "jacobian": jacobian,
    }


def _poly_add(left, right, right_scale=1.0):
    result = dict(left)
    for powers, coefficient in right.items():
        result[powers] = result.get(powers, 0.0) + right_scale * coefficient
    return {powers: value for powers, value in result.items() if abs(value) > 1e-14}


def _poly_multiply(left, right, max_terms=2000):
    result = {}
    for left_powers, left_coefficient in left.items():
        for right_powers, right_coefficient in right.items():
            powers = tuple(a + b for a, b in zip(left_powers, right_powers))
            result[powers] = result.get(powers, 0.0) + left_coefficient * right_coefficient
            if len(result) > max_terms:
                raise ValueError("Polynomial expansion exceeded its term limit.")
    return {powers: value for powers, value in result.items() if abs(value) > 1e-14}


def _polynomial_from_ast(node, point, params):
    dimension = len(point)
    zero = (0,) * dimension
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {zero: float(node.value)}
    if isinstance(node, ast.Name):
        if node.id in ("x", "y", "z")[:dimension]:
            index = ("x", "y", "z").index(node.id)
            powers = [0] * dimension
            powers[index] = 1
            return {zero: float(point[index]), tuple(powers): 1.0}
        if node.id in params:
            return {zero: float(params[node.id])}
        if node.id == "pi":
            return {zero: math.pi}
        if node.id == "e":
            return {zero: math.e}
        return None
    if isinstance(node, ast.UnaryOp):
        value = _polynomial_from_ast(node.operand, point, params)
        if value is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return {powers: -coefficient for powers, coefficient in value.items()}
        return None
    if isinstance(node, ast.BinOp):
        left = _polynomial_from_ast(node.left, point, params)
        right = _polynomial_from_ast(node.right, point, params)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return _poly_add(left, right)
        if isinstance(node.op, ast.Sub):
            return _poly_add(left, right, -1.0)
        if isinstance(node.op, ast.Mult):
            return _poly_multiply(left, right)
        if isinstance(node.op, ast.Div):
            denominator = right.get(zero, 0.0)
            if len(right) != 1 or abs(denominator) < 1e-14:
                return None
            return {powers: coefficient / denominator for powers, coefficient in left.items()}
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            exponent = node.right.value
            if not isinstance(exponent, int) or exponent < 0 or exponent > 8:
                return None
            result = {zero: 1.0}
            for _ in range(exponent):
                result = _poly_multiply(result, left)
            return result
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
        args = [_polynomial_from_ast(arg, point, params) for arg in node.args]
        if any(arg is None or len(arg) > 1 for arg in args):
            return None
        values = [arg.get(zero, 0.0) for arg in args]
        try:
            result = float(FUNCTIONS[node.func.id](*values))
        except (ArithmeticError, FloatingPointError, TypeError, ValueError):
            return None
        return {zero: result} if np.isfinite(result) else None
    return None


def _polynomial_lie_derivative(fields, point, params, candidate):
    """Symbolically expand the Lie derivative of a polynomial candidate."""
    dimension = len(fields)
    zero = (0,) * dimension
    vector_field = [_polynomial_from_ast(field, point, params) for field in fields]
    if any(component is None for component in vector_field):
        return None
    lie = {}
    for index, component in enumerate(vector_field):
        derivative = {}
        for powers, coefficient in candidate.items():
            if powers[index] == 0:
                continue
            reduced = list(powers)
            reduced[index] -= 1
            key = tuple(reduced)
            derivative[key] = derivative.get(key, 0.0) + coefficient * powers[index]
        lie = _poly_add(lie, _poly_multiply(derivative, component))

    magnitude = max(1.0, max((abs(value) for value in lie.values()), default=0.0))
    active = {power: value for power, value in lie.items()
              if abs(value) > 1e-10 * magnitude}
    at_origin = np.allclose(point, 0.0, atol=1e-10)
    names = (("x", "y", "z") if at_origin else ("δx", "δy", "δz"))[:dimension]
    covered = set()
    terms = []
    for powers, coefficient in sorted(
            active.items(),
            key=lambda item: (next((i for i, power in enumerate(item[0]) if power), dimension),
                              sum(item[0]), item[0])):
        if coefficient >= 0 or any(power % 2 for power in powers) or not any(powers):
            return None
        for index, power in enumerate(powers):
            if power and sum(powers) == power:
                covered.add(index)
        factors = [name if power == 1 else
                   name + str(power).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
                   for name, power in zip(names, powers) if power]
        terms.append(f" − {abs(coefficient):.4g}" + "·".join(factors))
    if covered != set(range(dimension)):
        return None
    return "V̇ = " + "".join(terms).lstrip()


def _local_polynomial_certificate(fields, point, params, candidate):
    """Certify a local semidefinite derivative and its LaSalle invariant set."""
    dimension = len(fields)
    if dimension != 2:
        return None
    vector_field = [_polynomial_from_ast(field, point, params) for field in fields]
    if any(component is None for component in vector_field):
        return None
    lie = {}
    for index, component in enumerate(vector_field):
        gradient = {}
        for powers, coefficient in candidate.items():
            if powers[index] == 0:
                continue
            reduced = list(powers)
            degree = reduced[index]
            reduced[index] -= 1
            key = tuple(reduced)
            gradient[key] = gradient.get(key, 0.0) + coefficient * degree
        lie = _poly_add(lie, _poly_multiply(gradient, component))
    scale = max(1.0, max((abs(value) for value in lie.values()), default=0.0))
    lie = {power: value for power, value in lie.items() if abs(value) > 1e-10 * scale}
    if not lie or any(any(power % 2 for power in powers) for powers in lie):
        return None
    negative = [(powers, -value) for powers, value in lie.items() if value < 0]
    positive = [(powers, value) for powers, value in lie.items() if value > 0]
    if not negative or not positive:
        return None

    radius = 0.5
    bounds_by_negative = {powers: 0.0 for powers, _ in negative}
    for positive_powers, coefficient in positive:
        parents = [(powers, amount) for powers, amount in negative
                   if all(p <= q for p, q in zip(powers, positive_powers))]
        if not parents:
            return None
        powers, amount = max(parents, key=lambda item: sum(item[0]))
        degree_gap = sum(q - p for p, q in zip(powers, positive_powers))
        bounds_by_negative[powers] += coefficient / amount * radius ** degree_gap
    if any(bound >= 1.0 for bound in bounds_by_negative.values()):
        return None

    active_coordinates = {index for powers, _ in negative
                          for index, degree in enumerate(powers) if degree}
    if len(active_coordinates) != dimension - 1:
        return None
    passive = next(index for index in range(dimension) if index not in active_coordinates)
    if any(not any(powers[index] for index in active_coordinates)
           for powers, coefficient in vector_field[passive].items() if abs(coefficient) > 1e-10):
        return None
    transverse = vector_field[next(iter(active_coordinates))]
    restricted = {}
    for powers, coefficient in transverse.items():
        if any(powers[index] for index in active_coordinates):
            continue
        if powers[passive]:
            restricted[powers[passive]] = coefficient
    if len(restricted) != 1 or min(restricted) % 2 != 1:
        return None

    pure_terms = []
    for index in range(dimension):
        power = next((powers[index] for powers, coefficient in candidate.items()
                      if powers[index] and sum(powers) == powers[index]), None)
        if power is None:
            return None
        pure_terms.append((candidate[tuple(power if i == index else 0
                                           for i in range(dimension))], power))
    level = min(coefficient * radius ** power for coefficient, power in pure_terms)
    return lie, radius, level


def _format_polynomial_derivative(lie, point):
    dimension = len(point)
    at_origin = np.allclose(point, 0.0, atol=1e-10)
    names = (("x", "y", "z") if at_origin else ("δx", "δy", "δz"))[:dimension]
    terms = []
    for powers, coefficient in sorted(lie.items(), key=lambda item: (item[0], sum(item[0]))):
        factors = [name if power == 1 else
                   name + str(power).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
                   for name, power in zip(names, powers) if power]
        term = f"{abs(coefficient):.4g}" + "·".join(factors)
        terms.append((coefficient, term))
    if not terms:
        return "V̇ = 0"
    first = terms[0][1]
    text = ("− " if terms[0][0] < 0 else "") + first
    for coefficient, term in terms[1:]:
        text += (" − " if coefficient < 0 else " + ") + term
    return "V̇ = " + text


def _rational_lyapunov_certificate(fields, point, params):
    """Check V=sum(q_i^2/(1+q_i^2)) by clearing its positive denominator."""
    dimension = len(fields)
    zero = (0,) * dimension
    vector_field = [_polynomial_from_ast(field, point, params) for field in fields]
    if any(component is None for component in vector_field):
        return None
    factors = []
    for index in range(dimension):
        powers = [0] * dimension
        powers[index] = 2
        factors.append(_poly_add({zero: 1.0}, {tuple(powers): 1.0}))
    squared = [_poly_multiply(factor, factor) for factor in factors]
    numerator = {}
    for index, component in enumerate(vector_field):
        coordinate = [0] * dimension
        coordinate[index] = 1
        term = _poly_multiply({tuple(coordinate): 2.0}, component)
        for other, factor in enumerate(squared):
            if other != index:
                term = _poly_multiply(term, factor)
        numerator = _poly_add(numerator, term)
    magnitude = max(1.0, max((abs(value) for value in numerator.values()), default=0.0))
    active = {powers: value for powers, value in numerator.items()
              if abs(value) > 1e-10 * magnitude}
    covered = set()
    terms = []
    at_origin = np.allclose(point, 0.0, atol=1e-10)
    names = (("x", "y", "z") if at_origin else ("δx", "δy", "δz"))[:dimension]
    for powers, coefficient in sorted(
            active.items(),
            key=lambda item: (next((i for i, power in enumerate(item[0]) if power), dimension),
                              sum(item[0]), item[0])):
        if coefficient >= 0 or any(power % 2 for power in powers) or not any(powers):
            return None
        for index, power in enumerate(powers):
            if power and sum(powers) == power:
                covered.add(index)
        monomial = "·".join(name if power == 1 else
                            name + str(power).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
                            for name, power in zip(names, powers) if power)
        terms.append(f" − {abs(coefficient):.4g}{monomial}")
    if covered != set(range(dimension)):
        return None
    denominator = "·".join(f"(1+{name}²)²" for name in names)
    derivative = "V̇ = (" + "".join(terms).lstrip() + f")/{denominator}"
    formula = "V = " + " + ".join(f"{name}²/(1+{name}²)" for name in names)
    return formula, derivative


def _candidate_is_positive_definite(candidate, dimension):
    quadratic = np.zeros((dimension, dimension), dtype=float)
    has_quadratic_cross_term = False
    for powers, coefficient in candidate.items():
        degree = sum(powers)
        if degree == 2:
            indices = [index for index, power in enumerate(powers) if power]
            if len(indices) == 1 and powers[indices[0]] == 2:
                if coefficient <= 0:
                    return False
                quadratic[indices[0], indices[0]] += coefficient
            elif len(indices) == 2 and all(powers[index] == 1 for index in indices):
                has_quadratic_cross_term = True
                first, second = indices
                quadratic[first, second] += coefficient / 2
                quadratic[second, first] += coefficient / 2
            else:
                return False
        elif coefficient <= 0 or any(power % 2 for power in powers):
            return False
    # Sylvester's criterion for the quadratic part; the remaining terms are
    # nonnegative even monomials and cannot destroy positive definiteness.
    if has_quadratic_cross_term:
        for size in range(1, dimension + 1):
            block = quadratic[:size, :size]
            if size == 1:
                determinant = block[0, 0]
            elif size == 2:
                determinant = block[0, 0] * block[1, 1] - block[0, 1] ** 2
            else:
                determinant = (
                    block[0, 0] * (block[1, 1] * block[2, 2] - block[1, 2] ** 2)
                    - block[0, 1] * (block[0, 1] * block[2, 2] - block[1, 2] * block[0, 2])
                    + block[0, 2] * (block[0, 1] * block[1, 2] - block[1, 1] * block[0, 2])
                )
            if determinant <= 1e-12:
                return False
    return all(any(powers[index] > 0 and sum(powers) == powers[index]
                   and coefficient > 0 for powers, coefficient in candidate.items())
               for index in range(dimension))


def _format_polynomial(candidate, point):
    dimension = len(next(iter(candidate)))
    at_origin = np.allclose(point, 0.0, atol=1e-10)
    names = (("x", "y", "z") if at_origin else ("δx", "δy", "δz"))[:dimension]
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    terms = []
    for powers, coefficient in sorted(
            candidate.items(),
            key=lambda item: (next((i for i, power in enumerate(item[0]) if power), dimension),
                              sum(item[0]), item[0])):
        if abs(coefficient) < 1e-10:
            continue
        factors = [name if power == 1 else name + str(power).translate(superscripts)
                   for name, power in zip(names, powers) if power]
        if not factors:
            continue
        magnitude = abs(coefficient)
        product = "·".join(factors)
        if len(factors) == 1 and magnitude == 1:
            term = product
        else:
            term = f"{magnitude:.4g}" + product
        terms.append((coefficient, term))
    if not terms:
        return "V = 0"
    _, first = terms[0]
    result = ("− " if terms[0][0] < 0 else "") + first
    for coefficient, term in terms[1:]:
        result += (" − " if coefficient < 0 else " + ") + term
    return "V = " + result


def analyze_lyapunov_function(fields, point, params, bounds, details):
    """Enumerate polynomial Lyapunov candidates and retain symbolic certificates only."""
    point = np.asarray(point, dtype=float)
    dimension = len(fields)
    zero = (0,) * dimension
    candidates = []

    # Positive diagonal quadratic forms with common useful rational weights.
    for weights in itertools.product((0.5, 1.0, 2.0, 4.0), repeat=dimension):
        candidates.append({tuple(2 if i == j else 0 for i in range(dimension)): weight
                           for j, weight in enumerate(weights)})
        for first in range(dimension):
            for second in range(first + 1, dimension):
                for cross_weight in (-1.0, -0.5, 0.5, 1.0):
                    mixed = {tuple(2 if i == j else 0 for i in range(dimension)): weight
                             for j, weight in enumerate(weights)}
                    powers = [0] * dimension
                    powers[first] = powers[second] = 1
                    mixed[tuple(powers)] = cross_weight
                    candidates.append(mixed)

    # Higher even powers and mixed-degree sums, e.g. x²+y⁴ or x⁴+y⁶.
    for powers in itertools.product((2, 4, 6), repeat=dimension):
        for weights in itertools.product((1.0, 2.0), repeat=dimension):
            candidates.append({tuple(power if i == j else 0 for i in range(dimension)): weight
                               for j, (power, weight) in enumerate(zip(powers, weights))})

    # Positive cross-polynomial combinations retain an immediate positivity
    # argument while allowing quartic and higher mixed terms to be tested.
    for base_weights in itertools.product((1.0, 2.0), repeat=dimension):
        base = {tuple(2 if i == j else 0 for i in range(dimension)): weight
                for j, weight in enumerate(base_weights)}
        for first in range(dimension):
            for second in range(first + 1, dimension):
                powers = [0] * dimension
                powers[first] = powers[second] = 2
                for cross_weight in (0.25, 0.5, 1.0):
                    mixed = dict(base)
                    mixed[tuple(powers)] = cross_weight
                    candidates.append(mixed)

    certificates = []
    local_certificates = []
    seen = set()
    for candidate in candidates:
        # Candidate V must be a positive sum of even monomials and contain a
        # positive pure term for every state coordinate.
        if not _candidate_is_positive_definite(candidate, dimension):
            continue
        scale = min(candidate.values())
        normalized = {powers: value / scale for powers, value in candidate.items()}
        formula = _format_polynomial(normalized, point)
        if formula in seen:
            continue
        derivative = _polynomial_lie_derivative(fields, point, params, normalized)
        if derivative is not None:
            seen.add(formula)
            certificates.append((sum(sum(power) for power in normalized), formula, derivative))
        else:
            local = _local_polynomial_certificate(fields, point, params, normalized)
            if local is not None:
                local_certificates.append((formula, normalized, local))
    certificates.sort(key=lambda item: item[0])
    rational_certificate = _rational_lyapunov_certificate(fields, point, params)

    lines = []
    if certificates:
        for _, formula, derivative in certificates[:3]:
            lines.extend((f"Lyapunov function: {formula}", derivative))
        if details["hyperbolic"] and details["linearization"].startswith("Unstable"):
            lines.append("Lyapunov stability: unstable (hyperbolic linearization).")
        else:
            lines.append("Lyapunov stability: globally asymptotically stable (polynomial certificate).")
    elif rational_certificate is not None:
        formula, derivative = rational_certificate
        lines.extend((f"Lyapunov function: {formula}", derivative,
                      "Lyapunov stability: locally asymptotically stable (rational certificate)."))
    elif local_certificates:
        formula, candidate, (lie, radius, level) = local_certificates[0]
        derivative = _format_polynomial_derivative(lie, point)
        negative = [(powers, -coefficient) for powers, coefficient in lie.items() if coefficient < 0]
        positive = [(powers, coefficient) for powers, coefficient in lie.items() if coefficient > 0]
        if len(negative) == len(positive) == 1:
            base_powers, base_coefficient = negative[0]
            high_powers, high_coefficient = positive[0]
            differences = [high - base for base, high in zip(base_powers, high_powers)]
            if sum(differences) == 2 and all(value in (0, 2) for value in differences):
                index = differences.index(2)
                names = (("x", "y", "z") if np.allclose(point, 0.0, atol=1e-10)
                         else ("δx", "δy", "δz"))
                factors = "".join(name if power == 1 else name + str(power).translate(
                    str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
                                  for name, power in zip(names, base_powers) if power)
                ratio = high_coefficient / base_coefficient
                bracket = "1 − " + ("" if abs(ratio - 1) < 1e-10 else f"{ratio:.4g}") + names[index] + "²"
                derivative = f"V̇ = − {base_coefficient:.4g}{factors}({bracket})"
        lines.extend((f"Lyapunov function: {formula}", derivative,
                      f"For V < {level:.4g}, V̇ ≤ 0; LaSalle's invariance principle gives local asymptotic stability."))
    else:
        lines.extend(("Lyapunov function: no certifying candidate found.",
                      "Lyapunov stability: inconclusive."))
    return lines


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
            # GLSL `pow(x, y)` is undefined for negative x, even when y is an
            # integer. Expand integer powers so fields such as x^4 work in
            # every quadrant instead of freezing particles on negative axes.
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                exponent = float(node.right.value)
                if exponent.is_integer() and abs(exponent) <= 32:
                    integer_exponent = int(exponent)
                    if integer_exponent == 0:
                        return "1.0"
                    power = "(" + " * ".join([left] * abs(integer_exponent)) + ")"
                    return f"(1.0 / {power})" if integer_exponent < 0 else power
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
        name = {"ln": "log", "minimum": "min", "maximum": "max", "atan2": "atan"}.get(name, name)
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

    def textFromValue(self, value):
        text = super().textFromValue(value)
        if self.property("compactDecimals"):
            separator = self.locale().decimalPoint()
            if separator in text:
                text = text.rstrip("0")
                if text.endswith(separator):
                    text += "0"
        return text


class _FractionInput(QWidget):
    """Compact stacked numerator/denominator input used inside a formula."""
    def __init__(self, numerator="", denominator="", parent=None):
        super().__init__(parent)
        self.setObjectName("FractionInput")
        self.setFixedWidth(62)
        self.setMinimumHeight(44)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(0)
        self.numerator = QLineEdit(numerator, self)
        self.denominator = QLineEdit(denominator, self)
        for field in (self.numerator, self.denominator):
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            field.setMinimumWidth(34)
            field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            field.setStyleSheet("QLineEdit { border: none; background: transparent; padding: 0 2px; }")
        layout.addWidget(self.numerator)
        self.bar = QFrame(self)
        self.bar.setFrameShape(QFrame.Shape.HLine)
        self.bar.setFrameShadow(QFrame.Shadow.Plain)
        self.bar.setFixedHeight(1)
        self.bar.setStyleSheet("QFrame { border: none; background: #777777; }")
        layout.addWidget(self.bar)
        layout.addWidget(self.denominator)

    def expression(self):
        return f"({self.numerator.text()})/({self.denominator.text()})"


class FormulaEditor(QWidget):
    """Inline formula editor with slash-to-fraction insertion and editable slots."""
    textChanged = Signal(str)
    focusEntered = Signal()

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setObjectName("FormulaEditor")
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(5, 2, 5, 2)
        self._layout.setSpacing(2)
        self._parts = []
        self._active_input = None
        self._placeholder = ""
        self._mutating = False
        self._undo, self._redo = [], []
        self._last_expression = ""
        self._theme = ("#ffffff", "#b8b8b8", "#111111")
        self._load_expression(text)
        self._last_expression = self.text()
        self._apply_theme()

    def _new_text_field(self, text=""):
        field = QLineEdit(text, self)
        field.setMinimumWidth(30)
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        field.setStyleSheet(
            f"QLineEdit {{ border: none; background: transparent; color: {self._theme[2]}; padding: 1px 2px; }}"
        )
        field.installEventFilter(self)
        field.textEdited.connect(lambda value, target=field: self._on_field_edited(target, value))
        self._active_input = self._active_input or field
        return field

    @staticmethod
    def _operand_start(text, end):
        end = min(end, len(text))
        while end and text[end - 1].isspace():
            end -= 1
        if end == 0:
            return 0
        if text[end - 1] == ")":
            depth = 0
            for index in range(end - 1, -1, -1):
                if text[index] == ")":
                    depth += 1
                elif text[index] == "(":
                    depth -= 1
                    if depth == 0:
                        start = index
                        while start and (text[start - 1].isalnum() or text[start - 1] == "_"):
                            start -= 1
                        return start
        index = end - 1
        while index >= 0 and (text[index].isalnum() or text[index] in "_.^⁰¹²³⁴⁵⁶⁷⁸⁹"):
            index -= 1
        if index >= 0 and text[index] in "+-−" and (index == 0 or text[index - 1] in "+-−*/^"):
            index -= 1
        return index + 1

    @staticmethod
    def _operand_end(text, start):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            return start
        if text[start] == "(":
            depth = 0
            for index in range(start, len(text)):
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        return index + 1
        index = start
        if text[index] in "+-−":
            index += 1
        while index < len(text) and (text[index].isalnum() or text[index] in "_.^⁰¹²³⁴⁵⁶⁷⁸⁹"):
            index += 1
        return max(start, index)

    @classmethod
    def _fraction_parts(cls, expression):
        depth = 0
        slash = None
        for index, char in enumerate(expression):
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "/" and depth == 0:
                slash = index
                break
        if slash is None:
            return None
        start = cls._operand_start(expression, slash)
        end = cls._operand_end(expression, slash + 1)
        return expression[:start], expression[start:slash].strip(), expression[slash + 1:end].strip(), expression[end:]

    def _clear_layout(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self._parts.clear()
        self._active_input = None

    def _append_segment(self, text=""):
        field = self._new_text_field(format_formula_input(text))
        if not self._parts and self._placeholder:
            field.setPlaceholderText(self._placeholder)
        self._parts.append(field)
        self._layout.addWidget(field, 1)
        return field

    def _append_fraction(self, numerator="", denominator=""):
        fraction = _FractionInput(format_formula_input(numerator), format_formula_input(denominator), self)
        fraction.numerator.installEventFilter(self)
        fraction.denominator.installEventFilter(self)
        fraction.numerator.textEdited.connect(
            lambda value, target=fraction.numerator: self._on_field_edited(target, value)
        )
        fraction.denominator.textEdited.connect(
            lambda value, target=fraction.denominator: self._on_field_edited(target, value)
        )
        self._style_fraction(fraction)
        self._active_input = self._active_input or fraction.denominator
        self._parts.append(fraction)
        self._layout.addWidget(fraction, 0, Qt.AlignmentFlag.AlignVCenter)
        return fraction

    def _style_fraction(self, fraction):
        foreground = self._theme[2]
        for field in (fraction.numerator, fraction.denominator):
            field.setStyleSheet(
                f"QLineEdit {{ border: none; background: transparent; color: {foreground}; padding: 0 2px; }}"
            )
        fraction.bar.setStyleSheet(f"QFrame {{ border: none; background: {foreground}; }}")

    def _apply_theme(self):
        background, border, foreground = self._theme
        self.setStyleSheet(
            f"QWidget#FormulaEditor {{ border: 1px solid {border}; border-radius: 4px; "
            f"background-color: {background}; }}"
        )
        for part in self._parts:
            if isinstance(part, _FractionInput):
                self._style_fraction(part)
            else:
                part.setStyleSheet(
                    f"QLineEdit {{ border: none; background: transparent; color: {foreground}; padding: 1px 2px; }}"
                )

    def set_theme(self, background, border, foreground):
        self._theme = (background, border, foreground)
        self._apply_theme()

    def _load_expression(self, expression):
        self._mutating = True
        self._clear_layout()
        expression = str(expression).replace("÷", "/")
        split = self._fraction_parts(expression)
        if split is None:
            self._append_segment(expression)
        else:
            prefix, numerator, denominator, suffix = split
            self._append_segment(prefix)
            fraction = self._append_fraction(numerator, denominator)
            self._append_segment(suffix)
            self._active_input = fraction.denominator
        self._mutating = False

    def _record_change(self):
        current = self.text()
        if self._mutating or current == self._last_expression:
            return
        self._undo.append(self._last_expression)
        del self._undo[:-100]
        self._redo.clear()
        self._last_expression = current
        self.textChanged.emit(current)

    def _on_field_edited(self, field, value):
        if self._mutating:
            return
        formatted = format_formula_input(value)
        if formatted != value:
            cursor = field.cursorPosition()
            formatted_cursor = len(format_formula_input(value[:cursor]))
            blocker = QSignalBlocker(field)
            field.setText(formatted)
            field.setCursorPosition(min(formatted_cursor, len(formatted)))
            del blocker
        self._record_change()

    def _insert_fraction(self, field):
        if field.parent() is not self:
            field.insert("/")
            return
        index = self._parts.index(field)
        value = field.text()
        selection_start = field.selectionStart()
        if selection_start >= 0:
            start = selection_start
            end = start + len(field.selectedText())
            numerator = value[start:end]
        else:
            end = start = field.cursorPosition()
            start = self._operand_start(value, end)
            numerator = value[start:end].strip()
        prefix, suffix = value[:start], value[end:]
        fraction = _FractionInput(parent=self)
        fraction.numerator.setText(format_formula_input(numerator))
        fraction.numerator.installEventFilter(self)
        fraction.denominator.installEventFilter(self)
        fraction.numerator.textEdited.connect(
            lambda text, target=fraction.numerator: self._on_field_edited(target, text)
        )
        fraction.denominator.textEdited.connect(
            lambda text, target=fraction.denominator: self._on_field_edited(target, text)
        )
        self._style_fraction(fraction)
        self._mutating = True
        before = self._new_text_field(format_formula_input(prefix))
        after = self._new_text_field(format_formula_input(suffix))
        self._parts[index:index + 1] = [before, fraction, after]
        self._layout.insertWidget(index, before, 1)
        self._layout.insertWidget(index + 1, fraction, 0, Qt.AlignmentFlag.AlignVCenter)
        self._layout.insertWidget(index + 2, after, 1)
        field.hide()
        field.deleteLater()
        self._mutating = False
        self._active_input = fraction.denominator
        self._record_change()
        fraction.denominator.setFocus()

    def _move_after_fraction(self, denominator):
        for index, part in enumerate(self._parts):
            if isinstance(part, _FractionInput) and part.denominator is denominator:
                if index + 1 < len(self._parts) and isinstance(self._parts[index + 1], QLineEdit):
                    target = self._parts[index + 1]
                else:
                    target = self._append_segment()
                target.setFocus()
                target.setCursorPosition(0)
                return True
        return False

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.FocusIn:
            self._active_input = watched
            self.focusEntered.emit()
        if event.type() == QEvent.Type.KeyPress:
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier and event.key() in (Qt.Key.Key_Z, Qt.Key.Key_Y):
                redo = event.key() == Qt.Key.Key_Y or bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                source, destination = (self._redo, self._undo) if redo else (self._undo, self._redo)
                if source:
                    destination.append(self.text())
                    target = source.pop()
                    self._load_expression(target)
                    self._last_expression = self.text()
                    self.textChanged.emit(self.text())
                    self.setFocus()
                    return True
            if event.key() == Qt.Key.Key_Slash and isinstance(watched, QLineEdit):
                self._insert_fraction(watched)
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
                if isinstance(watched, QLineEdit) and self._move_after_fraction(watched):
                    return True
            if event.key() == Qt.Key.Key_Right and isinstance(watched, QLineEdit):
                if watched.cursorPosition() == len(watched.text()) and self._move_after_fraction(watched):
                    return True
        return super().eventFilter(watched, event)

    def text(self):
        output = []
        for part in self._parts:
            output.append(part.expression() if isinstance(part, _FractionInput) else part.text())
        return "".join(output)

    def setText(self, expression):
        expression = str(expression)
        old = self.text()
        self._load_expression(expression)
        self._undo.clear()
        self._redo.clear()
        self._last_expression = self.text()
        if old != self._last_expression:
            self.textChanged.emit(self._last_expression)

    def setPlaceholderText(self, text):
        self._placeholder = str(text)
        if self._parts and isinstance(self._parts[0], QLineEdit):
            self._parts[0].setPlaceholderText(self._placeholder)

    def setFocus(self, *args):
        target = self._active_input
        if target is not None:
            target.setFocus(*args)
        else:
            super().setFocus(*args)

    def contains_widget(self, widget):
        while widget is not None:
            if widget is self:
                return True
            widget = widget.parentWidget()
        return False

    def insert_text(self, value):
        target = self._active_input
        if target is None:
            target = next((part for part in reversed(self._parts) if isinstance(part, QLineEdit)), None)
        if target is None:
            target = self._append_segment()
        if value in ("/", "÷") and target.parent() is self:
            self._insert_fraction(target)
            return
        target.insert(value)
        self._active_input = target
        self._on_field_edited(target, target.text())


class FlowCanvas(QOpenGLWidget):
    # Trail frames remain nominal 60 Hz frames; each contains real ODE samples.
    SAMPLES_PER_FRAME = 16
    NOMINAL_FPS = 60.0
    gpuStatus = Signal(str)
    canvasResized = Signal()
    viewChanged = Signal(float, float, float, float)
    initialPointPicked = Signal(float, float)
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
    uniform sampler2D u_x;uniform sampler2D u_y;uniform float u_count,u_dt;
    uniform int u_normalize;uniform vec4 u_metric;
    layout(location=0) out vec4 out_x;layout(location=1) out vec4 out_y;
    float decodeF(vec4 c){uvec4 b=uvec4(round(c*255.0));uint n=(b.r<<24u)|(b.g<<16u)|(b.b<<8u)|b.a;return uintBitsToFloat(n);}
    vec4 encodeF(float v){uint n=floatBitsToUint(v);return vec4(float((n>>24u)&255u),float((n>>16u)&255u),float((n>>8u)&255u),float(n&255u))/255.0;}
    vec2 velocity(vec2 p){float x=p.x;float y=p.y;return vec2(__VX__,__VY__);}
    vec2 field(vec2 p){vec2 v=velocity(p);if(u_normalize==1){float m=length(v);if(m>1e-12)v/=m;}return v;}
    bool finite_point(vec2 p){return !any(isnan(p))&&!any(isinf(p));}
    vec2 rk4(vec2 p,float h){
      vec2 k1=field(p),k2=field(p+0.5*h*k1),k3=field(p+0.5*h*k2),k4=field(p+h*k3);
      return p+(h/6.0)*(k1+2.0*k2+2.0*k3+k4);}
    vec2 integrate(vec2 p){
      if(!finite_point(p)||u_dt<=0.0)return p;
      float elapsed=0.0,h=u_dt;
      // RK4 step doubling estimates local error. Reject inaccurate finite
      // jumps as well as NaN/Infinity; never draw an unvalidated endpoint.
      for(int attempt=0;attempt<256;attempt++){
        h=min(h,u_dt-elapsed);
        vec2 whole=rk4(p,h),mid=rk4(p,0.5*h),fine=rk4(mid,0.5*h);
        bool valid=finite_point(whole)&&finite_point(mid)&&finite_point(fine);
        float error=1e30;
        if(valid){
          vec2 tolerance=max(u_metric.xy*0.02,vec2(1e-7))+2e-6*max(abs(p),abs(fine));
          vec2 ratio=abs(fine-whole)/(15.0*tolerance);
          error=max(ratio.x,ratio.y);
        }
        if(valid&&error<=1.0){
          p=fine;elapsed+=h;
          if(elapsed>=u_dt)return p;
          h*=clamp(0.9*pow(max(error,1e-8),-0.2),0.5,2.0);
        }else{
          h*=valid?clamp(0.9*pow(max(error,1e-8),-0.2),0.1,0.5):0.25;
        }
        if(h<max(1e-12,u_dt*1e-8))break;
      }
      // Undefined fields, finite-time escape or an exhausted error budget
      // end this numerical path. Existing tails fade; normal respawn resumes it.
      return vec2(uintBitsToFloat(0x7fc00000u));
    }
    void main(){vec2 uv=vec2(gl_FragCoord.x/u_count,0.5);
      vec2 p=vec2(decodeF(texture(u_x,uv)),decodeF(texture(u_y,uv)));
      vec2 q=integrate(p);
      out_x=encodeF(q.x);out_y=encodeF(q.y);}
    """
    TRAIL_VS = """#version 330 core
    uniform sampler2D u_history_x,u_history_y,u_history_birth;
    uniform float u_count,u_history_length,u_history_latest,u_frame,u_samples_per_frame;
    out vec2 trail_a,trail_b;out float trail_age_a,trail_age_b,trail_birth_fade;
    flat out int trail_valid;
    float decodeF(vec4 c){uvec4 b=uvec4(round(c*255.0));uint n=(b.r<<24u)|(b.g<<16u)|(b.b<<8u)|b.a;return uintBitsToFloat(n);}
    vec2 sample_position(int particle,int age){
      int row=(int(u_history_latest)-age+int(u_history_length)*int(u_history_length))%int(u_history_length);
      vec2 uv=vec2((float(particle)+0.5)/u_count,(float(row)+0.5)/u_history_length);
      return vec2(decodeF(texture(u_history_x,uv)),decodeF(texture(u_history_y,uv)));}
    float sample_birth(int particle,int age){
      int row=(int(u_history_latest)-age+int(u_history_length)*int(u_history_length))%int(u_history_length);
      vec2 uv=vec2((float(particle)+0.5)/u_count,(float(row)+0.5)/u_history_length);
      return decodeF(texture(u_history_birth,uv));}
    void main(){int segment=gl_VertexID%int(u_history_length-1.0);
      int particle=gl_VertexID/int(u_history_length-1.0);
      gl_Position=vec4(0.0,0.0,0.0,1.0);trail_valid=0;
      trail_a=vec2(0.0);trail_b=vec2(0.0);trail_age_a=1.0;trail_age_b=1.0;trail_birth_fade=0.0;
      float newer_birth=sample_birth(particle,segment);
      float older_birth=sample_birth(particle,segment+1);
      if(segment>=int(min(u_frame,u_history_length-1.0)) || abs(newer_birth-older_birth)>0.25){
        return;}
      vec2 older=sample_position(particle,segment+1),newer=sample_position(particle,segment);
      if(any(isnan(older))||any(isinf(older))||any(isnan(newer))||any(isinf(newer))){
        return;}
      trail_a=older;trail_b=newer;trail_valid=1;
      trail_age_a=float(segment+1)/max(1.0,u_history_length-1.0);
      trail_age_b=float(segment)/max(1.0,u_history_length-1.0);
      trail_birth_fade=smoothstep(0.0,12.0*u_samples_per_frame,u_frame-newer_birth);}
    """
    TRAIL_GS = """#version 330 core
    layout(points) in;layout(triangle_strip,max_vertices=32) out;
    in vec2 trail_a[],trail_b[];in float trail_age_a[],trail_age_b[],trail_birth_fade[];
    flat in int trail_valid[];
    uniform vec4 u_bounds,u_plot,u_viewport;uniform float u_dt,u_line_width;
    uniform int u_normalize;
    out float v_speed,v_edge,v_age,v_birth_fade;
    vec2 velocity(vec2 p){float x=p.x;float y=p.y;return vec2(__VX__,__VY__);}
    bool finite_point(vec2 p){return !any(isnan(p))&&!any(isinf(p));}
    vec2 project(vec2 p){
      vec2 unit=(p-u_bounds.xy)/(u_bounds.zw-u_bounds.xy);
      return mix(u_plot.xz*u_viewport.xy,u_plot.yw*u_viewport.xy,unit);}
    vec2 project_tangent(vec2 v){
      return v/(u_bounds.zw-u_bounds.xy)*(u_plot.yw-u_plot.xz)*u_viewport.xy;}
    vec2 motion(vec2 v){
      if(u_normalize==1){float m=length(v);if(m>1e-12)return v/m;}return v;}
    vec2 position_at(float t,vec2 a,vec2 b,vec2 ta,vec2 tb){
      float t2=t*t,t3=t2*t;
      return (2.0*t3-3.0*t2+1.0)*a+(t3-2.0*t2+t)*ta
             +(-2.0*t3+3.0*t2)*b+(t3-t2)*tb;}
    vec2 tangent_at(float t,vec2 a,vec2 b,vec2 ta,vec2 tb){
      float t2=t*t;
      return (6.0*t2-6.0*t)*a+(3.0*t2-4.0*t+1.0)*ta
             +(-6.0*t2+6.0*t)*b+(3.0*t2-2.0*t)*tb;}
    void emit_side(vec2 p,vec2 normal,float side,float t,float speed_a,float speed_b){
      vec2 pixel=p+side*normal*(0.5*u_line_width);
      gl_Position=vec4(2.0*pixel/u_viewport.xy-1.0,0.0,1.0);
      v_edge=side;v_age=mix(trail_age_a[0],trail_age_b[0],t);
      v_speed=mix(speed_a,speed_b,t);v_birth_fade=trail_birth_fade[0];EmitVertex();}
    void main(){
      if(trail_valid[0]==0)return;
      vec2 va=velocity(trail_a[0]),vb=velocity(trail_b[0]);
      if(!finite_point(va)||!finite_point(vb))return;
      vec2 a=project(trail_a[0]),b=project(trail_b[0]);
      vec2 ta=project_tangent(motion(va)*u_dt),tb=project_tangent(motion(vb)*u_dt);
      if(!finite_point(a)||!finite_point(b)||!finite_point(ta)||!finite_point(tb))return;
      vec2 chord=b-a;float chord_length=length(chord);
      if(chord_length<1e-5&&max(length(ta),length(tb))<1e-5)return;
      // Hermite dense output uses ODE tangents, not arbitrary visual smoothing.
      // Collinear motion is already geometrically exact; keep it monotone.
      float bend_a=abs(chord.x*ta.y-chord.y*ta.x)/max(chord_length,1e-5);
      float bend_b=abs(chord.x*tb.y-chord.y*tb.x)/max(chord_length,1e-5);
      if(chord_length>1e-5&&max(bend_a,bend_b)<0.05&&dot(ta,chord)>=0.0&&dot(tb,chord)>=0.0){ta=chord;tb=chord;}
      vec2 c3=2.0*a+ta-2.0*b+tb,c2=-3.0*a-2.0*ta+3.0*b-tb;
      float second_derivative=max(length(2.0*c2),length(6.0*c3+2.0*c2));
      int pieces=int(clamp(ceil(sqrt(second_derivative/(8.0*0.20))),1.0,8.0));
      float sa=length(va),sb=length(vb);
      if(isnan(sa)||isinf(sa))sa=0.0;if(isnan(sb)||isinf(sb))sb=0.0;
      for(int i=0;i<8;i++){
        if(i>=pieces)break;
        float t0=float(i)/float(pieces),t1=float(i+1)/float(pieces);
        vec2 p0=position_at(t0,a,b,ta,tb),p1=position_at(t1,a,b,ta,tb);
        vec2 d0=tangent_at(t0,a,b,ta,tb),d1=tangent_at(t1,a,b,ta,tb);
        if(length(d0)<1e-5)d0=chord;if(length(d1)<1e-5)d1=chord;
        vec2 n0=vec2(-d0.y,d0.x)/max(length(d0),1e-5);
        vec2 n1=vec2(-d1.y,d1.x)/max(length(d1),1e-5);
        // Neighbouring pieces share identical end positions and normals.
        emit_side(p0,n0,1.0,t0,sa,sb);emit_side(p0,n0,-1.0,t0,sa,sb);
        emit_side(p1,n1,1.0,t1,sa,sb);emit_side(p1,n1,-1.0,t1,sa,sb);EndPrimitive();
      }
    }
    """
    TRAIL_FS = """#version 330 core
    in float v_speed;in float v_edge;in float v_age;in float v_birth_fade;uniform int u_speed_color,u_dark_mode;
    uniform float u_speed_max,u_density_scale;out vec4 out_color;
    void main(){float a=0.94*u_density_scale*v_birth_fade*(1.0-smoothstep(0.65,1.0,abs(v_edge)))*(1.0-v_age);
      float t=clamp(log(1.0+max(v_speed,0.0))/max(log(1.0+u_speed_max),1e-6),0.0,1.0);
      vec3 spectrum=clamp(1.5-abs(4.0*t-vec3(3.0,2.0,1.0)),0.0,1.0);
      vec3 base=u_speed_color==1?spectrum:(u_dark_mode==1?vec3(0.38,0.76,1.0):vec3(0.05,0.18,0.48));
      out_color=vec4(base*a,a);}
    """
    STORE_FS = """#version 330 core
    uniform sampler2D u_x,u_y,u_birth_frame;uniform float u_count;
    layout(location=0) out vec4 out_x;layout(location=1) out vec4 out_y;layout(location=2) out vec4 out_birth;
    void main(){vec2 uv=vec2(gl_FragCoord.x/u_count,0.5);
      out_x=texture(u_x,uv);out_y=texture(u_y,uv);out_birth=texture(u_birth_frame,uv);}
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
        self.count, self.trail_length, self.dt, self.normalize = 1000, 100, 0.01, True
        self.speed, self.simulation_substeps = 1.0, self.SAMPLES_PER_FRAME
        self.dt /= self.SAMPLES_PER_FRAME
        self._sampling_budget = 0.0
        self.respawn_period = 5.0
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
        if self.gl is not None and self.vao is not None and self.vao.isCreated():
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
        self.simulation_substeps = self.SAMPLES_PER_FRAME
        self.dt = float(dt) / self.SAMPLES_PER_FRAME

    @property
    def history_length(self):
        return self.trail_length * self.SAMPLES_PER_FRAME

    def update_runtime(self, count, trail, dt, speed, normalize, respawn_period):
        count = int(count)
        trail = int(trail)
        trail_changed = trail != self.trail_length
        old_dt, old_normalize = self.dt, self.normalize
        self.trail_length, self.normalize = trail, bool(normalize)
        self._set_simulation_rate(dt, speed)
        self.respawn_period = float(respawn_period)
        motion_changed = self.dt != old_dt or self.normalize != old_normalize
        if count == self.count:
            if (trail_changed or motion_changed) and self.ready:
                self.makeCurrent()
                try:
                    if trail_changed:
                        self._create_history_fbos()
                    # Dense output must use the same dt and vector field as
                    # the samples it joins. Speed alone does not change either.
                    self._clear_trails()
                except Exception as exc:
                    self._fail(exc)
                finally:
                    self.doneCurrent()
                self.update()
            return
        self.count = count
        if self.ready:
            self.makeCurrent()
            try:
                self._create_position_fbos()
                self._create_history_fbos()
                self._clear_trails()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()

    def set_view_bounds(self, bounds):
        self.bounds = tuple(float(v) for v in bounds)
        if self.ready:
            self.makeCurrent()
            try:
                self._clear_trails()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()
        self.update()

    def set_visual_options(self, speed_color, dark_mode, speed_max=None):
        speed_color = bool(speed_color)
        self.speed_color = speed_color
        self.dark_mode = bool(dark_mode)
        if speed_max is not None:
            self.speed_max = max(float(speed_max), 1e-9)
        if self.ready:
            self.makeCurrent()
            try:
                self._clear_trails()
            except Exception as exc:
                self._fail(exc)
            self.doneCurrent()
        self.update()

    def set_running(self, running):
        if running and not self.running:
            self._last_frame_time = time.monotonic()
        self.running = bool(running)

    def clear_trails(self):
        if self.ready:
            self.makeCurrent()
            try:
                self._clear_trails()
                self.gl.glFlush()
            finally:
                self.doneCurrent()
            self.update()

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
            if math.hypot(dx, dy) < QApplication.startDragDistance():
                return
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
            distance = (event.position() - self._pan_origin).manhattanLength()
            point = self._screen_to_view(event.position())
            if distance < QApplication.startDragDistance() and point is not None:
                self.initialPointPicked.emit(*point)
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

    def _make_program(self, vs, fs, geometry=None):
        p = QOpenGLShaderProgram(self)
        if not p.addShaderFromSourceCode(QOpenGLShader.Vertex, vs):
            raise RuntimeError(p.log())
        if geometry is not None and not p.addShaderFromSourceCode(QOpenGLShader.Geometry, geometry):
            raise RuntimeError(p.log())
        if not p.addShaderFromSourceCode(QOpenGLShader.Fragment, fs):
            raise RuntimeError(p.log())
        if not p.link():
            raise RuntimeError(p.log())
        return p

    def _build_resources(self):
        for name in ("sim_program", "trail_program", "store_program", "background_program"):
            program = getattr(self, name, None)
            if program is not None:
                program.release()
                program.removeAllShaders()
                program.deleteLater()
                setattr(self, name, None)
        sim = self.SIM_FS.replace("__VX__", self.formulas[0]).replace("__VY__", self.formulas[1])
        trail_gs = self.TRAIL_GS.replace("__VX__", self.formulas[0]).replace("__VY__", self.formulas[1])
        self.sim_program = self._make_program(self.QUAD_VS, sim)
        self.trail_program = self._make_program(self.TRAIL_VS, self.TRAIL_FS, geometry=trail_gs)
        self.store_program = self._make_program(self.QUAD_VS, self.STORE_FS)
        self.background_program = self._make_program(self.QUAD_VS, self.BACKGROUND_FS)
        self._create_position_fbos()
        self._create_history_fbos()
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
        for pair in self.position_fbos:
            self._attach_render_targets(pair)

    def _attach_render_targets(self, targets):
        """Write matching coordinate/birth textures in one draw call."""
        targets[0].bind()
        for index, target in enumerate(targets[1:], 1):
            self.gl.glFramebufferTexture2D(self.FRAMEBUFFER, 0x8CE0+index,
                                           self.TEX2D, target.texture(), 0)
        self.gl.glDrawBuffers(len(targets), [0x8CE0+index for index in range(len(targets))])
        status = self.gl.glCheckFramebufferStatus(self.FRAMEBUFFER)
        targets[0].release()
        if status != 0x8CD5:
            raise RuntimeError(f"Particle framebuffer is incomplete: {status:#x}")

    def _create_history_fbos(self):
        self.history_fbos = [
            self._new_fbo(self.count, self.history_length),
            self._new_fbo(self.count, self.history_length),
        ]
        self.birth_fbo = self._new_fbo(self.count, 1)
        self.history_birth_fbo = self._new_fbo(self.count, self.history_length)
        self._attach_render_targets([*self.history_fbos, self.history_birth_fbo])
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
        self._sampling_budget = 0.0
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
            rows = np.broadcast_to(raw, (self.history_length, raw.size)).copy()
            self._bind_texture(0, fbo.texture())
            _upload_rgba8(rows, len(values), start, height=self.history_length)
        birth_raw = np.zeros((self.history_length, len(xs), 4), dtype=np.uint8)
        self._bind_texture(0, self.history_birth_fbo.texture())
        _upload_rgba8(birth_raw, len(xs), start, height=self.history_length)

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
            # Constant/zero fields may legitimately optimize uniforms away.
            return
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
        xmin, xmax, ymin, ymax = self.bounds
        left, right, bottom, top = self.plot_rect
        ratio = self.devicePixelRatioF()
        self._set_uniform(self.sim_program, "u_metric", QVector4D(
            (xmax-xmin)/max(1.0, self.width()*ratio*(right-left)),
            (ymax-ymin)/max(1.0, self.height()*ratio*(top-bottom)), 0.0, 0.0,
        ))
        gl.glDisable(self.BLEND)
        self.position_fbos[next_ping][0].bind()
        gl.glViewport(0, 0, self.count, 1)
        self._draw_quad()
        self.position_fbos[next_ping][0].release()
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
        self.history_fbos[0].bind()
        gl.glViewport(0, self.history_cursor, self.count, 1)
        self._draw_quad()
        self.history_fbos[0].release()
        self.store_program.release()
        gl.glBindFramebuffer(self.FRAMEBUFFER, self.defaultFramebufferObject())
        self.history_latest = self.history_cursor
        self.history_cursor = (self.history_cursor + 1) % self.history_length
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
        self._set_uniform(self.trail_program, "u_history_length", float(self.history_length))
        self._set_uniform(self.trail_program, "u_samples_per_frame", float(self.SAMPLES_PER_FRAME))
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
        self._set_uniform(self.trail_program, "u_dt", float(self.dt))
        self._set_uniform(self.trail_program, "u_normalize", 1 if self.normalize else 0)
        left, right, bottom, top = self.plot_rect
        gl.glEnable(self.SCISSOR_TEST)
        gl.glScissor(int(left*width), int(bottom*height),
                     max(1, int((right-left)*width)), max(1, int((top-bottom)*height)))
        gl.glDrawArrays(self.POINTS, 0, self.count*(self.history_length-1))
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
            # Sample simulation time independently of repaint frequency and
            # playback speed. Keep the unprocessed budget on a slow frame.
            self._sampling_budget += elapsed * self.NOMINAL_FPS * self.SAMPLES_PER_FRAME * self.speed
            samples = min(256, int(self._sampling_budget))
            self._sampling_budget -= samples
            self._respawn_due_particles(samples / (self.NOMINAL_FPS*self.SAMPLES_PER_FRAME))
            for _ in range(samples):
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
        self._equilibrium_artist = None
        self._marker, = self.ax.plot([], [], [], marker="o", markersize=5, linestyle="None")
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._advance)
        self._draw_empty()

    @staticmethod
    def _integrate(fields, params, initial, dt, steps=9000):
        f = lambda point: _field_vector(fields, point, params)
        times, points, reason = integrate_trajectory(
            f, initial, dt * (steps - 1), dt, samples=steps,
        )
        if len(points) < 2:
            raise ValueError(reason)
        speeds = np.asarray([np.linalg.norm(f(point)) for point in points])
        return points, speeds, reason

    def set_model(self, fields, params, initial, dt, dark, speed_color, playback_speed=1.0,
                  speed_max=None, trajectory_data=None):
        trajectory, speeds, reason = trajectory_data or self._integrate(fields, params, initial, dt)
        self._model_config = (fields, dict(params), tuple(initial), dt, dark, speed_color,
                              playback_speed, speed_max, trajectory_data)
        self.integration_status = reason
        self._speeds = speeds
        self.trajectory = trajectory
        self.index = 0
        self.set_playback_speed(playback_speed)
        self._dark = bool(dark)
        self.ax.clear()
        self._equilibrium_artist = None
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
                segments, cmap=speed_colormap(), norm=speed_normalization(
                    speed_max if speed_max is not None else max(np.percentile(speeds, 95), 1e-9)
                ), linewidth=0.75, alpha=0.9,
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
        if running and not len(self.trajectory) and hasattr(self, "_model_config"):
            self.set_model(*self._model_config)
        self.running = bool(running)
        if self.running:
            self.timer.start()
        else:
            self.timer.stop()

    def set_equilibrium_points(self, points, visible=True):
        if self._equilibrium_artist is not None:
            try:
                self._equilibrium_artist.remove()
            except ValueError:
                pass
            self._equilibrium_artist = None
        if visible and points:
            values = np.asarray([point for point, _details in points], dtype=float)
            self._equilibrium_artist = self.ax.scatter(
                values[:, 0], values[:, 1], values[:, 2],
                s=34, marker="o", color="#ffad33" if self._dark else "#d98200",
                edgecolors="none", linewidths=0, depthshade=False, zorder=8,
            )
        self.canvas.draw_idle()

    def set_playback_speed(self, speed):
        self.playback_speed = max(float(speed), 0.01)
        self._playback_remainder = 0.0
        if hasattr(self, "_model_config"):
            config = list(self._model_config)
            config[6] = self.playback_speed
            self._model_config = tuple(config)

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

    def update_appearance(self, dark, speed_color, speed_max=None):
        if not hasattr(self, "_model_config"):
            return
        camera = (self.ax.elev, self.ax.azim,
                  self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim())
        config = list(self._model_config)
        config[4:6] = [dark, speed_color]
        config[7] = speed_max
        if len(self.trajectory):
            config[8] = (self.trajectory.copy(), self._speeds.copy(), self.integration_status)
            index = self.index
            self.set_model(*config)
            self.index = index
        else:
            # Changing appearance after Clear must not restore a deleted path.
            self._model_config = tuple(config)
            self._dark = dark
            self.figure.set_facecolor("#000000" if dark else "#ffffff")
            self.ax.set_facecolor("#000000" if dark else "#ffffff")
            foreground = "#dedede" if dark else "#222222"
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                axis.label.set_color(foreground)
                axis.set_tick_params(colors=foreground)
                axis.set_pane_color((0, 0, 0, 0) if dark else (1, 1, 1, 0))
            self.ax.title.set_color(foreground)
        self.ax.view_init(elev=camera[0], azim=camera[1])
        self.ax.set_xlim(camera[2])
        self.ax.set_ylim(camera[3])
        self.ax.set_zlim(camera[4])
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

ODE_POLAR_MODEL_PRESETS = {
    "Uniform rotation": ("0", "ω", "ω=1"),
    "Radial decay": ("-a*r", "0", "a=1"),
    "Spiral sink": ("-a*r", "ω", "a=0.2, ω=1"),
    "Hopf normal form": ("μ*r-r^3", "ω", "μ=1, ω=1"),
}


class ODEFlowApp(QMainWindow):
    def __init__(self):
        super().__init__()
        if sys.platform == "win32":
            for font_file in ("times.ttf", "timesi.ttf", "timesbd.ttf", "timesbi.ttf"):
                QFontDatabase.addApplicationFont("C:\\Windows\\Fonts\\" + font_file)
        self.setWindowTitle("ODE Flow Lab — 2D / 3D Phase Portrait")
        self.resize(1360, 820)
        self.setMinimumWidth(900)

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
        self._equilibrium_cache_key = None
        self._equilibrium_results = []
        self._selected_solution = None
        self._parameter_editors = {}
        self._initial_states = {"2D": (1.0, 0.0, 0.0), "3D": (0.1, 0.0, 0.0)}
        self._loading_experiment = False
        self._previous_model_name = "Custom"
        self._coordinates = "Cartesian"
        self._custom_model_states = {
            "2D": ("x^3 - 2*x*y^2", "2*x^2*y - y^3", "", ""),
            "Polar": ("r*(1-r^2)", "1", "", ""),
            "3D": ODE_3D_MODEL_PRESETS["Lorenz attractor"][:4],
        }
        self._dimension = "2D"
        self._mode_states = {
            "2D": ("Custom", "x^3 - 2*x*y^2", "2*x^2*y - y^3", "", ""),
            "Polar": ("Custom", "r*(1-r^2)", "1", "", ""),
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
        splitter.setChildrenCollapsible(False)

        panel = QWidget()
        self.control_panel = panel
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(390)
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(12, 6, 12, 6)
        controls.setSpacing(4)

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
        self.coordinate_selector = QComboBox()
        self.coordinate_selector.addItems(["Cartesian", "Polar"])
        self.coordinate_selector.setToolTip(
            "Polar input: dr/dt and dθ/dt. The plot remains in x,y.\n"
            "θ=atan2(y,x), in radians from -π to π; r≥0.\n"
            "Use theta or θ. At r=0, only analytically extendable fields are defined.\n"
            "Angular dependence must be 2π-periodic to define a continuous field across the angle cut."
        )
        self.coordinate_selector.currentTextChanged.connect(self._on_coordinates_changed)
        self._coordinate_label = QLabel("Coordinates")
        self._coordinate_row = QWidget()
        coordinate_layout = QHBoxLayout(self._coordinate_row)
        coordinate_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_layout.setSpacing(4)
        coordinate_layout.addWidget(self.coordinate_selector, 1)
        self.to_cartesian_btn = QPushButton("To Cartesian")
        self.to_cartesian_btn.setToolTip("Convert the current polar equations into editable dx/dt and dy/dt formulas.")
        self.to_cartesian_btn.clicked.connect(self._convert_polar_to_cartesian)
        self.to_cartesian_btn.hide()
        coordinate_layout.addWidget(self.to_cartesian_btn)
        eq_form.addRow(self._coordinate_label, self._coordinate_row)
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
        self._active_formula_editor = self.fx_input
        self._active_calculator_editor = self.fx_input
        for editor in (self.fx_input, self.fy_input, self.fz_input):
            editor.setClearButtonEnabled(True)
            self._formula_history[editor] = {"undo": [], "redo": [], "last": editor.text()}
            editor.textChanged.connect(self._on_formula_edited)
        self._fx_label = QLabel("dx/dt =")
        self._fy_label = QLabel("dy/dt =")
        eq_form.addRow(self._fx_label, self.fx_input)
        eq_form.addRow(self._fy_label, self.fy_input)
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
            ("α", "α"), ("β", "β"), ("γ", "γ"),
            ("^", "^"), ("(", "("), (")", ")"), ("π", "pi"),
            ("sin", "sin("), ("cos", "cos("), ("tan", "tan("),
            ("exp", "exp("), ("sqrt", "sqrt("), ("log", "log("),
        ]
        self._coordinate_keys = {}
        for i, (label, value) in enumerate(symbols):
            btn = QPushButton(label)
            btn.setMaximumHeight(29)
            btn.setProperty("insertValue", value)
            btn.clicked.connect(lambda checked=False, button=btn: self.insert_symbol(button.property("insertValue")))
            if value in ("x", "y", "z"):
                self._coordinate_keys[value] = btn
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
        self.particle_count.setValue(1000)
        self._add_2d_setting(form, "Particles", 2, 0)
        form.addWidget(self.particle_count, 2, 1)
        self._two_d_only_widgets.append(self.particle_count)

        self.trail_length = NoWheelSpinBox()
        self.trail_length.setRange(3, 300)
        self.trail_length.setValue(100)
        self._add_2d_setting(form, "Trail frames", 2, 2)
        form.addWidget(self.trail_length, 2, 3)
        self._two_d_only_widgets.append(self.trail_length)

        self.respawn_period = self.make_spin(5.0, 1.0, 60.0, 1.0)
        self.respawn_period.setSuffix(" s")
        self._add_2d_setting(form, "Respawn cycle", 4, 0, 1, 2)
        form.addWidget(self.respawn_period, 4, 2, 1, 2)
        self._two_d_only_widgets.append(self.respawn_period)

        self.dt = self.make_spin(0.01, 0.001, 0.5, 0.005)
        self.dt.setToolTip("Positive integration step. Minimum: 0.001 with three decimal places.")
        form.addWidget(QLabel("Step"), 3, 0)
        form.addWidget(self.dt, 3, 1)
        self.speed = self.make_spin(1.0, 0.01, 10.0, 0.1, compact_decimals=True)
        form.addWidget(QLabel("Playback ×"), 3, 2)
        self.speed.setToolTip("Playback multiplier; does not change the vector field or the numerical sample step.")
        form.addWidget(self.speed, 3, 3)
        controls.addWidget(settings)

        self.show_field = QCheckBox("Vectors")
        self.show_field.setChecked(True)
        self.normalize = QCheckBox("Normalize direction")
        self.normalize.setToolTip("Move along F/|F|. Preserves flow direction, but not original equation time.\nSelected initial-value solutions always use the original field F.")
        self._two_d_only_widgets.extend((self.show_field, self.normalize))
        self.show_equilibria = QCheckBox("Show equilibria")
        self.show_equilibria.setToolTip("Mark detected equilibrium points in orange on the phase plot.")
        self.show_equilibria.setChecked(False)
        self.show_equilibria.toggled.connect(self._on_equilibrium_visibility_changed)
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
        view_options = QGridLayout()
        view_options.setHorizontalSpacing(8)
        view_options.addWidget(self.show_field, 0, 0)
        view_options.addWidget(self.normalize, 0, 1)
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
        self.clear_btn = QPushButton("Clear dynamic flow")
        self.clear_btn.setToolTip("Clear animated flow only; keep the selected initial-value trajectory.")
        self.clear_btn.clicked.connect(self.clear_trails)
        buttons.addWidget(self.clear_btn, 1, 0, 1, 2)
        self.reset_view_btn = QPushButton("Reset view · 10 × 10")
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.center_view_btn = QPushButton("Center origin")
        self.center_view_btn.setToolTip("Center the view at (0, 0) while preserving its current width and height.")
        self.center_view_btn.clicked.connect(self.center_view)
        for button in (self.reset_view_btn, self.center_view_btn):
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            button.setMinimumWidth(0)
        buttons.setColumnStretch(0, 1)
        buttons.setColumnStretch(1, 1)
        buttons.addWidget(self.reset_view_btn, 2, 0)
        buttons.addWidget(self.center_view_btn, 2, 1)
        self._two_d_only_widgets.append(self.center_view_btn)
        self._action_buttons = buttons
        controls.addLayout(buttons)

        self.status = QLabel("Ready · equations update as you edit")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #666666; padding: 5px 2px;")
        controls.addWidget(self.status)
        controls.addStretch()

        plot_panel = QWidget()
        plot_panel.setMinimumWidth(500)
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.flow_canvas = FlowCanvas()
        self.flow_canvas.gpuStatus.connect(self._on_gpu_status)
        self.flow_canvas.viewChanged.connect(self._on_canvas_view_changed)
        self.flow_canvas.initialPointPicked.connect(self._pick_initial_point)
        self.flow_canvas.canvasResized.connect(lambda: self._static_refresh_timer.start())
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.flow_canvas)
        self.canvas3d = Flow3DCanvas()
        self.view_stack.addWidget(self.canvas3d)
        self.right_tabs = QTabWidget()
        phase_page = QWidget()
        phase_layout = QVBoxLayout(phase_page)
        phase_layout.setContentsMargins(0, 0, 0, 0)
        phase_header = QHBoxLayout()
        phase_header.addWidget(self.show_equilibria)
        self.phase_hint = QLabel("Click: select only · outside: dismiss · drag: pan · scroll: zoom")
        self.phase_hint.setWordWrap(True)
        phase_header.addWidget(self.phase_hint, 1)
        phase_layout.addLayout(phase_header)
        phase_layout.addWidget(self.view_stack, 1)
        self.speed_legend = QLabel()
        self.speed_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speed_legend.setFixedHeight(44)
        self.speed_legend.hide()
        phase_layout.addWidget(self.speed_legend)
        self.right_tabs.addTab(phase_page, "Phase portrait")
        self._build_experience_tabs()
        plot_layout.addWidget(self.right_tabs, 1)
        equilibrium_box = QGroupBox("Equilibria · local stability")
        equilibrium_layout = QVBoxLayout(equilibrium_box)
        equilibrium_layout.setContentsMargins(8, 7, 8, 7)
        self.equilibrium_info = QLabel("Apply a system to find its equilibrium points.")
        self.equilibrium_info.setWordWrap(True)
        self.equilibrium_info.setTextFormat(Qt.TextFormat.RichText)
        self.equilibrium_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.equilibrium_info.setMinimumHeight(80)
        self.equilibrium_info.setStyleSheet("padding: 3px 2px;")
        self.equilibrium_scroll = QScrollArea()
        self.equilibrium_scroll.setWidgetResizable(True)
        self.equilibrium_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.equilibrium_scroll.setMinimumHeight(94)
        self.equilibrium_scroll.setMaximumHeight(190)
        self.equilibrium_scroll.setWidget(self.equilibrium_info)
        equilibrium_layout.addWidget(self.equilibrium_scroll)
        plot_layout.addWidget(equilibrium_box)

        splitter.addWidget(panel)
        splitter.addWidget(plot_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([360, 1000])

        self._mode_states["2D"] = self._read_editor_state()
        self._refresh_model_items("2D", "Custom")

        self._apply_theme(False)
        # Observe right-side child widgets too. Formula focus/undo is handled
        # here once, instead of registering this filter twice on each editor.
        QApplication.instance().installEventFilter(self)

    def _build_experience_tabs(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        initial_box = QGroupBox("Initial-value solution · original equation time")
        form = QGridLayout(initial_box)
        self.initial_spins = [self.make_spin(value, -1e6, 1e6, 0.1) for value in (1, 0, 0)]
        self.initial_labels = [QLabel(value) for value in ("x₀", "y₀", "z₀")]
        for column, (label, spin) in enumerate(zip(self.initial_labels, self.initial_spins)):
            form.addWidget(label, 0, column * 2)
            form.addWidget(spin, 0, column * 2 + 1)
            spin.valueChanged.connect(self._on_initial_changed)
        self.duration = self.make_spin(20, 0.001, 1000, 1)
        form.addWidget(QLabel("End time"), 1, 0)
        form.addWidget(self.duration, 1, 1)
        self.duration.valueChanged.connect(self._invalidate_selected_solution)
        self.trace_btn = QPushButton("Trace solution")
        self.trace_btn.clicked.connect(self._trace_from_controls)
        form.addWidget(self.trace_btn, 1, 2, 1, 2)
        self.show_selected = QCheckBox("Show on phase plot")
        self.show_selected.setChecked(True)
        self.show_selected.toggled.connect(self._on_selected_visibility_changed)
        form.addWidget(self.show_selected, 1, 4, 1, 2)
        layout.addWidget(initial_box)
        self.solution_status = QLabel("Click the 2D phase plot or enter an initial point, then Trace solution.\nPlayback and direction normalization do not change this solution.")
        self.solution_status.setWordWrap(True)
        layout.addWidget(self.solution_status)
        self.time_figure = Figure(figsize=(7, 4), dpi=100)
        self.time_canvas = FigureCanvasQTAgg(self.time_figure)
        layout.addWidget(self.time_canvas, 1)
        self.right_tabs.addTab(page, "Initial point · time")
        self.initial_labels[2].hide()
        self.initial_spins[2].hide()

        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        self.parameter_box = QGroupBox("Parameters")
        self.parameter_form = QFormLayout(self.parameter_box)
        content_layout.addWidget(self.parameter_box)
        self.parameter_note = QLabel("No parameters in this model. Params on the left accepts symbol=value pairs.")
        self.parameter_note.setWordWrap(True)
        content_layout.addWidget(self.parameter_note)

        colors = QGroupBox("Speed spectrum · original |F|, log(1 + |F|)")
        form = QFormLayout(colors)
        self.fixed_colors = QCheckBox("Use fixed color range")
        self.color_max = self.make_spin(10, 0.001, 1e9, 1)
        self.color_max.setEnabled(False)
        form.addRow(self.fixed_colors)
        form.addRow("Maximum |F|", self.color_max)
        self.color_note = QLabel("Automatic: 95th percentile in the current view (2D) or trajectory (3D).\nFixed: the same speed keeps the same color when the view or parameters change.")
        self.color_note.setWordWrap(True)
        form.addRow(self.color_note)
        self.fixed_colors.toggled.connect(self.color_max.setEnabled)
        self.fixed_colors.toggled.connect(self._on_color_range_changed)
        self.color_max.valueChanged.connect(self._on_color_range_changed)
        content_layout.addWidget(colors)

        files = QGroupBox("Save and export")
        buttons = QGridLayout(files)
        actions = (("Save experiment…", self.save_experiment), ("Load experiment…", self.load_experiment),
                   ("Export phase PNG…", self.export_phase_png), ("Export solution CSV…", self.export_solution_csv),
                   ("Export solution plot…", self.export_solution_plot))
        for i, (label, callback) in enumerate(actions):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button, i // 2, i % 2)
        content_layout.addWidget(files)
        note = QLabel("Experiment files preserve equations, parameters, initial coordinates and view settings.\nCSV contains equation time and coordinates. Solution plots support PNG, SVG and PDF.\nPhase PNG captures the current visible flow, including its speed legend.")
        note.setWordWrap(True)
        content_layout.addWidget(note)
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.right_tabs.addTab(page, "Parameters · files")

    def _solution_signature(self):
        fields = (self.fx, self.fy) if self._dimension == "2D" else (self.fx, self.fy, self.fz)
        return (self._dimension, tuple(ast.dump(field) for field in fields if field is not None),
                tuple(sorted(self.params.items())),
                tuple(spin.value() for spin in self.initial_spins[:len(fields)]),
                self.duration.value(), self.dt.value())

    def _invalidate_selected_solution(self, *_args):
        if self._selected_solution is not None:
            self._selected_solution = None
            self.solution_status.setText("Settings changed · Trace solution to recalculate.")
            self._draw_time_series()
            self._static_refresh_timer.start(80)

    def _on_initial_changed(self, *_args):
        self._initial_states[self._dimension] = tuple(spin.value() for spin in self.initial_spins)
        self._invalidate_selected_solution()
        if self._dimension == "3D" and not self._loading_experiment:
            self._auto_apply_timer.start()

    def _restore_initial_controls(self):
        for spin, value in zip(self.initial_spins, self._initial_states[self._dimension]):
            blocker = QSignalBlocker(spin)
            spin.setValue(value)
            del blocker
        is_3d = self._dimension == "3D"
        self.initial_spins[2].setVisible(is_3d)
        self.initial_labels[2].setVisible(is_3d)
        self._invalidate_selected_solution()

    def _pick_initial_point(self, x, y):
        self.stop_animation(clear_dynamic=True)
        blocker = QSignalBlocker(self.show_selected)
        self.show_selected.setChecked(True)
        del blocker
        for spin, value in zip(self.initial_spins, (x, y)):
            blocker = QSignalBlocker(spin)
            spin.setValue(value)
            del blocker
        self._initial_states["2D"] = tuple(spin.value() for spin in self.initial_spins)
        self.trace_solution()

    def _trace_from_controls(self, *_args):
        self.stop_animation()
        blocker = QSignalBlocker(self.show_selected)
        self.show_selected.setChecked(True)
        del blocker
        self.trace_solution()

    def _dismiss_selected_trace(self):
        if self._selected_solution is None or not self.show_selected.isChecked():
            return
        # Hide the phase overlay without discarding its time-series/CSV data.
        blocker = QSignalBlocker(self.show_selected)
        self.show_selected.setChecked(False)
        del blocker
        self._static_refresh_timer.stop()
        self._refresh_static_background()

    def _after_model_applied(self):
        self._last_apply_valid = True
        if self._selected_solution is not None and self._selected_solution["signature"] != self._solution_signature():
            self._invalidate_selected_solution()
        self._sync_parameter_fields()

    def trace_solution(self, *_args, apply_pending=True):
        pending = self._auto_apply_timer.isActive()
        self._auto_apply_timer.stop()
        try:
            # Commit pending edits first, so exported data and the visible field agree.
            if apply_pending and (pending or not getattr(self, "_last_apply_valid", False)):
                self.apply_model()
            if not self._last_apply_valid:
                return
            fields = (self.fx, self.fy) if self._dimension == "2D" else (self.fx, self.fy, self.fz)
            initial = [spin.value() for spin in self.initial_spins[:len(fields)]]
            plot_scale = None
            if self._dimension == "2D":
                xmin, xmax, ymin, ymax = self.flow_canvas.bounds
                left, right, bottom, top = self.flow_canvas.plot_rect
                ratio = self.flow_canvas.devicePixelRatioF()
                plot_scale = (
                    self.flow_canvas.width()*ratio*(right-left)/(xmax-xmin),
                    self.flow_canvas.height()*ratio*(top-bottom)/(ymax-ymin),
                )
            times, points, reason = integrate_trajectory(
                lambda point: _field_vector(fields, point, self.params), initial,
                self.duration.value(), self.dt.value(),
                retain_steps=self._dimension == "2D", spatial_scale=plot_scale,
            )
            self._selected_solution = {
                "times": times, "points": points, "reason": reason,
                "velocities": np.asarray([_field_vector(fields, point, self.params) for point in points]),
                "signature": self._solution_signature(), "experiment": self.experiment_data(),
            }
            self.solution_status.setText(f"{reason} · t = {times[0]:g} to {times[-1]:g} · {len(times)} samples\nOriginal equation time; adaptive integration with relative tolerance 10⁻⁷.")
            self._draw_time_series()
            if self._dimension == "3D" and self.show_selected.isChecked():
                speeds = np.array([np.linalg.norm(_field_vector(fields, p, self.params)) for p in points])
                self.canvas3d.set_model(fields, self.params, initial, self.dt.value(),
                    self.dark_mode_check.isChecked(), self.speed_color.isChecked(), self.speed.value(),
                    self.color_max.value() if self.fixed_colors.isChecked() else None,
                    (points, speeds, reason))
                self._update_speed_legend()
            else:
                self._static_refresh_timer.start(20)
        except Exception as exc:
            self._invalidate_selected_solution()
            self.solution_status.setText(f"Solution unavailable · {exc}")
            self.status.setText(f"Solution unavailable · {exc}")

    def _on_selected_visibility_changed(self, *_args):
        if self._dimension == "2D":
            self._static_refresh_timer.start(50)
        elif self._selected_solution is not None:
            # Rebuild the normal 3D path when hiding the selected solution.
            if not self.show_selected.isChecked():
                self.apply_model()
            else:
                self.trace_solution()

    def _draw_time_series(self):
        figure = self.time_figure
        figure.clear()
        ax = figure.add_subplot(111)
        dark = self.dark_mode_check.isChecked()
        background, foreground = ("#000000", "#eeeeee") if dark else ("#ffffff", "#111111")
        figure.set_facecolor(background)
        ax.set_facecolor(background)
        ax.tick_params(colors=foreground)
        for spine in ax.spines.values():
            spine.set_color(foreground)
        ax.set_xlabel("t · original equation time", color=foreground, fontfamily="Times New Roman")
        ax.set_ylabel("State", color=foreground, fontfamily="Times New Roman")
        ax.grid(alpha=0.15, color=foreground)
        if self._selected_solution is not None:
            data = self._selected_solution
            colors = ("#75bdff", "#ffac6d", "#cc9dff") if dark else ("#2563ad", "#bf5714", "#8449b5")
            for index, name in enumerate(("x", "y", "z")[:data["points"].shape[1]]):
                ax.plot(data["times"], data["points"][:, index], label=name, color=colors[index], linewidth=1.2)
            legend = ax.legend(facecolor=background, edgecolor="none")
            for text in legend.get_texts():
                text.set_color(foreground)
                text.set_fontfamily("Times New Roman")
        else:
            ax.text(0.5, 0.5, "Trace an initial-value solution to show x(t), y(t), z(t).",
                    transform=ax.transAxes, ha="center", color=foreground, wrap=True)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontfamily("Times New Roman")
        figure.tight_layout()
        self.time_canvas.draw_idle()

    def _sync_parameter_fields(self):
        # Preserve focus and unfinished text while editing an individual parameter.
        if set(self._parameter_editors) != set(self.params):
            while self.parameter_form.rowCount():
                self.parameter_form.removeRow(0)
            self._parameter_editors = {}
            for name in self.params:
                editor = QLineEdit()
                editor.setMaximumWidth(240)
                editor.setToolTip("Finite numeric value; Enter or leave the field to apply. Mouse wheel does not edit it.")
                editor.editingFinished.connect(lambda symbol=name: self._commit_parameter_field(symbol))
                label = QLabel(format_formula_input(name))
                label.setStyleSheet("font-family:Times New Roman; font-size:12pt;")
                self.parameter_form.addRow(label, editor)
                self._parameter_editors[name] = editor
        for name, editor in self._parameter_editors.items():
            if not editor.hasFocus():
                # Keep extra digits only when the raw Params value actually has them;
                # never display a rounded value while calculating another value.
                editor.setText(format(self.params[name], ".12g"))
        self.parameter_note.setText("Edit a symbol here, or use Params on the left for symbol=value pairs."
                                    if self.params else "No parameters in this model.")

    def _commit_parameter_field(self, name):
        if name not in self._parameter_editors:
            return
        editor = self._parameter_editors[name]
        try:
            value = float(normalize_display_expression(editor.text()))
            if not np.isfinite(value):
                raise ValueError("Value must be finite.")
            value = round(value, 3)
            params = parse_parameters(self.param_input.text())
            if name not in params:
                return
            params[name] = value
            editor.setText(f"{value:.3f}")
            self.param_input.setText(", ".join(f"{symbol}={format(number, '.12g')}" for symbol, number in params.items()))
        except Exception as exc:
            self.status.setText(f"Parameter {name} · {exc}")

    def _on_color_range_changed(self, *_args):
        if self._loading_experiment:
            return
        if self._dimension == "3D":
            self._update_3d_appearance()
        else:
            self._static_refresh_timer.start(50)

    def _update_3d_appearance(self):
        if not self._model_applied:
            return
        self.canvas3d.update_appearance(self.dark_mode_check.isChecked(), self.speed_color.isChecked(),
            self.color_max.value() if self.fixed_colors.isChecked() else None)
        self.canvas3d.set_equilibrium_points(self._equilibrium_results, self.show_equilibria.isChecked())
        self._update_speed_legend()

    def _update_speed_legend(self):
        enabled = self.speed_color.isChecked()
        self.speed_legend.setVisible(enabled)
        if not enabled:
            return
        maximum = self._speed_max
        if self._dimension == "3D" and len(self.canvas3d.trajectory):
            maximum = max(float(np.percentile(self.canvas3d._speeds, 95)), 1e-9)
        if self.fixed_colors.isChecked():
            maximum = self.color_max.value()
        dark = self.dark_mode_check.isChecked()
        signature = (float(maximum), dark, self.fixed_colors.isChecked())
        if getattr(self, "_speed_legend_signature", None) == signature:
            return
        bg, fg = ("#000000", "#eeeeee") if dark else ("#ffffff", "#111111")
        figure = Figure(figsize=(5.0, 0.4), dpi=110, facecolor=bg)
        ax = figure.add_axes((0.05, 0.66, 0.9, 0.26))
        from matplotlib.colorbar import ColorbarBase
        bar = ColorbarBase(ax, cmap=speed_colormap(), norm=speed_normalization(maximum), orientation="horizontal")
        bar.set_ticks([0, np.expm1(np.log1p(maximum)/2), maximum])
        ax.tick_params(colors=fg, labelsize=8, length=2, pad=1)
        label = "fixed" if self.fixed_colors.isChecked() else "auto"
        bar.set_label(f"|F| · log(1 + |F|) · {label} · values above maximum saturated", color=fg, fontsize=8, labelpad=0)
        output = io.BytesIO()
        FigureCanvasAgg(figure).print_png(output)
        self.speed_legend.setPixmap(QPixmap.fromImage(QImage.fromData(output.getvalue())))
        self._speed_legend_signature = signature
        figure.clear()

    def experiment_data(self):
        modes = dict(self._mode_states)
        modes[self._editor_state_key()] = self._read_editor_state()
        return {
            "format": "ODE Flow Lab", "version": 1,
            "dimension": self._dimension, "coordinates": self._coordinates,
            "modes": modes, "initial": [spin.value() for spin in self.initial_spins],
            "solution_computed": self._selected_solution is not None,
            "view3d": {"elevation": float(self.canvas3d.ax.elev), "azimuth": float(self.canvas3d.ax.azim),
                       "bounds": [list(self.canvas3d.ax.get_xlim()), list(self.canvas3d.ax.get_ylim()),
                                  list(self.canvas3d.ax.get_zlim())]},
            "bounds": list(self.flow_canvas.bounds),
            "settings": {
                "particles": self.particle_count.value(), "trail_frames": self.trail_length.value(),
                "respawn": self.respawn_period.value(), "step": self.dt.value(), "playback": self.speed.value(),
                "duration": self.duration.value(), "color_max": self.color_max.value(),
                "normalized": self.normalize.isChecked(), "speed_colors": self.speed_color.isChecked(),
                "dark": self.dark_mode_check.isChecked(), "vectors": self.show_field.isChecked(),
                "equilibria": self.show_equilibria.isChecked(), "fixed_colors": self.fixed_colors.isChecked(),
                "show_selected": self.show_selected.isChecked(),
            },
        }

    def _validate_experiment_equations(self, data):
        key = "3D" if data["dimension"] == "3D" else (
            "Polar" if data["coordinates"] == "Polar" else "2D")
        state = data["modes"][key]
        params = parse_parameters(state[4])
        texts = state[1:4] if key == "3D" else state[1:3]
        fields = [compile_formula(text, params) for text in texts]
        if key == "Polar":
            if {"r", "θ", "theta"}.intersection(params):
                raise ValueError("r and θ are coordinates, not parameters.")
            fields = polar_to_cartesian(*fields, params)
        _field_vector(fields, [0.2, 0.3, 0.4][:len(fields)], params)

    def save_experiment(self, *_args):
        path, _ = QFileDialog.getSaveFileName(self, "Save experiment", "ode-experiment.json", "Experiment (*.json)")
        if not path:
            return
        try:
            data = self.experiment_data()
            validate_experiment(json.loads(json.dumps(data)))
            self._validate_experiment_equations(data)
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.setText("Experiment saved.")
        except Exception as exc:
            self.status.setText(f"Save failed · {exc}")

    def load_experiment(self, *_args):
        path, _ = QFileDialog.getOpenFileName(self, "Load experiment", "", "Experiment (*.json)")
        if not path:
            return
        try:
            if Path(path).stat().st_size > 1000000:
                raise ValueError("Experiment file is too large.")
            data = validate_experiment(json.loads(Path(path).read_text(encoding="utf-8")))
            # Inactive modes may contain unfinished drafts. Validate the active
            # equations before changing the UI; drafts are checked when opened.
            self._validate_experiment_equations(data)
            self._restore_experiment(data)
            self.status.setText("Experiment loaded · paused.")
        except Exception as exc:
            self.status.setText(f"Load failed · {exc}")

    def _restore_experiment(self, data):
        self.stop_animation()
        self._auto_apply_timer.stop()
        self._static_refresh_timer.stop()
        self._loading_experiment = True
        self._updating_model_inputs = True
        controls = [self.dimension_selector, self.coordinate_selector, self.model_selector,
                    self.fx_input, self.fy_input, self.fz_input, self.param_input,
                    *self.initial_spins, self.duration, self.color_max, self.fixed_colors,
                    self.particle_count, self.trail_length, self.respawn_period, self.dt, self.speed,
                    self.normalize, self.speed_color, self.dark_mode_check, self.show_field,
                    self.show_equilibria, self.show_selected]
        blockers = [QSignalBlocker(widget) for widget in controls]
        try:
            self._mode_states = {key: tuple(value) for key, value in data["modes"].items()}
            self._coordinates = data["coordinates"]
            self.coordinate_selector.setCurrentText(self._coordinates)
            # Let the normal dimension handler update visibility, while keeping
            # the imported mode drafts intact and all control signals blocked.
            self._dimension = "3D" if data["dimension"] == "2D" else "2D"
            self._on_dimension_changed(data["dimension"])
            self._mode_states = {key: tuple(value) for key, value in data["modes"].items()}
            self.dimension_selector.setCurrentText(self._dimension)
            state = self._mode_states[self._editor_state_key()]
            selected = state[0] if state[0] in self._current_presets() else "Custom"
            self._refresh_model_items(self._dimension, selected)
            for widget, text in zip((self.fx_input, self.fy_input, self.fz_input, self.param_input), state[1:]):
                widget.setText(text)
            self._previous_model_name = selected
            self._custom_model_states[self._editor_state_key()] = tuple(state[1:])
            settings = data["settings"]
            for name, widget in (("particles", self.particle_count), ("trail_frames", self.trail_length),
                                 ("respawn", self.respawn_period), ("step", self.dt), ("playback", self.speed),
                                 ("duration", self.duration), ("color_max", self.color_max)):
                widget.setValue(settings[name])
            for name, widget in (("normalized", self.normalize), ("speed_colors", self.speed_color),
                                 ("dark", self.dark_mode_check), ("vectors", self.show_field),
                                 ("equilibria", self.show_equilibria), ("fixed_colors", self.fixed_colors),
                                 ("show_selected", self.show_selected)):
                widget.setChecked(settings[name])
            for spin, value in zip(self.initial_spins, data["initial"]):
                spin.setValue(value)
            self._initial_states[self._dimension] = tuple(data["initial"])
            self._selected_solution = None
            self._equilibrium_cache_key = None
            self._equilibrium_results = []
            self._set_equilibrium_message("Loading experiment…")
            self._set_view_bounds(data["bounds"])
            self.color_max.setEnabled(self.fixed_colors.isChecked())
            self._active_formula_editor = self.fx_input
            for editor in (self.fx_input, self.fy_input, self.fz_input):
                self._formula_history[editor] = {"undo": [], "redo": [], "last": editor.text()}
        finally:
            del blockers
            self._loading_experiment = False
            self._updating_model_inputs = False
        self._auto_apply_timer.stop()
        self._apply_theme(self.dark_mode_check.isChecked())
        self.apply_model()
        if data.get("solution_computed") and self._last_apply_valid:
            self.trace_solution(apply_pending=False)
        if "view3d" in data:
            view = data["view3d"]
            self.canvas3d.ax.view_init(elev=view["elevation"], azim=view["azimuth"])
            self.canvas3d.ax.set_xlim(view["bounds"][0])
            self.canvas3d.ax.set_ylim(view["bounds"][1])
            self.canvas3d.ax.set_zlim(view["bounds"][2])
            self.canvas3d.canvas.draw_idle()

    def export_phase_png(self, *_args):
        path, _ = QFileDialog.getSaveFileName(self, "Export phase portrait", "phase-portrait.png", "PNG (*.png)")
        if not path:
            return
        try:
            if self._dimension == "2D":
                if not self.flow_canvas.ready:
                    raise ValueError("The phase view is not ready.")
                self._refresh_static_background()
                frame = self.flow_canvas.grabFramebuffer()
                if frame.isNull():
                    raise ValueError("The phase view is not ready.")
            else:
                output = io.BytesIO()
                self.canvas3d.figure.savefig(output, format="png", dpi=240,
                                            facecolor=self.canvas3d.figure.get_facecolor())
                frame = QImage.fromData(output.getvalue())
            # Framebuffer images carry the display's device pixel ratio. The
            # export canvas uses physical pixels, so do not draw a scaled-down
            # image into a larger physical canvas on high-DPI displays.
            frame.setDevicePixelRatio(1.0)
            if self.speed_color.isChecked():
                self._update_speed_legend()
                legend = self.speed_legend.pixmap().toImage()
                legend.setDevicePixelRatio(1.0)
                scale = min(frame.width() / legend.width(), frame.width() / 600)
                width, height = round(legend.width() * scale), round(legend.height() * scale)
                combined = QImage(frame.width(), frame.height() + height + 8, QImage.Format.Format_RGB32)
                combined.fill(Qt.GlobalColor.black if self.dark_mode_check.isChecked() else Qt.GlobalColor.white)
                painter = QPainter(combined)
                painter.drawImage(0, 0, frame)
                painter.drawImage(QRect((frame.width()-width)//2, frame.height()+4, width, height), legend)
                painter.end()
                frame = combined
            if not frame.save(path, "PNG"):
                raise ValueError("Could not write the image.")
            self.status.setText("Phase portrait exported.")
        except Exception as exc:
            self.status.setText(f"Export failed · {exc}")

    def export_solution_csv(self, *_args):
        if self._selected_solution is None:
            self.status.setText("Trace a solution before exporting CSV.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export solution", "solution.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            data = self._selected_solution
            header = json.dumps(data["experiment"], ensure_ascii=False) + "\n" + data["reason"]
            columns = ",".join(("t", "x", "y", "z")[:data["points"].shape[1]+1])
            with Path(path).open("w", encoding="utf-8", newline="") as output:
                for line in header.splitlines():
                    output.write("# " + line + "\n")
                output.write(columns + "\n")
                np.savetxt(output, np.column_stack((data["times"], data["points"])), delimiter=",", fmt="%.12g")
            self.status.setText("Solution CSV exported with equation and parameter metadata.")
        except Exception as exc:
            self.status.setText(f"Export failed · {exc}")

    def export_solution_plot(self, *_args):
        if self._selected_solution is None:
            self.status.setText("Trace a solution before exporting its plot.")
            return
        path, selected = QFileDialog.getSaveFileName(self, "Export solution plot", "solution.svg", "SVG (*.svg);;PDF (*.pdf);;PNG (*.png)")
        if not path:
            return
        try:
            if not Path(path).suffix:
                path += "." + selected.split("*.")[1].split(")")[0]
            self.time_figure.savefig(path, dpi=240, facecolor=self.time_figure.get_facecolor())
            self.status.setText("Solution plot exported.")
        except Exception as exc:
            self.status.setText(f"Export failed · {exc}")

    @staticmethod
    def make_spin(value, minimum, maximum, step, compact_decimals=False):
        spin = NoWheelDoubleSpinBox()
        spin.setProperty("compactDecimals", compact_decimals)
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
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
                margin-top: 6px; padding: 4px 7px 4px; font-weight: 600; color: {foreground}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; }}
            QLabel {{ color: {secondary}; }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {background}; border: 1px solid {field_border};
                border-radius: 4px; padding: 3px 6px; color: {foreground}; selection-background-color: #777777; }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {foreground}; }}
            QPushButton {{ background: {background}; border: 1px solid {field_border}; border-radius: 4px;
                padding: 4px; color: {foreground}; font-weight: 600; }}
            QPushButton:hover {{ background: {hover}; border-color: #888888; }}
            QCheckBox {{ spacing: 8px; color: {secondary}; }}
            QCheckBox::indicator {{ width: 15px; height: 15px; }}
            QSplitter::handle {{ background: {split}; width: 1px; }}
        """)
        for editor in (self.fx_input, self.fy_input, self.fz_input):
            editor.setStyleSheet(
                f"QLineEdit {{ background-color: {background}; border: 1px solid {field_border}; "
                f"border-radius: 4px; padding: 4px 6px; color: {foreground}; }}"
                f"QLineEdit:focus {{ border: 1px solid {foreground}; }}"
            )
        self.title_label.setStyleSheet(f"font-size: 21px; font-weight: 700; color: {foreground};")
        self.status.setStyleSheet(f"color: {secondary}; padding: 5px 2px;")
        scroll_track = "#1c1c1c" if dark else "#f3f3f3"
        scroll_thumb = "#6c6c6c" if dark else "#b0b0b0"
        scroll_hover = "#888888" if dark else "#939393"
        self.equilibrium_scroll.setStyleSheet(f"""
            QScrollBar:vertical {{ background: {scroll_track}; width: 6px;
                border: none; border-radius: 3px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {scroll_thumb};
                min-height: 24px; border-radius: 3px; }}
            QScrollBar::handle:vertical:hover {{ background: {scroll_hover}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0; width: 0; border: none; background: transparent; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
            QScrollBar:horizontal {{ background: {scroll_track}; height: 6px;
                border: none; border-radius: 3px; margin: 0; }}
            QScrollBar::handle:horizontal {{ background: {scroll_thumb};
                min-width: 24px; border-radius: 3px; }}
            QScrollBar::handle:horizontal:hover {{ background: {scroll_hover}; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                height: 0; width: 0; border: none; background: transparent; }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
            QScrollBar::up-arrow, QScrollBar::down-arrow,
            QScrollBar::left-arrow, QScrollBar::right-arrow {{
                image: none; border: none; width: 0; height: 0; }}
        """)
        if getattr(self, "_equilibrium_rows", None):
            self._render_equilibrium_rows(self._equilibrium_rows)
        elif getattr(self, "_equilibrium_message", None):
            self._set_equilibrium_message(self._equilibrium_message)
        if hasattr(self, "time_figure"):
            self._draw_time_series()

    def _on_visual_options_changed(self, _checked=False):
        dark = self.dark_mode_check.isChecked()
        self._apply_theme(dark)
        if hasattr(self, "flow_canvas"):
            self.flow_canvas.set_visual_options(self.speed_color.isChecked(), dark, self._speed_max)
        if self._dimension == "3D" and self._model_applied:
            self._update_3d_appearance()
        else:
            self._static_refresh_timer.start(80)

    def _on_equilibrium_visibility_changed(self, _checked=False):
        if self._dimension == "2D":
            self._static_refresh_timer.start(80)
        elif hasattr(self, "canvas3d"):
            self.canvas3d.set_equilibrium_points(
                self._equilibrium_results, self.show_equilibria.isChecked()
            )

    def _set_view_bounds(self, bounds):
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        values = np.asarray([xmin, xmax, ymin, ymax])
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1e9:
            return
        # The view and its three-decimal controls must describe the same box.
        xmin, xmax, ymin, ymax = (round(v, 3) for v in values)
        if xmax - xmin < 0.001 or ymax - ymin < 0.001:
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

    def _editor_state_key(self):
        return "Polar" if self._dimension == "2D" and self._coordinates == "Polar" else self._dimension

    def _current_presets(self):
        if self._dimension == "3D":
            return ODE_3D_MODEL_PRESETS
        return ODE_POLAR_MODEL_PRESETS if self._coordinates == "Polar" else ODE_MODEL_PRESETS

    def _update_coordinate_controls(self):
        polar = self._editor_state_key() == "Polar"
        self._fx_label.setText("dr/dt =" if polar else "dx/dt =")
        self._fy_label.setText("dθ/dt =" if polar else "dy/dt =")
        self.fx_input.setPlaceholderText("e.g. r*(1-r^2)" if polar else "e.g. x^2 - y")
        self.fy_input.setPlaceholderText("e.g. 1 (radians/time)" if polar else "e.g. x + y^2")
        for key, value in (("x", "r" if polar else "x"), ("y", "θ" if polar else "y")):
            self._coordinate_keys[key].setText(value)
            self._coordinate_keys[key].setProperty("insertValue", value)
        self._coordinate_keys["z"].setEnabled(not polar)
        self.to_cartesian_btn.setVisible(polar)
        for widget in (self._coordinate_label, self._coordinate_row):
            widget.setVisible(self._dimension == "2D")

    def _convert_polar_to_cartesian(self):
        if self._editor_state_key() != "Polar":
            return
        self._auto_apply_timer.stop()
        try:
            self.read_model()
            fx, fy = (coordinate_formula_for_editor(field, self.params) for field in (self.fx, self.fy))
            # Validate the actual editable text before replacing the Cartesian
            # draft. Invalid polar input must leave both modes untouched.
            for formula in (fx, fy):
                field = compile_formula(formula, self.params)
                evaluate(field, np.array([0.2, 0.5]), np.array([0.3, -0.2]), self.params)
            params = self.param_input.text()
            self._custom_model_states["2D"] = (fx, fy, "", params)
            self._mode_states["2D"] = ("Custom", fx, fy, "", params)
            self.coordinate_selector.setCurrentText("Cartesian")
            self._auto_apply_timer.stop()
            self.apply_model()
        except Exception as exc:
            self.status.setText(f"Coordinate conversion error · {exc}")

    def _on_coordinates_changed(self, coordinates):
        if coordinates == self._coordinates or self._dimension != "2D":
            return
        self._mode_states[self._editor_state_key()] = self._read_editor_state()
        self.stop_animation()
        self._coordinates = coordinates
        state = self._mode_states[self._editor_state_key()]
        self._refresh_model_items("2D", state[0])
        self._updating_model_inputs = True
        try:
            _, fx, fy, fz, params = state
            self.param_input.setText(params)
            parameter_names = self._formula_parameter_names()
            self.fx_input.setText(format_formula_input(fx, parameter_names))
            self.fy_input.setText(format_formula_input(fy, parameter_names))
            self.fz_input.setText(fz)
            self._previous_model_name = state[0]
        finally:
            self._updating_model_inputs = False
        self._update_coordinate_controls()
        self._active_formula_editor = self.fx_input
        self._equilibrium_cache_key = None
        self._auto_apply_timer.start()

    def _refresh_model_items(self, dimension, selected):
        blocker = QSignalBlocker(self.model_selector)
        self.model_selector.clear()
        self.model_selector.addItem("Custom")
        self.model_selector.addItems(self._current_presets())
        self.model_selector.setCurrentText(selected)
        del blocker

    def _on_dimension_changed(self, dimension):
        if dimension == self._dimension or not hasattr(self, "model_selector"):
            return
        self._initial_states[self._dimension] = tuple(spin.value() for spin in self.initial_spins)
        self._mode_states[self._editor_state_key()] = self._read_editor_state()
        self.stop_animation()
        self._dimension = dimension
        self._restore_initial_controls()
        state = self._mode_states[self._editor_state_key()]
        self._refresh_model_items(dimension, state[0])
        self.view_stack.setCurrentIndex(0 if dimension == "2D" else 1)
        self.phase_hint.setText(
            "Click: select only · outside: dismiss · drag: pan · scroll: zoom" if dimension == "2D" else
            "Drag: rotate · scroll: zoom · edit initial coordinates in the time tab"
        )
        is_2d = dimension == "2D"
        for widget in self._two_d_only_widgets:
            widget.setVisible(is_2d)
        self.clear_btn.setText("Clear dynamic flow")
        self.reset_view_btn.setText("Reset view · 10 × 10" if is_2d else "Reset 3D view")
        self._action_buttons.removeWidget(self.reset_view_btn)
        self._action_buttons.addWidget(self.reset_view_btn, 2, 0, 1, 1 if is_2d else 2)
        for widget in (self._fz_label, self.fz_input):
            widget.setVisible(dimension == "3D")
        self._update_coordinate_controls()
        self._updating_model_inputs = True
        try:
            _, fx, fy, fz, params = state
            self.param_input.setText(params)
            parameter_names = self._formula_parameter_names()
            self.fx_input.setText(format_formula_input(fx, parameter_names))
            self.fy_input.setText(format_formula_input(fy, parameter_names))
            self.fz_input.setText(format_formula_input(fz, parameter_names))
            self._previous_model_name = state[0]
        finally:
            self._updating_model_inputs = False
        self._auto_apply_timer.start()

    def _on_model_selected(self, name):
        if self._updating_model_inputs:
            return
        presets = self._current_presets()
        if self._previous_model_name == "Custom" and name != "Custom":
            self._custom_model_states[self._editor_state_key()] = (
                self.fx_input.text(), self.fy_input.text(), self.fz_input.text(), self.param_input.text())
        self._updating_model_inputs = True
        try:
            if name == "Custom":
                values = self._custom_model_states[self._editor_state_key()]
            else:
                values = presets[name][:4] if self._dimension == "3D" else (*presets[name][:2], "", presets[name][2])
            self.param_input.setText(values[3])
            parameter_names = self._formula_parameter_names()
            self.fx_input.setText(format_formula_input(values[0], parameter_names))
            self.fy_input.setText(format_formula_input(values[1], parameter_names))
            self.fz_input.setText(format_formula_input(values[2], parameter_names))
        finally:
            self._updating_model_inputs = False
        self._previous_model_name = name
        if self._dimension == "3D" and name in ODE_3D_MODEL_PRESETS:
            self._initial_states["3D"] = ODE_3D_MODEL_PRESETS[name][4]
            self._restore_initial_controls()
        self._mode_states[self._editor_state_key()] = self._read_editor_state()
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
        if self._updating_model_inputs:
            return
        self._invalidate_selected_solution()
        parameter_names = self._formula_parameter_names()
        formula_text = re.sub(r"\btheta\b", "θ", text) if self._editor_state_key() == "Polar" else text
        formatted = format_formula_input(formula_text, parameter_names)
        if formatted != text:
            cursor = sender.cursorPosition()
            prefix = text[:cursor]
            if self._editor_state_key() == "Polar":
                prefix = re.sub(r"\btheta\b", "θ", prefix)
            formatted_cursor = len(format_formula_input(prefix, parameter_names))
            blocker = QSignalBlocker(sender)
            sender.setText(formatted)
            sender.setCursorPosition(min(formatted_cursor, len(formatted)))
            del blocker
            text = formatted
        history["last"] = text
        self._remember_custom_equations()

    def _formula_parameter_names(self):
        parameter_text = getattr(self, "param_input", None)
        if parameter_text is None:
            return set()
        names = set()
        for item in parameter_text.text().split(","):
            match = re.match(r"\s*([^=,\s]+)\s*=", item)
            if match and match.group(1).isidentifier():
                names.add(match.group(1))
        return names

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ToolTip and watched in (self.flow_canvas, self.canvas3d.canvas):
            return True
        if (event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and self._dimension == "2D"
                and self.right_tabs.currentIndex() == 0
                and self._selected_solution is not None
                and self.show_selected.isChecked()
                and isinstance(watched, QWidget)
                and (watched is self or self.isAncestorOf(watched))):
            global_point = event.globalPosition().toPoint()
            on_controls = self.control_panel.rect().contains(self.control_panel.mapFromGlobal(global_point))
            on_plot = (self.flow_canvas.isVisible()
                       and self.flow_canvas._screen_to_view(self.flow_canvas.mapFromGlobal(global_point)) is not None)
            tab_bar = self.right_tabs.tabBar()
            on_tab_bar = tab_bar.rect().contains(tab_bar.mapFromGlobal(global_point))
            # The explicit visibility checkbox must keep its usual toggle
            # behavior; all other right-side clicks outside the axes dismiss.
            if not on_controls and not on_plot and not on_tab_bar and watched is not self.show_selected:
                self._dismiss_selected_trace()
        if event.type() == QEvent.Type.FocusIn and watched in (self.fx_input, self.fy_input, self.fz_input, self.param_input):
            self._active_calculator_editor = watched
        if watched in self._formula_history and event.type() == QEvent.Type.FocusIn:
            self._active_formula_editor = watched
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
        self._invalidate_selected_solution()
        self._remember_custom_equations()

    def _remember_custom_equations(self):
        self._custom_model_states[self._editor_state_key()] = (
            self.fx_input.text(), self.fy_input.text(), self.fz_input.text(), self.param_input.text())
        if self.model_selector.currentText() != "Custom":
            self.model_selector.setCurrentText("Custom")
        else:
            self._mode_states[self._editor_state_key()] = self._read_editor_state()
            self._auto_apply_timer.start()

    def _apply_model_after_edit(self):
        self.apply_model()

    def insert_symbol(self, value):
        focused = QApplication.focusWidget()
        if focused in (self.fx_input, self.fy_input, self.fz_input, self.param_input):
            self._active_calculator_editor = focused
        if self._active_calculator_editor is self.param_input:
            self.param_input.insert(value)
            self.param_input.setFocus()
            return
        if focused in (self.fx_input, self.fy_input, self.fz_input):
            self._active_formula_editor = focused
        widget = self._active_formula_editor
        if self._dimension == "2D" and widget is self.fz_input:
            widget = self.fx_input
            self._active_formula_editor = widget
        text = widget.text()
        selection_start = widget.selectionStart()
        if selection_start >= 0:
            start = selection_start
            end = start + len(widget.selectedText())
        else:
            start = end = widget.cursorPosition()
        inserted = text[:start] + value + text[end:]
        widget.setText(inserted)
        cursor = len(format_formula_input(
            text[:start] + value, self._formula_parameter_names()
        ))
        widget.setCursorPosition(min(cursor, len(widget.text())))
        widget.setFocus()

    def read_model(self):
        params = parse_parameters(self.param_input.text())
        polar = self._editor_state_key() == "Polar"
        if polar and {"r", "θ", "theta"}.intersection(params):
            raise ValueError("r and θ are coordinates in Polar mode, not parameters.")
        expressions = [self.fx_input.text(), self.fy_input.text()]
        if polar:
            expressions = [re.sub(r"\btheta\b", "θ", value) for value in expressions]
        fx, fy = (compile_formula(value, params) for value in expressions)
        if polar:
            fx, fy = polar_to_cartesian(fx, fy, params)
        # Check formula at a vector point; catches most input mistakes.
        test_x = np.array([0.2, 0.5])
        test_y = np.array([0.3, -0.2])
        if self._dimension == "2D":
            evaluate(fx, test_x, test_y, params)
            evaluate(fy, test_x, test_y, params)
        else:
            fz = compile_formula(self.fz_input.text(), params)
            evaluate_3d(fx, 0.2, 0.3, 0.4, params)
            evaluate_3d(fy, 0.2, 0.3, 0.4, params)
            evaluate_3d(fz, 0.2, 0.3, 0.4, params)
            self.fz = fz
        self.params, self.fx, self.fy = params, fx, fy

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
        self._last_apply_valid = False
        was_running = self.running
        try:
            self.read_model()
            if was_running:
                self.stop_animation()
            if self._dimension == "3D":
                initial = tuple(spin.value() for spin in self.initial_spins)
                self.canvas3d.set_model(
                    (self.fx, self.fy, self.fz), self.params, initial,
                    self.dt.value(), self.dark_mode_check.isChecked(),
                    self.speed_color.isChecked(), self.speed.value(),
                    self.color_max.value() if self.fixed_colors.isChecked() else None,
                )
                self._model_applied = True
                self._update_equilibrium_info()
                self._gpu_ready = True
                self.status.setText(f"3D trajectory ready · {self.canvas3d.integration_status}.")
                self._after_model_applied()
                self._update_speed_legend()
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
            self._update_equilibrium_info(config["bounds"])
            static_image, plot_rect = self.render_static_background(self._active_bounds)
            self.flow_canvas.set_model(
                config["vx"], config["vy"], config["bounds"],
                config["particleCount"], config["trailLength"], config["dt"], config["speed"],
                config["normalize"], config["respawnPeriod"], config["speedColor"],
                config["darkMode"], self._speed_max, static_image, plot_rect,
            )
            self._model_applied = True
            self._after_model_applied()
            if self.flow_canvas.ready:
                self.status.setText("GPU particle flow ready.")
                if was_running:
                    self.toggle_animation()
            elif not self._gpu_error:
                self.status.setText("Preparing OpenGL flow…")
        except Exception as exc:
            self.stop_animation()
            self._equilibrium_results = []
            self._equilibrium_cache_key = None
            self.status.setText(f"Input error · {exc}")
            if hasattr(self, "equilibrium_info"):
                self._set_equilibrium_message("Equilibrium analysis unavailable until the system is valid.")

    def _update_equilibrium_info(self, view_bounds=None):
        if not hasattr(self, "equilibrium_info") or self.fx is None or self.fy is None:
            return
        if self._dimension == "2D":
            xmin, xmax, ymin, ymax = view_bounds or self._active_bounds
            bounds = ((xmin, xmax), (ymin, ymax))
            fields = (self.fx, self.fy)
            search_note = "Search region: current 2D view"
        else:
            trajectory = self.canvas3d.trajectory
            if trajectory.size == 0 or self.fz is None:
                self._set_equilibrium_message("No 3D trajectory available for equilibrium search.")
                return
            minimum = np.minimum(trajectory.min(axis=0), 0.0)
            maximum = np.maximum(trajectory.max(axis=0), 0.0)
            padding = np.maximum((maximum - minimum) * 0.06, 0.5)
            bounds = np.column_stack((minimum - padding, maximum + padding))
            fields = (self.fx, self.fy, self.fz)
            search_note = "Search region: 3D trajectory range"

        key = (
            self._dimension,
            tuple(ast.dump(field, include_attributes=False) for field in fields),
            tuple(sorted(self.params.items())),
            tuple(np.round(np.asarray(bounds, dtype=float).ravel(), 9)),
        )
        if key == self._equilibrium_cache_key:
            if self._dimension == "3D":
                self.canvas3d.set_equilibrium_points(
                    self._equilibrium_results, self.show_equilibria.isChecked()
                )
            return
        self._set_equilibrium_message(f"{search_note} · Searching equilibria…")
        try:
            x_axis_only = False
            if self._dimension == "2D":
                try:
                    fy_names = {node.id for node in ast.walk(fields[1]) if isinstance(node, ast.Name)}
                    if not fy_names.intersection({"x", "y"}):
                        x_axis_only = float(eval_node(fields[1], self.params)) == 0.0
                except (ArithmeticError, TypeError, ValueError, OverflowError):
                    pass
            if x_axis_only:
                search_note = "Search restricted to y = 0 because dy/dt ≡ 0"
            results = find_equilibria(fields, self.params, bounds, x_axis_only=x_axis_only)
            self._equilibrium_results = results
            if not results:
                self._set_equilibrium_message(f"{search_note} · No isolated equilibrium detected.")
            else:
                rows = []
                if x_axis_only:
                    rows.append("Search restricted to y = 0; the full system may also have non-isolated equilibria away from the x-axis.")
                for point, details in results:
                    coordinates = ", ".join(f"{value:.5g}" for value in point)
                    linear_result = f"{details['type']} · {details['linearization']}"
                    hyperbolic_text = (
                        "Hyperbolic" if details["hyperbolic"] else "Non-hyperbolic"
                    ) if details["hyperbolic"] is not None else "Hyperbolicity unavailable"
                    rows.append(
                        f"Equilibrium point: $({coordinates})$    "
                        f"$\\lambda=({details['eigenvalues']})$    "
                        f"{hyperbolic_text}    "
                        f"Linear result: {linear_result}"
                    )
            if results:
                self._render_equilibrium_rows(rows)
            self._equilibrium_cache_key = key
            if self._dimension == "3D":
                self.canvas3d.set_equilibrium_points(
                    self._equilibrium_results, self.show_equilibria.isChecked()
                )
        except Exception as exc:
            self._equilibrium_results = []
            self._set_equilibrium_message(f"Equilibrium search unavailable: {exc}")

    def _render_equilibrium_rows(self, rows):
        self._equilibrium_message = None
        dark = self.dark_mode_check.isChecked()
        foreground = "#ffffff" if dark else "#000000"
        self._equilibrium_rows = list(rows)
        blocks = []
        for row in rows:
            segments = re.split(r"(\$[^$]*\$)", row)
            rendered = []
            for segment in segments:
                if segment.startswith("$") and segment.endswith("$"):
                    rendered.append(self._math_text_html(segment[1:-1], foreground))
                else:
                    rendered.append(html.escape(segment).replace("  ", "&nbsp;&nbsp;"))
            blocks.append(
                "<div style='margin:0 0 5px 0; white-space:normal; line-height:1.5;'>"
                + "".join(rendered) + "</div>"
            )
        self.equilibrium_info.setText(
            f"<div style='color:{foreground}; font-family:Times New Roman,serif; "
            f"font-size:11pt; letter-spacing:0;'>{''.join(blocks)}</div>"
        )
        self.equilibrium_info.setMinimumSize(0, 80)

    def _set_equilibrium_message(self, message):
        self._equilibrium_rows = []
        self._equilibrium_message = message
        foreground = "#ffffff" if self.dark_mode_check.isChecked() else "#000000"
        self.equilibrium_info.setText(
            f"<span style='color:{foreground}; font-family:Times New Roman,serif; "
            f"font-size:11pt; letter-spacing:0;'>{html.escape(message)}</span>"
        )

    @staticmethod
    def _math_text_html(expression, foreground):
        expression = expression.replace("<", "\ue000").replace(">", "\ue001")
        expression = html.escape(expression)
        expression = expression.replace(r"\delta ", "δ")
        expression = expression.replace(r"\cdot ", " · ")
        expression = expression.replace(r"\lambda", "λ")
        expression = expression.replace(r"\dot V", "V̇")
        expression = expression.replace(r"\leq", "≤")
        expression = expression.replace(r"\quad", "  ")
        expression = re.sub(r"\\(?:mathrm|mathit)\{([^{}]*)\}", r"\1", expression)
        expression = expression.replace("+", " + ").replace("-", "−")
        expression = re.sub(r"\s+", " ", expression).replace("=", " = ")
        expression = re.sub(r"−\s+(?=\d)", "−", expression)
        expression = re.sub(r"([A-Za-z]+̇?|[λδ])", r"<i>\1</i>", expression)
        expression = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", expression)
        expression = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", expression)
        expression = expression.replace("\ue000", "&lt;").replace("\ue001", "&gt;")
        return (f"<span style='font-family:Times New Roman,serif; font-size:11pt; "
                f"color:{foreground};'>{expression}</span>")

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
        if self.fixed_colors.isChecked():
            self._speed_max = self.color_max.value()
        self._update_speed_legend()
        if self._selected_solution is not None and self.show_selected.isChecked():
            solution = self._selected_solution
            if solution["signature"] == self._solution_signature():
                left, bottom, width, height = ax.get_position().bounds
                plot_scale = np.array([width*pixel_width/(xmax-xmin), height*pixel_height/(ymax-ymin)])
                points = trajectory_curve_points(solution["times"], solution["points"],
                    solution["velocities"], plot_scale)
                color = "#c4a0ff" if dark else "#8449b5"
                line, = ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.3,
                        linestyle="-", antialiased=True, solid_joinstyle="round", solid_capstyle="round",
                        zorder=5)
                # Keep curvature-controlled points; renderer simplification may
                # otherwise turn a precisely sampled curve back into long chords.
                line.get_path().should_simplify = False
                ax.scatter(*points[0, :2], color=color, s=18, edgecolors="none", zorder=6)

        if self.show_field.isChecked():
            mag = np.hypot(U, V)
            ax.quiver(
                X, Y, U / np.maximum(mag, 1e-10), V / np.maximum(mag, 1e-10),
                color=primary, alpha=0.42, pivot="mid", scale=38,
                width=0.002
            )
        if self.show_equilibria.isChecked() and self._equilibrium_results:
            points = np.asarray([point for point, _details in self._equilibrium_results])
            visible = (
                (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
            )
            if np.any(visible):
                ax.scatter(
                    points[visible, 0], points[visible, 1], s=35,
                    color="#ffad33" if dark else "#d98200",
                    edgecolors="none", linewidths=0, zorder=6,
                )
        canvas.draw()
        left, bottom, width, height = ax.get_position().bounds
        plot_rect = (left, left + width, bottom, bottom + height)
        output = io.BytesIO()
        canvas.print_png(output)
        figure.clear()
        return output.getvalue(), plot_rect

    def _refresh_static_background(self):
        if not self._model_applied or self._dimension != "2D" or not self._last_apply_valid:
            return
        try:
            self._update_equilibrium_info(self._active_bounds)
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
        self._invalidate_selected_solution()
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

    def center_view(self):
        if self._dimension != "2D":
            return
        xmin, xmax, ymin, ymax = self.flow_canvas.bounds
        half_width, half_height = (xmax-xmin)/2, (ymax-ymin)/2
        self._set_view_bounds((-half_width, half_width, -half_height, half_height))

    def clear_trails(self):
        if self._dimension == "2D":
            self.flow_canvas.clear_trails()
        elif self._model_applied:
            self.stop_animation()
            if self._selected_solution is not None and self.show_selected.isChecked():
                self.canvas3d._marker.set_data([], [])
                self.canvas3d._marker.set_3d_properties([])
                self.canvas3d.canvas.draw_idle()
            else:
                self.canvas3d.clear_trails()
                self.status.setText("Dynamic path cleared · Start regenerates it.")

    def stop_animation(self, clear_dynamic=False):
        self.running = False
        if hasattr(self, "flow_canvas"):
            self.flow_canvas.set_running(False)
        if hasattr(self, "canvas3d"):
            self.canvas3d.set_running(False)
        if hasattr(self, "start_btn"):
            self.start_btn.setText("Start")
        if clear_dynamic:
            self.clear_trails()

    def toggle_animation(self):
        if self.running:
            self.stop_animation()
            self.status.setText("Paused.")
            return
        if not getattr(self, "_last_apply_valid", False):
            self.status.setText("Correct the equation inputs before starting.")
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
