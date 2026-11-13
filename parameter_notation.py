"""Reversible parameter subscripts and symbolic superscripts (never Python eval)."""
from collections import defaultdict, deque
import re


_SUB = dict(zip("0123456789aehijklmnoprstuvxβγρφχ",
                "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵦᵧᵨᵩᵪ"))
_UNSUB = {value: key for key, value in _SUB.items()}
_UP = {"+": "⁺", "-": "⁻", "'": "′", "''": "″"}
_TAG = {"+": "plus", "-": "minus", "'": "prime", "''": "doubleprime"}
_UNTAG = {value: key for key, value in _TAG.items()}
_LETTERS = r"A-Za-zΑ-Ωα-ω"
_BASE = rf"[{_LETTERS}]+[0-9]*"
_SUBCHARS = re.escape("".join(_UNSUB))
_INDEX = rf"[{_LETTERS}0-9]+"
_INDEX_GROUP = rf"\({_INDEX}\)"
_INDEX_TEXT = rf"\{{{_INDEX}\}}|{_INDEX_GROUP}|{_INDEX}"
_SCRIPT = re.compile(
    rf"(?<![{_LETTERS}])(?P<base>{_BASE})(?P<scripts>(?:"
    rf"_(?:{_INDEX_TEXT})"
    rf"|[{_SUBCHARS}]+"
    r"|\^(?:\{(?:''|'|[+-])\}|''|'|[+-](?![0-9.]))"
    r"|[⁺⁻](?![0-9⁰¹²³⁴⁵⁶⁷⁸⁹])|[′″]|''|'){1,2})"
)
_PART = re.compile(
    rf"_({_INDEX_TEXT})"
    rf"|([{_SUBCHARS}]+)|\^(\{{(?:''|'|[+-])\}}|''|'|[+-])|([⁺⁻′″]|''|')"
)
ENCODED_PARAMETER_PATTERN = rf"[{_LETTERS}_][{_LETTERS}_0-9]*?__ode_up_(?:doubleprime|prime|plus|minus)__"
_ENCODED = re.compile(r"^(.*)__ode_up_(plus|minus|prime|doubleprime)__$")
_DISPLAY_NAMES = re.compile(ENCODED_PARAMETER_PATTERN + rf"|[{_LETTERS}_][{_LETTERS}_0-9]*")
_SOURCE_NAMES = re.compile(_SCRIPT.pattern + "|" + _DISPLAY_NAMES.pattern)
_EXPLICIT_INDEX = re.compile(rf"_(?:{_INDEX_GROUP}|[0-9]+)")
EDITABLE_SUBSCRIPT = re.compile(
    rf"(?<![{_LETTERS}]){_BASE}(?:_(?:\((?P<group>{_INDEX})\)?|(?P<plain>{_INDEX}))"
    rf"|(?P<unicode>[{_SUBCHARS}]+))")
_INCOMPLETE_GROUP = re.compile(rf"_\((?!{_INDEX}\))")


def has_incomplete_subscript(text):
    return bool(_INCOMPLETE_GROUP.search(text))


def subscript_exit_position(text, cursor):
    for match in EDITABLE_SUBSCRIPT.finditer(text):
        start = match.start(match.lastgroup)
        end = match.end()
        if start <= cursor <= end and (cursor < end or not match[0].endswith(")")):
            return end
    return None


def canonicalize_parameter_notation(text):
    text = text.replace("′'", "″").replace("−", "-")
    # Repair previously exposed internal names, including a following factor.
    text = re.sub(r"__ode_up_(doubleprime|prime|plus|minus)(?!__)",
                  lambda match: match[0] + "__", text)
    def convert(match):
        base = match["base"]
        sub, sup = None, None
        grouped = False
        for part in _PART.finditer(match["scripts"]):
            if part[1] is not None:
                if sub is not None:
                    return match[0]
                grouped = part[1].startswith("(")
                sub = part[1].strip("{}()")
            elif part[2] is not None:
                if sub is not None:
                    return match[0]
                sub = "".join(_UNSUB[c] for c in part[2])
            else:
                if sup is not None:
                    return match[0]
                sup = (part[3].strip("{}") if part[3] else part[4]
                       if part[4] in {"'", "''"} else {v: k for k, v in _UP.items()}[part[4]])
        # Keep numeric powers distinct from symbolic superscript labels.
        if base in {"x", "y", "z", "r", "theta", "θ", "pi", "e"} and sub is None:
            return match[0]
        if sub is not None:
            base += sub if sub.isdecimal() else "_" + sub
        name = base + ("__ode_up_" + _TAG[sup] + "__" if sup is not None else "")
        if grouped and re.match(rf"[{_LETTERS}0-9_]", text[match.end():match.end()+1]):
            return name + "*"
        return name
    return _SCRIPT.sub(convert, text)


def display_parameter_name(name, reserved=()):
    if name in reserved:
        return name
    match = _ENCODED.fullmatch(name)
    pending = ""
    if not match and name.endswith("_"):
        match = _ENCODED.fullmatch(name[:-1])
        pending = "_" if match else ""
    if match:
        stem, sup = match[1], _UP[_UNTAG[match[2]]]
    else:
        stem, sup = name, ""
    sub = re.fullmatch(rf"([{_LETTERS}]+)_([{_LETTERS}0-9]+)", stem)
    if not sub:
        sub = re.fullmatch(rf"([{_LETTERS}]+)([0-9]+)", stem)
        if sub and sub[1] in {"x", "y", "z", "r", "θ", "theta"}:
            sub = None
    if sub:
        base, index = sub.groups()
        stem = base + ("".join(_SUB[c] for c in index) if index.isdecimal() else "_" + index)
    return stem + sup + pending


def display_parameter_expression(text, reserved=(), original=""):
    """Decode each bounded parameter token, leaving adjacent factors separate."""
    # Retain explicitly typed delimiters and underscores in editable formulas.
    spellings = defaultdict(deque)
    for match in _SOURCE_NAMES.finditer(original):
        name = canonicalize_parameter_notation(match[0])
        explicit = match["scripts"] and _EXPLICIT_INDEX.fullmatch(match["scripts"])
        spellings[name].append(match[0] if explicit else None)

    def display(match):
        name = match[0]
        spelling = spellings[name].popleft() if spellings[name] else None
        return spelling or display_parameter_name(name, reserved)

    return _DISPLAY_NAMES.sub(display, text)


def format_parameter_definitions(text, reserved=()):
    items = []
    for item in text.split(","):
        name, separator, value = item.partition("=")
        raw = name.strip()
        normalized = canonicalize_parameter_notation(raw)
        formatted = display_parameter_expression(normalized, reserved, raw) if normalized.isidentifier() else raw
        leading = name[:len(name)-len(name.lstrip())]
        trailing = name[len(name.rstrip()):] if name.rstrip() != name else ""
        items.append(leading + formatted + trailing + separator + value)
    return ",".join(items)


def infer_parameter_names(expressions, explicit, reserved, include_unused=True):
    """Single adjacent letters multiply; declared multiletter names stay intact."""
    names = set(explicit) if include_unused else set()
    candidates = sorted(set(explicit) | set(reserved), key=len, reverse=True)
    for expression in expressions:
        source = canonicalize_parameter_notation(expression)
        # A notation token is indivisible, including letters in a subscript.
        tokens = re.finditer(
            rf"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
            rf"|(?P<name>{ENCODED_PARAMETER_PATTERN}|[{_LETTERS}_][{_LETTERS}_0-9]*)", source)
        for token in tokens:
            word = token["name"]
            if word is None:
                continue
            if word in reserved:
                continue
            if word in explicit:
                names.add(word)
                continue
            if "_" in word:
                names.add(word)
                continue
            index = 0
            while index < len(word):
                indexed = re.match(rf"([{_LETTERS}])([0-9]+)", word[index:])
                if indexed and indexed[1] not in {"x", "y", "z", "r", "θ"}:
                    names.add(indexed[0])
                    index += len(indexed[0])
                    continue
                atom = next((candidate for candidate in candidates if word.startswith(candidate, index)), None)
                if atom:
                    if atom in explicit:
                        names.add(atom)
                    index += len(atom)
                    continue
                token = re.match(rf"[{_LETTERS}][0-9]*", word[index:])
                if token:
                    names.add(token[0])
                    index += len(token[0])
                else:
                    index += 1
    return names
