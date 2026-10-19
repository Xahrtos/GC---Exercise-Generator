"""Esercizi parametrici.

Un esercizio parametrico e' descritto da:
  * random_params : parametri generati a caso, con un intervallo diverso per ogni livello
                    di difficolta' (easy / medium / hard);
  * formulas      : parametri derivati, calcolati in ordine con una formula (expr) e con una
                    versione testuale (text) riutilizzabile nei testi ({formula_<nome>});
  * constraints   : condizioni da rispettare (se non valgono si rigenera);
  * problem_template, intermediate_steps, correct_solution : testi con segnaposto {nome}.

Tutti i calcoli usano Fraction, quindi radici frazionarie e radici quadrate esatte non
hanno errori di arrotondamento. Le formule sono valutate con un interprete AST ristretto
(niente eval).
"""
import ast
import json
import math
import random
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional

LEVELS = ("easy", "medium", "hard")
MAX_ATTEMPTS = 2000


class ExpressionError(Exception):
    """Errore nella valutazione di una formula o di un vincolo."""


# ---------------------------------------------------------------------------
# Numeri
# ---------------------------------------------------------------------------
def to_fraction(value: Any) -> Optional[Fraction]:
    """Converte numeri e stringhe ('3', '-1/2', '0,5', '−4') in Fraction; None se impossibile."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    text = str(value).strip().replace("−", "-").replace(",", ".").replace(" ", "")
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def fmt(val: Any) -> str:
    """Formatta un numero: interi senza decimali, frazioni come 'p/q'."""
    if isinstance(val, bool):
        return str(int(val))
    if isinstance(val, Fraction):
        if val.denominator == 1:
            return str(val.numerator)
        return f"{val.numerator}/{val.denominator}"
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    return str(val)


def normalize_level(difficulty: str) -> str:
    d = (difficulty or "easy").strip().lower()
    if d in ("medio", "media", "medium", "intermedio"):
        return "medium"
    if d in ("difficile", "hard", "avanzato"):
        return "hard"
    return "easy"


# ---------------------------------------------------------------------------
# Valutazione sicura di formule e vincoli
# ---------------------------------------------------------------------------
def _exact_sqrt(x: Fraction) -> Fraction:
    if x < 0:
        raise ExpressionError("radice di un numero negativo")
    n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
    if n * n != x.numerator or d * d != x.denominator:
        raise ExpressionError(f"radice non esatta di {fmt(x)}")
    return Fraction(n, d)


def _as_int(x: Fraction) -> int:
    if x.denominator != 1:
        raise ExpressionError("serve un numero intero")
    return x.numerator


def _lcm(*args: Fraction) -> Fraction:
    out = 1
    for a in args:
        v = abs(_as_int(a))
        out = out * v // math.gcd(out, v) if v else 0
    return Fraction(out)


_FUNCS = {
    "abs": lambda x: abs(x),
    "min": lambda *a: min(a),
    "max": lambda *a: max(a),
    "sqrt": _exact_sqrt,
    "num": lambda x: Fraction(x.numerator),
    "den": lambda x: Fraction(x.denominator),
    "gcd": lambda *a: Fraction(math.gcd(*[_as_int(x) for x in a])),
    "lcm": _lcm,
}


def _div(a: Fraction, b: Fraction) -> Fraction:
    if b == 0:
        raise ExpressionError("divisione per zero")
    return a / b


def _mod(a: Fraction, b: Fraction) -> Fraction:
    if b == 0:
        raise ExpressionError("modulo zero")
    return a % b


def _pow(a: Fraction, b: Fraction) -> Fraction:
    if b.denominator != 1 or abs(b.numerator) > 12:
        raise ExpressionError("esponente non ammesso")
    if a == 0 and b < 0:
        raise ExpressionError("divisione per zero")
    return a ** b.numerator


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _div,
    ast.Mod: _mod,
    ast.Pow: _pow,
}
_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _ev(node: ast.AST, ctx: Dict[str, Fraction]) -> Any:
    if isinstance(node, ast.Expression):
        return _ev(node.body, ctx)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        raise ExpressionError("costante non ammessa")
    if isinstance(node, ast.Name):
        if node.id not in ctx:
            raise ExpressionError(f"variabile sconosciuta: {node.id}")
        return ctx[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, ctx)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return not v
        raise ExpressionError("operatore unario non ammesso")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError("operatore non ammesso")
        return op(_ev(node.left, ctx), _ev(node.right, ctx))
    if isinstance(node, ast.Compare):
        left = _ev(node.left, ctx)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, ctx)
            fn = _CMPOPS.get(type(op))
            if fn is None or not fn(left, right):
                if fn is None:
                    raise ExpressionError("confronto non ammesso")
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, ctx) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise ExpressionError("funzione non ammessa")
        return _FUNCS[node.func.id](*[_ev(a, ctx) for a in node.args])
    raise ExpressionError("espressione non ammessa")


def evaluate_expression(expr: Any, context: Dict[str, Any]) -> Any:
    """Valuta una formula (es. '-a*(x1+x2)') con i valori in context; ritorna Fraction (o bool)."""
    ctx = {}
    for k, v in context.items():
        f = to_fraction(v)
        if f is not None:
            ctx[k] = f
    try:
        tree = ast.parse(str(expr).strip().replace("^", "**"), mode="eval")
        return _ev(tree, ctx)
    except ExpressionError:
        raise
    except SyntaxError as e:
        raise ExpressionError(f"formula non valida '{expr}': {e}")
    except (ValueError, ZeroDivisionError, TypeError, OverflowError) as e:
        raise ExpressionError(f"impossibile valutare '{expr}': {e}")


# ---------------------------------------------------------------------------
# Testi con segnaposto
# ---------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r"\{(\w+)(?::([^{}]+))?\}")


def _format_value(v: Any, spec: str) -> str:
    f = to_fraction(v)
    if f is None:
        return fmt(v)
    neg = f < 0
    mag = fmt(abs(f))
    if spec == "abs":
        return mag
    if spec == "paren":
        return f"({fmt(f)})" if neg else fmt(f)
    if spec == "signed":
        if f == 0:
            return ""
        return f" - {mag}" if neg else f" + {mag}"
    if spec.startswith("lead_"):          # primo termine: 'x²', '-x²', '3x²'
        suf = spec[5:]
        if f == 1:
            return suf
        if f == -1:
            return "-" + suf
        return fmt(f) + suf
    if spec.startswith("term_"):          # termine successivo: ' + 3x', ' - x', '' se zero
        suf = spec[5:]
        if f == 0:
            return ""
        sign = " - " if neg else " + "
        return sign + ("" if abs(f) == 1 else mag) + suf
    return fmt(f)


def render(template: str, context: Dict[str, Any]) -> str:
    """Sostituisce {nome} e {nome:formato} nel testo. Formati: signed, abs, paren, lead_<s>, term_<s>.
    I segnaposto sconosciuti restano invariati."""
    if not template:
        return ""

    def sub(m: "re.Match[str]") -> str:
        name, spec = m.group(1), m.group(2)
        if name not in context:
            return m.group(0)
        v = context[name]
        return _format_value(v, spec) if spec else fmt(v)

    return _PLACEHOLDER.sub(sub, str(template))


# ---------------------------------------------------------------------------
# Generazione
# ---------------------------------------------------------------------------
def _load_json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value is not None else default


def _spec_for_level(spec: Any, level: str) -> Any:
    """Se lo spec ha una variante per livello la sceglie (con ripiego sul livello piu' vicino)."""
    if isinstance(spec, dict) and any(l in spec for l in LEVELS):
        order = list(reversed(LEVELS[:LEVELS.index(level) + 1])) + list(LEVELS)
        for l in order:
            if l in spec:
                return spec[l]
    return spec


def _draw(spec: Any) -> Fraction:
    """Estrae un valore: costante, [min, max], {min,max,denominators,values,exclude}."""
    if isinstance(spec, (int, float, str)):
        f = to_fraction(spec)
        if f is None:
            raise ExpressionError(f"valore non valido: {spec}")
        return f
    if isinstance(spec, list):
        spec = {"min": spec[0], "max": spec[1]} if len(spec) >= 2 else {"values": spec}
    if not isinstance(spec, dict):
        raise ExpressionError(f"specifica non valida: {spec}")
    exclude = [to_fraction(e) for e in spec.get("exclude", [])]
    for _ in range(200):
        if "values" in spec:
            v = to_fraction(random.choice(spec["values"]))
        else:
            lo, hi = to_fraction(spec.get("min", 0)), to_fraction(spec.get("max", 10))
            den = int(random.choice(spec.get("denominators", [1])))
            n = random.randint(math.ceil(lo * den), math.floor(hi * den))
            v = Fraction(n, den)
        if v is not None and v not in exclude:
            return v
    raise ExpressionError("impossibile estrarre un valore valido")


def _constraints_for(constraints: Any, level: str) -> List[str]:
    """Accetta una lista (valida per tutti) o un dict {'all': [...], 'easy': [...], ...}."""
    if isinstance(constraints, list):
        return [str(c) for c in constraints]
    if isinstance(constraints, dict):
        out = [str(c) for c in constraints.get("all", [])]
        out += [str(c) for c in constraints.get(level, [])]
        return out
    return []


def _formulas_of(ex: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Normalizza 'formulas' (e il vecchio 'derived_params') in {nome: {expr, text, text_values}}."""
    raw: Dict[str, Any] = {}
    raw.update(_load_json(ex.get("derived_params"), {}) or {})
    raw.update(_load_json(ex.get("formulas"), {}) or {})
    out: Dict[str, Dict[str, str]] = {}
    for name, f in raw.items():
        if isinstance(f, dict):
            out[name] = {"expr": str(f.get("expr", "")), "text": str(f.get("text", "")),
                         "text_values": str(f.get("text_values", ""))}
        else:
            out[name] = {"expr": str(f), "text": "", "text_values": ""}
    return out


def _generate_values(ex: Dict[str, Any], level: str) -> Dict[str, Fraction]:
    """Estrae i parametri casuali, calcola i derivati e verifica i vincoli (rigenera finche' validi)."""
    random_params = _load_json(ex.get("random_params"), {}) or {}
    formulas = _formulas_of(ex)
    constraints = _constraints_for(_load_json(ex.get("constraints"), []), level)
    last_error = "vincoli non soddisfatti"
    for _ in range(MAX_ATTEMPTS):
        try:
            ctx: Dict[str, Fraction] = {}
            deferred: Dict[str, Any] = {}
            for name, spec in random_params.items():
                s = _spec_for_level(spec, level)
                if isinstance(s, dict) and "times" in s:
                    deferred[name] = s       # es. a moltiplicato per i denominatori delle radici
                else:
                    ctx[name] = _draw(s)
            for name, s in deferred.items():
                ctx[name] = _draw(s) * evaluate_expression(s["times"], ctx)
            for name, f in formulas.items():
                ctx[name] = evaluate_expression(f["expr"], ctx)
            if all(evaluate_expression(c, ctx) for c in constraints):
                return ctx
        except ExpressionError as e:
            last_error = str(e)
    raise ExpressionError(f"nessun valore valido dopo {MAX_ATTEMPTS} tentativi ({last_error})")


def materialize(record: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
    """Genera una istanza concreta dell'esercizio per il livello richiesto."""
    level = normalize_level(difficulty)
    ex = dict(record)
    values = _generate_values(ex, level)
    formulas = _formulas_of(ex)

    context: Dict[str, Any] = dict(values)
    for name, f in formulas.items():          # testi delle formule: {formula_<nome>}
        context[f"formula_{name}"] = render(f["text"], context) if f["text"] else name
    for name, f in formulas.items():          # formule con i valori sostituiti
        if f["text_values"]:
            context[f"formula_{name}_values"] = render(f["text_values"], context)

    ex["parameters"] = {k: fmt(v) for k, v in values.items()}
    ex["difficulty"] = level
    ex["problem_statement"] = render(str(ex.get("problem_template") or ex.get("problem_statement") or ""), context)
    ex["correct_solution"] = render(str(ex.get("correct_solution", "")), context)

    steps = []
    for st in _load_json(ex.get("intermediate_steps"), []) or []:
        if not isinstance(st, dict):
            continue
        exprs = st.get("values") if isinstance(st.get("values"), list) else [st.get("value", "")]
        expected = [fmt(evaluate_expression(e, values)) for e in exprs if str(e).strip()]
        hints_tpl = st.get("hints") if isinstance(st.get("hints"), list) else ([st["hint"]] if st.get("hint") else [])
        hints = [render(str(h), context) for h in hints_tpl] or ["Ricontrolla i calcoli."]
        steps.append({
            "description": render(str(st.get("description", "")), context),
            "value": expected[0] if expected else "",
            "values": expected,
            "unordered": bool(st.get("unordered", False)),
            "hint": hints[0],
            "hints": hints,
        })
    ex["intermediate_steps"] = steps
    ex["mode"] = f"PARAMETRIC_{level.upper()}"
    return ex


# ---------------------------------------------------------------------------
# Confronto e validazione
# ---------------------------------------------------------------------------
def numeric_match(student: str, expected_values: List[str], tol: float, unordered: bool) -> bool:
    """True se il valore dello studente (intero, decimale o frazione) coincide con uno atteso."""
    s = to_fraction(student)
    if s is None:
        return False
    for exp in expected_values:
        e = to_fraction(exp)
        if e is not None and abs(float(s - e)) <= max(tol * abs(float(e)), 1e-9):
            return True
    return False


def validate_template(item: Dict[str, Any]) -> List[str]:
    """Controlla un esercizio: campi obbligatori e, se parametrico, che si generi a tutti i livelli."""
    errors: List[str] = []
    if not item.get("exercise_id"):
        errors.append("Manca exercise_id")
    if not item.get("correct_solution"):
        errors.append("Manca correct_solution")
    if str(item.get("type", "")).lower() == "parametric":
        for level in LEVELS:
            try:
                for _ in range(30):
                    ex = materialize(item, level)
                    texts = [ex["problem_statement"], ex["correct_solution"]]
                    for st in ex["intermediate_steps"]:
                        texts += [st["description"]] + st["hints"]
                        if not st["value"]:
                            errors.append(f"[{level}] passaggio senza valore")
                    if any("{" in t for t in texts):
                        errors.append(f"[{level}] segnaposto non risolto")
                        break
            except ExpressionError as e:
                errors.append(f"[{level}] {e}")
    return errors
