import inspect
import json
import os
import random
import re
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from fastapi import UploadFile
from starlette.responses import StreamingResponse

from cat import (
    AuthPermission,
    AuthResource,
    AuthorizedInfo,
    StrayCat,
    check_permissions,
    endpoint,
    hook,
    tool,
)
from cat.db.cruds.settings import crud as crud_settings

try:
    from cat import log
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("exercise_generator")

from .constants import FILENAME_EXERCISES, FILENAME_TEMPLATE, PLUGIN_DIR, PLUGIN_ID
from .models import ExerciseDataManager, ExerciseTemplate
from .parametric import (
    ExpressionError,
    fmt,
    materialize,
    normalize_level,
    numeric_match,
    render,
    validate_template,
)
from .core import check_answer


_NESTED_KEYS = (
    "difficulty", "difficulty_levels", "random_params", "derived_params", "formulas", "formula_texts", "constraints",
    "intermediate_steps_template", "intermediate_steps", "parameters", "solution_exprs",
)


def _get_file_path(cat: StrayCat) -> str:
    return os.path.join(cat.agent_key, "plugins", PLUGIN_ID)


def _get_db_key(cat: StrayCat) -> str:
    return f"plugins:{PLUGIN_ID}:{cat.agent_key}:{cat.id}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return fmt(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    return obj


async def _store_active_exercise(cat: StrayCat, exercise: Dict[str, Any]) -> None:
    await crud_settings.store(_get_db_key(cat), _jsonable(exercise))


def _unwrap_state(raw: Any) -> Dict[str, Any] | None:
    """Normalizza quanto restituito da crud_settings.read in un dict dell'esercizio.

    Accetta: dict, wrapper {"value": ...}, lista di dict, lista di coppie (chiave, valore).
    """
    for _ in range(6):
        if isinstance(raw, list):
            if raw and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in raw):
                raw = {k: v for k, v in raw}
                continue
            dicts = [x for x in raw if isinstance(x, (dict, list))]
            raw = dicts[-1] if dicts else None
        elif isinstance(raw, dict) and "intermediate_steps" not in raw and isinstance(raw.get("value"), (dict, list)):
            raw = raw["value"]
        else:
            break
    return raw if isinstance(raw, dict) else None


async def _get_current_exercise(cat: StrayCat) -> Dict[str, Any]:
    raw = await crud_settings.read(_get_db_key(cat))
    ex = _unwrap_state(raw)
    if not ex:
        if raw:
            log.warning(f"exercise_generator: stato esercizio non riconosciuto ({type(raw)}): {str(raw)[:300]}")
        raise ValueError("Nessun esercizio attivo. Avviane uno con start_exercise.")
    return ex


def _load_dataset(cat: StrayCat):
    """Legge exercises.json dal file manager del Cat (data/storage/<agente>/plugins/<plugin>/),
    con ripiego sulla cartella del plugin. Non usa la funzione open vietata."""
    content = cat.file_manager.read_file(FILENAME_EXERCISES, _get_file_path(cat))
    if not content:
        local = Path(PLUGIN_DIR) / FILENAME_EXERCISES
        if local.exists():
            content = local.read_bytes()
    if content:
        if isinstance(content, str):
            content = content.encode("utf-8")
        return ExerciseDataManager.load_data(content)
    return None


async def _get_tolerance(cat: StrayCat) -> float:
    try:
        settings = cat.mad_hatter.get_plugin().load_settings()
        if inspect.isawaitable(settings):
            settings = await settings
        if isinstance(settings, dict):
            return float(settings.get("tolerance", 0.01))
    except Exception:
        pass
    return 0.01


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items() if x is not None and not (isinstance(x, float) and x != x)}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist"):
        return _clean(v.tolist())
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:
            pass
    return v


def _as_step_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = ExerciseDataManager.parse_steps(raw)
    return [s for s in (raw or []) if isinstance(s, dict)]


def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec = _clean(dict(rec))
    for key in _NESTED_KEYS:
        val = rec.get(key)
        if isinstance(val, str) and val.strip()[:1] in ("{", "["):
            try:
                rec[key] = json.loads(val)
            except Exception:
                pass
    return rec


def _prepare_static(rec: Dict[str, Any]) -> Dict[str, Any]:
    ex = dict(rec)
    params = ex.get("parameters") or {}
    display = {k: fmt(v) for k, v in params.items()} if isinstance(params, dict) else {}

    tpl = str(ex.get("problem_template") or "")
    ex["problem_statement"] = ex.get("problem_statement") or (render(tpl, display) if tpl else "")
    ex["correct_solution"] = render(str(ex.get("correct_solution", "")), display)

    steps = []
    for st in _as_step_list(ex.get("intermediate_steps")):
        value = render(str(st.get("value", "")), display).strip()
        hint = render(str(st.get("hint") or ""), display).strip()
        steps.append({
            "description": render(str(st.get("description", "")), display),
            "value": value,
            "values": [value] if value else [],
            "unordered": bool(st.get("unordered", False)),
            "hint": hint or "Ricontrolla i calcoli.",
            "hints": [hint] if hint else [],
        })
    ex["intermediate_steps"] = steps
    ex["mode"] = "EXACT"
    return ex


def _new_progress() -> Dict[str, int]:
    return {"current_step": 1, "hint_level": 0, "wrong_attempts": 0}


async def _activate(cat: StrayCat, record: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
    rec = _normalize_record(record)
    if str(rec.get("type", "static")).lower() == "parametric":
        ex = materialize(rec, difficulty)
    else:
        ex = _prepare_static(rec)
    ex["progress"] = _new_progress()
    await _store_active_exercise(cat, ex)
    if not _unwrap_state(await crud_settings.read(_get_db_key(cat))):
        raise RuntimeError("Esercizio generato ma non salvato nello stato attivo.")
    return ex


def _intro(ex: Dict[str, Any], origin: str) -> str:
    return (
        f"--- ESERCIZIO ATTIVATO ({origin}) ---\n"
        f"{ex.get('problem_statement', '')}\n\n"
        f"Istruzioni per te: presenta il testo dello studente e invitalo a iniziare dal passaggio 1 "
        f"({len(ex.get('intermediate_steps', []))} passaggi in tutto). "
        f"NON rivelare valori intermedi né la soluzione."
    )


async def _start_from_dataset(cat: StrayCat, topic: str, difficulty: str) -> str:
    df = _load_dataset(cat)
    if df is None or df.empty:
        return f"Errore: dataset {FILENAME_EXERCISES} non trovato o vuoto."

    topics = sorted({str(t) for t in df["topic"].to_list()})
    if topic:
        wanted = topic.strip().lower().replace(" ", "_")
        col = df["topic"].astype(str).str.lower().str.replace(" ", "_")
        filtered = df[col == wanted]
        if filtered.empty:
            # corrispondenza parziale (es. "equazioni" -> "equazioni_secondo_grado")
            filtered = df[col.apply(lambda t: wanted in t or t in wanted)]
        if filtered.empty and len(topics) == 1:
            # un solo argomento disponibile: usalo
            filtered = df
        if filtered.empty:
            return f"Nessun esercizio per il topic '{topic}'. Argomenti disponibili: {', '.join(topics)}."
        df = filtered

    # preferisci il livello richiesto, se presente nel dataset
    if "difficulty" in df.columns:
        by_level = df[df["difficulty"].astype(str).str.lower() == str(difficulty).lower()]
        if not by_level.empty:
            df = by_level

    ex_id = str(df.sample(1).iloc[0]["exercise_id"])
    record = ExerciseDataManager.get_exercise_by_id(df, ex_id)
    ex = await _activate(cat, record, difficulty)
    return _intro(ex, f"dataset {ex_id}")


async def _start_by_id(cat: StrayCat, exercise_id: str, difficulty: str) -> str:
    df = _load_dataset(cat)
    if df is None:
        return f"Errore: dataset {FILENAME_EXERCISES} non trovato."
    clean_id = str(exercise_id).strip().upper()
    try:
        record = ExerciseDataManager.get_exercise_by_id(df, clean_id)
        if record is None:
            record = ExerciseDataManager.get_exercise_by_id(df, str(exercise_id).strip())
    except Exception:
        record = None
    if record is None:
        return f"Impossibile trovare l'esercizio '{clean_id}' nel dataset."
    ex = await _activate(cat, record, difficulty)
    return _intro(ex, f"dataset {clean_id}")


def _validate_generated(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("JSON non è un oggetto")
    if not str(data.get("problem_statement", "")).strip():
        raise ValueError("manca problem_statement")
    if not str(data.get("correct_solution", "")).strip():
        raise ValueError("manca correct_solution")
    steps = _as_step_list(data.get("intermediate_steps"))
    if not steps:
        raise ValueError("mancano intermediate_steps")
    norm = []
    for st in steps:
        value = str(st.get("value", "")).strip()
        hint = str(st.get("hint") or "").strip()
        if not value:
            raise ValueError("uno step non ha 'value'")
        norm.append({
            "description": str(st.get("description", "")),
            "value": value, "values": [value], "unordered": False,
            "hint": hint or "Ricontrolla i calcoli.", "hints": [hint] if hint else [],
        })
    data["intermediate_steps"] = norm
    return data


async def _start_ai_generated(cat: StrayCat, topic: str) -> str:
    df = _load_dataset(cat)
    topics: List[str] = []
    if df is not None and not df.empty:
        topics = sorted({str(t) for t in df["topic"].dropna().to_list()})
    if topic and topics and topic.lower() not in [t.lower() for t in topics]:
        return f"Topic '{topic}' non presente. Topic disponibili: {', '.join(topics)}"
    if not topic and topics:
        topic = random.choice(topics)

    sample = ""
    if df is not None and not df.empty:
        sample = f"Esempio di stile: {df.sample(1).iloc[0].get('problem_template')}"

    prompt = f"""Sei un docente di matematica. Crea UN esercizio originale.
Argomento: {topic or 'matematica'}
{sample}

Rispondi SOLO con un oggetto JSON valido, senza testo prima o dopo:
{{"exercise_id": "AI_GEN", "topic": "{topic or 'matematica'}",
  "problem_statement": "testo con dati numerici",
  "correct_solution": "risultato finale",
  "intermediate_steps": [
    {{"description": "passaggio 1", "value": "risultato numerico 1", "hint": "suggerimento 1"}},
    {{"description": "passaggio 2", "value": "risultato numerico 2", "hint": "suggerimento 2"}}
  ]}}"""
    try:
        raw = cat.llm(prompt)
        match = re.search(r"\{[\s\S]*\}", str(raw).replace("```json", "").replace("```", ""))
        if not match:
            raise ValueError("nessun JSON nella risposta del modello")
        data = _validate_generated(json.loads(match.group(0)))
        data["mode"] = "GENERATED"
        data["verified"] = False
        data["progress"] = _new_progress()
        await _store_active_exercise(cat, data)
        return _intro(data, "generato dall'AI, non verificato")
    except Exception as e:
        log.warning(f"generate_ai_exercise fallita: {e}")
        return f"Errore nella generazione con l'LLM ({e})."


_SAFE_INPUT = re.compile(r"[\w\s+\-*/^().,;=]*")


def _symbolic_ok(expected: str, student: str) -> bool:
    if not expected:
        return False
    for text in (student, expected):
        if not _SAFE_INPUT.fullmatch(text) or "__" in text or re.search(r"\.[A-Za-z_]", text):
            return False
    try:
        res = check_answer(problem=expected, user_answer=student)
        return bool(res.get("ok") and res.get("result", {}).get("ok_answer"))
    except Exception as e:
        log.warning(f"check_answer fallita: {e}")
        return False


def _answer_ok(student: str, expected_values: List[str], expected_text: str,
               tol: float, unordered: bool) -> bool:
    student = str(student).strip()
    if not student:
        return False
    res = numeric_match(student, expected_values, tol, unordered)
    if res:
        return True
    if student.lower() == expected_text.strip().lower():
        return True
    return _symbolic_ok(expected_text or ", ".join(expected_values), student)


def _steps_of(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _as_step_list(ex.get("intermediate_steps"))


# ---------------------------------------------------------------------------
# Ultimo messaggio dello studente: i modelli piccoli sbagliano gli argomenti dei tool
# (es. passano 36 quando lo studente ha scritto 6). Per questo si legge il valore
# direttamente dal testo originale dello studente.
# ---------------------------------------------------------------------------
_LAST_USER_TEXT: Dict[str, str] = {}
_NUM_TOKEN = re.compile(r"-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?")


@hook(priority=10)
async def before_cat_reads_message(user_message_json, cat: StrayCat):
    try:
        if isinstance(user_message_json, dict):
            text = user_message_json.get("text")
        else:
            text = getattr(user_message_json, "text", None)
        _LAST_USER_TEXT[str(cat.id)] = str(text or "")
    except Exception as e:
        log.warning(f"exercise_generator: impossibile leggere il messaggio dello studente: {e}")
    return user_message_json


def _student_numbers(text: str) -> List[Fraction]:
    """Numeri scritti dallo studente, ignorando 'passaggio N' e le etichette x1, x2."""
    t = str(text or "").replace("\u2212", "-")
    t = re.sub(r"\bpassagg\w*\s*(?:n\.?\s*|numero\s*)?\d+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bx\s*_?\s*\d\b", " ", t, flags=re.IGNORECASE)
    out: List[Fraction] = []
    for m in _NUM_TOKEN.finditer(t):
        try:
            out.append(Fraction(m.group(0).replace(",", ".").replace(" ", "")))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def _raw_user_text(cat: StrayCat) -> str:
    return _LAST_USER_TEXT.get(str(cat.id), "")


@tool(examples=["Quali argomenti ci sono?", "Elenca gli argomenti degli esercizi"])
def list_topics(cat: StrayCat) -> str:
    """Elenca gli argomenti (topic) degli esercizi disponibili nel dataset."""
    df = _load_dataset(cat)
    if df is None or df.empty:
        return "Nessun argomento trovato o dataset vuoto."
    topics = sorted({str(t) for t in df["topic"].to_list()})
    return f"Argomenti disponibili: {', '.join(topics)}"


@tool(examples=[
    "Dammi un esercizio sulle equazioni",
    "Voglio un esercizio facile",
    "Carica l'esercizio MATH_QE_001",
    "Inventa un esercizio nuovo"
])
async def start_exercise(mode: str = "dataset", topic: str = "", difficulty: str = "easy",
                         exercise_id: str = "", cat: StrayCat = None) -> str:
    """Avvia un nuovo esercizio per lo studente. Argomenti: mode ('dataset' di default, 'by_id', 'generated');
    topic = nome esatto di un argomento restituito da list_topics (es. equazioni_secondo_grado), vuoto per qualsiasi;
    difficulty = easy, medium o hard."""
    try:
        m = (mode or "dataset").strip().lower()
        level = normalize_level(difficulty)
        if m == "by_id":
            if not exercise_id:
                return "Errore: exercise_id mancante."
            return await _start_by_id(cat, exercise_id, level)
        if m == "generated":
            return await _start_ai_generated(cat, topic)
        return await _start_from_dataset(cat, topic, level)
    except Exception as e:
        return f"Errore durante l'avvio dell'esercizio: {e}"


async def _do_show(cat: StrayCat) -> str:
    try:
        ex = await _get_current_exercise(cat)
    except ValueError as e:
        return str(e)
    steps = _steps_of(ex)
    cur = int((ex.get("progress") or {}).get("current_step", 1))
    nxt = steps[cur - 1]["description"] if 0 < cur <= len(steps) else "(tutti i passaggi completati)"
    return f"Testo: {ex.get('problem_statement', '')}\nPassaggio corrente: {min(cur, len(steps))}/{len(steps)} - {nxt}"


async def _do_hint(cat: StrayCat) -> str:
    try:
        ex = await _get_current_exercise(cat)
    except ValueError as e:
        return str(e)
    steps, prog = _steps_of(ex), ex.setdefault("progress", _new_progress())
    cur = int(prog.get("current_step", 1))
    if cur > len(steps):
        return "Tutti i passaggi sono completati: chiedi la soluzione finale allo studente."
    step = steps[cur - 1]
    hints = step.get("hints") or [step.get("hint") or "Ricontrolla i calcoli."]
    level = int(prog.get("hint_level", 0))
    hint = hints[min(level, len(hints) - 1)]
    prog["hint_level"] = level + 1
    await _store_active_exercise(cat, ex)
    return f"Suggerimento per il passaggio {cur} ({level + 1}/{len(hints)}): {hint}"


async def _do_verify(cat: StrayCat, step_number: int, student_value: str) -> str:
    try:
        ex = await _get_current_exercise(cat)
        tol = await _get_tolerance(cat)
        steps, prog = _steps_of(ex), ex.setdefault("progress", _new_progress())
        cur = int(prog.get("current_step", 1))
        if cur > len(steps):
            return "Tutti i passaggi sono completati: chiedi la soluzione finale allo studente."

        # il valore si legge dal testo originale dello studente, non dall'argomento scelto dal modello
        nums = _student_numbers(_raw_user_text(cat))
        if nums:
            value = fmt(nums[-1])
            if str(student_value).strip() != value:
                log.warning(f"exercise_generator: valore del modello '{student_value}' sostituito con '{value}' (testo dello studente)")
        else:
            value = str(student_value)

        # si verifica sempre il passaggio corrente (un passaggio gia' superato si puo' ricontrollare)
        try:
            n = int(step_number)
        except (TypeError, ValueError):
            n = cur
        if n < 1 or n > cur:
            n = cur

        step = steps[n - 1]
        expected_values = step.get("values") or ([step["value"]] if step.get("value") else [])
        ok = _answer_ok(value, expected_values, str(step.get("value", "")),
                        tol, bool(step.get("unordered", False)))

        if ok:
            if n >= cur:
                prog["current_step"] = n + 1
                prog["hint_level"] = 0
                prog["wrong_attempts"] = 0
            await _store_active_exercise(cat, ex)
            if n < len(steps):
                return f"Passaggio {n} CORRETTO ✓ (valore dello studente: {value}). Passa al passaggio {n + 1}: {steps[n]['description']}"
            return f"Passaggio {n} CORRETTO ✓ (valore dello studente: {value}). Tutti i passaggi completati: chiedi la soluzione finale."

        prog["wrong_attempts"] = int(prog.get("wrong_attempts", 0)) + 1
        await _store_active_exercise(cat, ex)
        hint = (step.get("hints") or [step.get("hint") or "Ricontrolla i calcoli."])[0]
        return f"Passaggio {n} NON corretto ✗ (valore dello studente: {value}). Suggerimento: {hint}"
    except Exception as e:
        return f"Errore durante la verifica: {e}"


async def _do_final(cat: StrayCat, student_solution: str) -> str:
    try:
        ex = await _get_current_exercise(cat)
        tol = await _get_tolerance(cat)
        expected_text = str(ex.get("correct_solution", "")).strip()
        expected_values = ex.get("solution_values") or ([expected_text] if expected_text else [])

        raw = _raw_user_text(cat)
        student_nums = _student_numbers(raw)
        expected_nums = _student_numbers(expected_text)
        shown = raw.strip() or student_solution

        ok = False
        if student_nums and expected_nums:
            # confronto dei valori, indipendente dall'ordine e dal formato ("x1 = 2, x2 = 4", "2 e 4", ...)
            ok = sorted(student_nums) == sorted(expected_nums)
        if not ok:
            ok = _answer_ok(student_solution, expected_values, expected_text,
                            tol, bool(ex.get("solution_unordered", True)))
        if ok:
            return f"Soluzione finale CORRETTA ✓ ({shown}). Ottimo lavoro!"
        return f"Soluzione finale NON corretta ✗ ({shown}). Invita lo studente a ricontrollare."
    except Exception as e:
        return f"Errore nel confronto: {e}"


@tool(examples=["Qual è l'esercizio attivo?", "A che punto sono?"])
async def show_active_exercise(cat: StrayCat) -> str:
    """Mostra il testo dell'esercizio attivo e il passaggio corrente."""
    return await _do_show(cat)


@tool(examples=["Dammi un suggerimento", "Non so come continuare"])
async def request_hint(cat: StrayCat) -> str:
    """Dà un suggerimento progressivo per il passaggio corrente senza rivelare la soluzione."""
    return await _do_hint(cat)


@tool(examples=["Il passaggio 1 fa 5", "Verifica il passaggio 2 con valore 16"])
async def verify_intermediate_step(step_number: int, student_value: str, cat: StrayCat) -> str:
    """Verifica il risultato che lo studente ha scritto per il passaggio corrente.
    student_value deve essere ESATTAMENTE il numero scritto dallo studente nel suo ultimo messaggio:
    non calcolarlo, non correggerlo e non usare valori visti nei suggerimenti."""
    return await _do_verify(cat, step_number, student_value)


@tool(examples=["La soluzione è x=3?", "Verifica la mia soluzione finale"])
async def compare_final_solution(student_solution: str, cat: StrayCat) -> str:
    """Verifica la soluzione finale proposta dallo studente.
    student_solution deve essere il testo ESATTO scritto dallo studente nel suo ultimo messaggio."""
    return await _do_final(cat, student_solution)


# ---------------------------------------------------------------------------
# Risposta deterministica: con un esercizio attivo, i messaggi dello studente che
# contengono una risposta numerica, una richiesta di suggerimento o di stato
# vengono gestiti QUI, senza passare dal LLM (che a volte salta i tool e inventa).
# ---------------------------------------------------------------------------
_HINT_RE = re.compile(r"\b(suggerimento|suggeriscimi|aiuto|aiutami|indizio|hint|non so|non riesco|help)\b", re.I)
_STATUS_RE = re.compile(r"(a che punto|esercizio attivo|qual e'? l'?esercizio|qual \u00e8 l'?esercizio|ripeti (il )?testo|ripeti l'?esercizio)", re.I)
_NEW_RE = re.compile(r"\b(nuovo|altro|un)\s+esercizio\b|\besercizio\s+(nuovo|diverso)\b|\bdammi un esercizio\b|\bargomenti\b", re.I)
_QUESTION_RE = re.compile(r"\b(come|perch[eé]|cosa|che cos|spiega|spiegami|quale|quando|dove)\b", re.I)
_FINAL_RE = re.compile(r"(soluzion|\bx\s*_?\s*[12]\b|radici|risultato finale)", re.I)
_STEP_RE = re.compile(r"\bpassagg\w*\s*(?:n\.?\s*|numero\s*)?(\d+)", re.I)


def _message_text(cat: StrayCat) -> str:
    """Testo del messaggio corrente. In fast_reply l'hook before_cat_reads_message non e' ancora
    stato eseguito, quindi si legge dalla working memory (mai dal valore memorizzato, che e' vecchio)."""
    wm = getattr(cat, "working_memory", None)
    for attr in ("user_message_json", "user_message", "current_message"):
        um = getattr(wm, attr, None)
        if um is None:
            continue
        txt = um.get("text") if isinstance(um, dict) else getattr(um, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
    log.warning(f"exercise_generator: messaggio corrente non trovato nella working memory (attributi: "
                f"{[a for a in dir(wm) if not a.startswith('_')][:60]})")
    return ""


_START_RE = re.compile(
    r"\b(dammi|dai|voglio|vorrei|fammi|proponi|assegna|carica|avvia|inizia|iniziamo|nuovo|altro|ancora)\b.*\besercizi\w*"
    r"|\besercizi\w*\s+(nuovo|diverso|facile|medio|difficile|easy|medium|hard)\b", re.I)
_GEN_RE = re.compile(r"\b(inventa|genera|crea)\b", re.I)
_TOPICS_RE = re.compile(r"\b(argomenti|topic|temi)\b", re.I)
_ID_RE = re.compile(r"\b([A-Z]{2,}_[A-Z]{2,}_\d+)\b")
_LEVELS = (
    (re.compile(r"\b(facile|facili|semplice|easy|base)\b", re.I), "easy"),
    (re.compile(r"\b(medio|media|intermedi\w*|medium)\b", re.I), "medium"),
    (re.compile(r"\b(difficile|difficili|avanzat\w*|hard|complesso)\b", re.I), "hard"),
)
_START_ERR = ("Errore", "Nessun", "Impossibile")


def _do_topics(cat: StrayCat) -> str:
    df = _load_dataset(cat)
    if df is None or df.empty:
        return "Nessun argomento trovato o dataset vuoto."
    return f"Argomenti disponibili: {', '.join(sorted({str(t) for t in df['topic'].to_list()}))}"


async def _start_directly(cat: StrayCat, text: str, previous: Dict[str, Any] | None) -> str:
    """Avvia un esercizio dal dataset senza passare dal LLM e restituisce il testo per lo studente."""
    level = None
    for rx, name in _LEVELS:
        if rx.search(text):
            level = name
            break
    if level is None:
        level = normalize_level((previous or {}).get("difficulty", "easy"))
    topic = ""
    df = _load_dataset(cat)
    if df is not None and not df.empty:
        low = text.lower()
        for t in sorted({str(x) for x in df["topic"].to_list()}):
            if t.lower().replace("_", " ") in low or t.lower().split("_")[0] in low:
                topic = t
                break
    m = _ID_RE.search(text.upper())
    out = await (_start_by_id(cat, m.group(1), level) if m else _start_from_dataset(cat, topic, level))
    if out.startswith(_START_ERR):
        return out
    ex = await _get_current_exercise(cat)
    n = len(_steps_of(ex))
    return (f"Ecco il tuo esercizio ({level}):\n\n{ex.get('problem_statement', '')}\n\n"
            f"Si risolve in {n} passaggi. Iniziamo dal passaggio 1: {_steps_of(ex)[0]['description']}. "
            f"Scrivimi il risultato (es. \"il passaggio 1 fa ...\") oppure chiedi un suggerimento.")


@hook(priority=10)
async def fast_reply(reply, cat: StrayCat):
    if reply:
        return reply
    try:
        text = _message_text(cat).strip()
        if not text:
            return reply
        _LAST_USER_TEXT[str(cat.id)] = text

        try:
            ex = await _get_current_exercise(cat)
        except Exception:
            ex = None

        if _TOPICS_RE.search(text) and not _student_numbers(text):
            return _do_topics(cat)
        if _START_RE.search(text) and not _GEN_RE.search(text):
            return await _start_directly(cat, text, ex)
        if ex is None:
            return reply  # nessun esercizio attivo: flusso normale

        nums = _student_numbers(text)
        steps = _steps_of(ex)
        cur = int((ex.get("progress") or {}).get("current_step", 1))

        if _STATUS_RE.search(text) and not nums:
            return await _do_show(cat)
        if _HINT_RE.search(text) and not nums:
            return await _do_hint(cat)
        if not nums:
            return reply
        if _QUESTION_RE.search(text) and len(text) > 60:
            return reply  # probabile domanda di spiegazione, la gestisce il LLM

        if cur > len(steps) or (_FINAL_RE.search(text) and not _STEP_RE.search(text)):
            return await _do_final(cat, text)
        m = _STEP_RE.search(text)
        n = int(m.group(1)) if m else cur
        return await _do_verify(cat, n, fmt(nums[-1]))
    except Exception as e:
        log.warning(f"exercise_generator: fast_reply fallita, si prosegue con il LLM: {e}")
        return reply


TUTOR_RULES = """
=== TUTOR DI MATEMATICA: REGOLE (OBBLIGATORIE) ===
- Rispondi nella STESSA lingua dello studente.
- NON inventare mai esercizi, numeri o soluzioni. Ignora qualsiasi frammento di "Context" che contenga testi di esercizi o segnaposto tra parentesi graffe.
- Quando lo studente chiede un esercizio, DEVI chiamare lo strumento start_exercise (senza argomenti inventati: topic vuoto o un nome restituito da list_topics) e presentare SOLO il testo che restituisce.
- Quando lo studente indica il risultato di un passaggio, DEVI chiamare verify_intermediate_step. Per la soluzione finale, DEVI chiamare compare_final_solution.
- Se chiede un suggerimento, DEVI chiamare request_hint; se chiede a che punto e', usa show_active_exercise.
- Sii un tutor socratico: guida passo passo e non dare mai la soluzione o i valori dei passaggi in anticipo.
"""


@hook(priority=1)
async def agent_prompt_prefix(prefix: str, cat: StrayCat) -> str:
    prefix += TUTOR_RULES
    try:
        ex = await _get_current_exercise(cat)
    except Exception:
        return prefix
    steps = _steps_of(ex)
    cur = int((ex.get("progress") or {}).get("current_step", 1))
    step_line = f"Passaggio corrente: {cur}/{len(steps)} - {steps[cur - 1].get('description', '')}" if cur <= len(steps) else "Tutti i passaggi completati."
    prefix += f"\n=== ESERCIZIO ATTIVO ===\nTesto: \"{ex.get('problem_statement', '')}\"\n{step_line}\n"
    return prefix