import sympy as sp
from typing import Any, Dict

def check_answer(problem: str, user_answer: str) -> Dict[str, Any]:
    """Verifica l'equivalenza simbolica o numerica tra la risposta attesa e quella dell'utente tramite SymPy."""
    try:
        # Parsing sicuro delle espressioni con SymPy
        expected_expr = sp.parse_expr(problem, evaluate=True)
        user_expr = sp.parse_expr(user_answer, evaluate=True)
        
        # Verifica dell'equivalenza
        difference = sp.simplify(expected_expr - user_expr)
        is_ok = (difference == 0)
        
        return {
            "ok": True,
            "result": {
                "ok_answer": is_ok,
                "expected": str(expected_expr),
                "user": str(user_expr)
            }
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }