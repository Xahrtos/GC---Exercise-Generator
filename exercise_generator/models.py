import json
import pandas as pd
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExerciseTemplate(BaseModel):
    exercise_id: str
    topic: str
    type: str = "static"
    problem_statement: Optional[str] = None
    problem_template: Optional[str] = None
    correct_solution: str
    difficulty: str = "easy"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    random_params: Dict[str, Any] = Field(default_factory=dict)
    derived_params: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list)


class ExerciseDataManager:
    @staticmethod
    def load_data(content: bytes) -> Optional[pd.DataFrame]:
        """Carica il dataset di esercizi da un flusso di byte JSON o CSV."""
        try:
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                return pd.DataFrame(data)
        except Exception:
            pass
        return None

    @staticmethod
    def get_exercise_by_id(df: pd.DataFrame, exercise_id: str) -> Optional[Dict[str, Any]]:
        """Cerca un esercizio specifico nel DataFrame in base all'ID."""
        if df is None or df.empty:
            return None
        match = df[df["exercise_id"].astype(str).str.strip().str.upper() == exercise_id.strip().upper()]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    @staticmethod
    def parse_steps(raw: Any) -> List[Dict[str, Any]]:
        """Converte i passaggi intermedi in una lista standard di dizionari."""
        if isinstance(raw, list):
            return [s for s in raw if isinstance(s, dict)]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [s for s in parsed if isinstance(s, dict)]
            except Exception:
                pass
        return []