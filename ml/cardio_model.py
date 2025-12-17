from pathlib import Path
from typing import Dict

import joblib
import numpy as np

# Lokasi file model (pipeline XGBoost) yang sudah Anda train sebelumnya.
MODEL_PATH = Path(__file__).resolve().parent / "best_xgb_pipeline.joblib"


class CardioRiskModel:
    """Wrapper sederhana untuk pipeline XGBoost penyakit jantung.

    Asumsi fitur input (urutan kolom) konsisten dengan saat training, misalnya:
    ['age_years','gender','bmi','map','cholesterol','gluc','smoke','alco','active']
    """

    _instance = None  # singleton sederhana agar model hanya dimuat sekali

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Reload model if it failed to initialize (Streamlit hot-reload safety)."""
        if not getattr(self, "pipeline", None):
            self._load_model()

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file tidak ditemukan di {MODEL_PATH}. "
                "Pastikan Anda sudah meletakkan best_xgb_pipeline.joblib di folder ml/."
            )

        # --- Compatibility shim for scikit-learn 1.8+ ---
        # shap<0.51 still imports the private `_is_pandas_df` helper that was removed
        # in newer scikit-learn releases. We add a lightweight replacement before
        # importing shap so the import does not crash.
        try:
            from sklearn.utils import validation as sk_validation
            if not hasattr(sk_validation, "_is_pandas_df"):
                import pandas as pd

                def _is_pandas_df(x):
                    return isinstance(x, pd.DataFrame)

                sk_validation._is_pandas_df = _is_pandas_df
        except Exception as e:
            print(f"Warning: sklearn/shap compatibility patch failed: {e}")

        # AdaBoostClassifier in scikit-learn 1.8 no longer accepts the deprecated
        # `algorithm` argument that older imbalanced-learn versions still pass
        # during import. We shim the signature to keep their imports working.
        try:
            from sklearn.ensemble import AdaBoostClassifier
            import inspect

            if "algorithm" not in inspect.signature(AdaBoostClassifier.__init__).parameters:
                _orig_init = AdaBoostClassifier.__init__

                def _patched_init(self, *args, algorithm=None, **kwargs):
                    return _orig_init(self, *args, **kwargs)

                AdaBoostClassifier.__init__ = _patched_init
        except Exception as e:
            print(f"Warning: AdaBoost compatibility patch failed: {e}")

        self.pipeline = joblib.load(MODEL_PATH)

        # Initialize SHAP Explainer
        # Assuming the pipeline has a step named 'classifier' or is just the model
        # If it's a pipeline, we need to handle the preprocessor separately if it exists
        try:
            import shap
            # Try to get the actual model object
            if hasattr(self.pipeline, 'named_steps') and 'classifier' in self.pipeline.named_steps:
                self.model_obj = self.pipeline.named_steps['classifier']
            else:
                self.model_obj = self.pipeline
                
            self.explainer = shap.TreeExplainer(self.model_obj)
        except Exception as e:
            print(f"Warning: SHAP initialization failed: {e}")
            self.explainer = None

    def _to_feature_array(self, data: Dict) -> np.ndarray:
        ordered = [
            data["age_years"],
            data["gender"],
            data["bmi"],
            data["map"],
            data["cholesterol"],
            data["gluc"],
            data["smoke"],
            data["alco"],
            data["active"],
        ]
        return np.array(ordered, dtype=float).reshape(1, -1)

    def predict_proba(self, data: Dict) -> float:
        self._ensure_loaded()
        X = self._to_feature_array(data)
        proba = self.pipeline.predict_proba(X)[0, 1]
        return float(proba)

    def predict_label(self, data: Dict, threshold: float = 0.5) -> int:
        return int(self.predict_proba(data) >= threshold)

    def get_shap_values(self, data: Dict) -> Dict[str, float]:
        self._ensure_loaded()
        if not self.explainer:
            return {}
            
        X = self._to_feature_array(data)
        # Note: If there's a preprocessor, X should be transformed first. 
        # Assuming simple pipeline for now or that X matches model input.
        
        shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP output formats (list for multiclass, array for binary)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] # Positive class
        else:
            sv = shap_values[0]
            
        feature_names = ['Usia', 'Gender', 'BMI', 'MAP', 'Kolesterol', 'Glukosa', 'Rokok', 'Alkohol', 'Aktif']
        
        return {k: float(v) for k, v in zip(feature_names, sv)}
