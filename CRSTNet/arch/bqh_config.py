from typing import Any, Dict


def get_bqh_config() -> Dict[str, Any]:
    """
    Legacy helper name preserved for compatibility.

    The returned defaults are paper-aligned CRSTNet parameters rather than the
    earlier BQH placeholder settings.
    """

    return {
        "theta": 0.2,
        "lambda_value": 0.5,
        "gamma": 0.5,
        "maintenance_interval": 5,
        "k_min": 2,
        "k_max": 6,
        "threshold_quantile": 0.9,
        "eta_scale": 0.1,
    }


def validate_bqh_config(config: Dict[str, Any]) -> bool:
    required_keys = [
        "theta",
        "lambda_value",
        "gamma",
        "maintenance_interval",
        "k_min",
        "k_max",
        "threshold_quantile",
    ]
    return all(key in config for key in required_keys)
