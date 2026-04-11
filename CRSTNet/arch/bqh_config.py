from typing import Any, Dict

def get_bqh_config() -> Dict[str, Any]:

    return {
        'keynode': {
            'hysteresis_alpha': 0.1,
            'k_min_ratio': 0.05,
            'k_max_ratio': 0.3,
            'budget_ratio': 0.1
        },
        'clustering': {
            'merge_threshold': 0.8,
            'split_threshold': 0.5,
            'cooldown': 20
        }
    }

def validate_bqh_config(config: Dict[str, Any]) -> bool:

    required_keys = ['keynode', 'clustering']
    return all(key in config for key in required_keys)
