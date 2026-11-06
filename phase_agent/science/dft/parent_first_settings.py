"""Validate the frozen DFT configuration in a parent-first input manifest."""


def validate_parent_first_dft_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {
        'config_version', 'model_version', 'approval_record_id', 'parameters'
    } or any(not isinstance(settings.get(key), str) or not settings[key] for key in
             ('config_version', 'model_version', 'approval_record_id')) or not isinstance(settings['parameters'], dict):
        raise ValueError('parent_first_dft_settings_invalid')
