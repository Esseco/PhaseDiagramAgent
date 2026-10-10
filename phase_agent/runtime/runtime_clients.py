"""Optional startup client: missing credentials are not configuration errors."""


def create_optional_runtime_client(parameters, *, credential_loader=None, client_factory=None):
    if credential_loader is None:
        from phase_agent.runtime.deepseek_credentials import load_deepseek_api_key

        credential_loader = load_deepseek_api_key
    if client_factory is None:
        from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client

        client_factory = create_deepseek_client
    if parameters.get("thinking") not in {None, "enabled", "disabled"}:
        raise ValueError("thinking 必须是 enabled、disabled 或 None")
    key = credential_loader()
    if not key:
        return None
    return client_factory(api_key=key, **parameters)
