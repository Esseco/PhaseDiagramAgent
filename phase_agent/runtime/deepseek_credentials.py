"""Store the local DeepSeek key in the current Windows user's Credential Manager."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


_TARGET = "PhaseSearchAgent/DeepSeekAPIKey"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def load_deepseek_api_key() -> str | None:
    """Prefer an explicit process environment value, then the OS credential store."""
    environment_key = os.environ.get("DEEPSEEK_API_KEY")
    if environment_key:
        return environment_key
    if os.name != "nt":
        return None
    return _read_windows_credential()


def save_deepseek_api_key(api_key: str) -> None:
    """Persist a key for this Windows user without writing it into project files."""
    key = str(api_key or "").strip()
    if len(key) < 16:
        raise ValueError("DeepSeek API Key 看起来过短，请检查是否粘贴完整。")
    if os.name != "nt":
        raise RuntimeError("当前平台没有配置系统凭据库；请在本机通过 DEEPSEEK_API_KEY 提供密钥。")

    _write_windows_credential(_TARGET, key, "Local Phase Search Agent DeepSeek API key")


def _write_windows_credential(target: str, value: str, comment: str) -> None:
    """Write one generic secret for the current Windows user."""
    advapi = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    advapi.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
    advapi.CredWriteW.restype = wintypes.BOOL
    secret = value.encode("utf-8")
    secret_buffer = ctypes.create_string_buffer(secret)
    credential = _Credential(
        Flags=0,
        Type=_CRED_TYPE_GENERIC,
        TargetName=target,
        Comment=comment,
        LastWritten=wintypes.FILETIME(0, 0),
        CredentialBlobSize=len(secret),
        CredentialBlob=ctypes.cast(secret_buffer, ctypes.POINTER(ctypes.c_ubyte)),
        Persist=_CRED_PERSIST_LOCAL_MACHINE,
        AttributeCount=0,
        Attributes=None,
        TargetAlias=None,
        UserName="DeepSeek API",
    )
    if not advapi.CredWriteW(ctypes.byref(credential), 0):
        error = ctypes.get_last_error()
        raise OSError(error, "Windows 凭据管理器保存失败")


def _read_windows_credential(target: str = _TARGET) -> str | None:
    advapi = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    advapi.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_Credential)),
    ]
    advapi.CredReadW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = [ctypes.c_void_p]
    advapi.CredFree.restype = None
    pointer = ctypes.POINTER(_Credential)()
    if not advapi.CredReadW(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        return None
    try:
        credential = pointer.contents
        if not credential.CredentialBlob or not credential.CredentialBlobSize:
            return None
        raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
        return raw.decode("utf-8")
    finally:
        advapi.CredFree(pointer)
