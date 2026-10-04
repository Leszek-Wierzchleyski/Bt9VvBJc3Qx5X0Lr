#!/usr/bin/env python3
"""Application configuration with Google Sheet + local JSON fallback."""
from __future__ import annotations
import json, math, os, re, time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
try:
    import gspread
except ImportError:
    gspread = None

CONFIG_VERSION = 1
DEFAULT_CONFIG = {
    "config_version": CONFIG_VERSION,
    "default_risk_level": 2,
    "default_dca_amount": 250.0,
    "market_symbol": "BTCUSDT",
    "market_data_interval": "1d",
    "market_data_limit": 100,
    "trading_enabled_by_default": False,
    "config_refresh_hours": 1,
}
SUPPORTED_INTERVALS = {"1s","1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{5,20}$")

class ConfigError(ValueError): pass

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool): return value
    if isinstance(value, (int,float)) and value in (0,1): return bool(value)
    if isinstance(value, str):
        v=value.strip().lower()
        if v in {"true","1","yes","on"}: return True
        if v in {"false","0","no","off"}: return False
    raise ConfigError("Expected a boolean value.")

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged=deepcopy(DEFAULT_CONFIG); merged.update(config)
    try:
        version=int(merged["config_version"]); risk=int(merged["default_risk_level"])
        dca=float(merged["default_dca_amount"]); limit=int(merged["market_data_limit"])
        refresh=float(merged["config_refresh_hours"])
    except (TypeError,ValueError,KeyError) as exc:
        raise ConfigError("Configuration contains an invalid numeric value.") from exc
    symbol=str(merged["market_symbol"]).strip().upper(); interval=str(merged["market_data_interval"]).strip()
    if version != CONFIG_VERSION: raise ConfigError(f"Unsupported config version: {version}.")
    if risk not in {1,2,3}: raise ConfigError("default_risk_level must be 1, 2, or 3.")
    if not math.isfinite(dca) or dca <= 0: raise ConfigError("default_dca_amount must be finite and greater than zero.")
    if not SYMBOL_PATTERN.fullmatch(symbol): raise ConfigError("market_symbol has an invalid format.")
    if interval not in SUPPORTED_INTERVALS: raise ConfigError(f"Unsupported market_data_interval: {interval}.")
    if limit < 1 or limit > 1000: raise ConfigError("market_data_limit must be between 1 and 1000.")
    if not math.isfinite(refresh) or refresh <= 0: raise ConfigError("config_refresh_hours must be greater than zero.")
    return {"config_version":version,"default_risk_level":risk,"default_dca_amount":dca,"market_symbol":symbol,"market_data_interval":interval,"market_data_limit":limit,"trading_enabled_by_default":_as_bool(merged["trading_enabled_by_default"]),"config_refresh_hours":refresh}

def _cache_path() -> Path:
    return Path(os.getenv("APZIVA_CONFIG_CACHE", str(Path(__file__).resolve().parent / "config_cache.json"))).expanduser()

def write_cache(config: Dict[str, Any], path: Optional[Path]=None) -> None:
    target=path or _cache_path(); target.parent.mkdir(parents=True,exist_ok=True)
    payload={"cached_at_epoch":time.time(),"config":validate_config(config)}
    tmp=target.with_suffix(target.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); tmp.replace(target)

def read_cache(path: Optional[Path]=None) -> Dict[str,Any]:
    payload=json.loads((path or _cache_path()).read_text(encoding="utf-8")); return validate_config(payload["config"])

def _sheet_values_to_config(values: list[list[Any]]) -> Dict[str,Any]:
    result={}
    for row in values:
        if len(row)<2: continue
        key=str(row[0]).strip()
        if not key or key.startswith("#"): continue
        result[key]=row[1]
    return result

def read_google_sheet(*,spreadsheet_id:str,worksheet_name:str="Config",credentials_file:Optional[str]=None)->Dict[str,Any]:
    if gspread is None: raise ConfigError("gspread is not installed; Google Sheet configuration is unavailable.")
    credentials_path=credentials_file or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not credentials_path: raise ConfigError("Google service-account credential path is not configured.")
    client=gspread.service_account(filename=credentials_path)
    ws=client.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    return validate_config(_sheet_values_to_config(ws.get_all_values()))

def load_config(*,force_refresh:bool=False,spreadsheet_id:Optional[str]=None,worksheet_name:str="Config",credentials_file:Optional[str]=None,cache_path:Optional[Path]=None):
    cache=cache_path or _cache_path(); spreadsheet_id=spreadsheet_id or os.getenv("APZIVA_CONFIG_SPREADSHEET_ID")
    cached=None; cached_at=None; refresh_hours=DEFAULT_CONFIG["config_refresh_hours"]
    try:
        payload=json.loads(cache.read_text(encoding="utf-8")); cached=validate_config(payload["config"]); cached_at=float(payload["cached_at_epoch"]); refresh_hours=cached["config_refresh_hours"]
    except Exception: pass
    due=force_refresh or cached is None or cached_at is None or (time.time()-cached_at)>=refresh_hours*3600
    if due and spreadsheet_id:
        try:
            fresh=read_google_sheet(spreadsheet_id=spreadsheet_id,worksheet_name=worksheet_name,credentials_file=credentials_file); write_cache(fresh,cache); return fresh,"google_sheet"
        except Exception as exc:
            # Keep the cache as the safe fallback, but expose a sanitised
            # diagnostic in the terminal so configuration failures are
            # observable during local development. Never print credential
            # contents or secret values.
            message = str(exc)
            if credentials_file:
                message = message.replace(str(credentials_file), "<credentials-file>")
            for secret_name in ("APZIVA_CONFIG_SPREADSHEET_ID", "GOOGLE_SERVICE_ACCOUNT_FILE"):
                message = re.sub(
                    rf"(?i){re.escape(secret_name)}\s*[=:]\s*[^,;\s]+",
                    f"{secret_name}=<redacted>",
                    message,
                )
            print(
                f"[Apziva_Config] Google Sheet refresh failed: "
                f"{type(exc).__name__}: {message}"
            )
    if cached is not None: return cached,"local_cache"
    return validate_config(DEFAULT_CONFIG),"defaults"

def get_config_status(*,cache_path:Optional[Path]=None)->Dict[str,Any]:
    try:
        payload=json.loads((cache_path or _cache_path()).read_text(encoding="utf-8")); config=validate_config(payload["config"])
        return {"source":"local_cache","cached_at_epoch":float(payload["cached_at_epoch"]),"config_version":config["config_version"]}
    except Exception: return {"source":"defaults","cached_at_epoch":None,"config_version":CONFIG_VERSION}
