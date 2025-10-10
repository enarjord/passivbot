# sitecustomize.py
"""
ML Tech CCXT retargeting hook for Passivbot.

Enable by setting env: MLT_ENABLE=1
Optional env:
  MLT_USER_AGENT="python-requests"
  MLT_BINANCE_WS_DIRECT=1   # keep Binance private WS direct to Binance (recommended)
"""

import os

if os.environ.get("MLT_ENABLE") != "1":
    raise SystemExit

import ccxt

UA = os.environ.get("MLT_USER_AGENT", "python-requests")

# ---- helpers ---------------------------------------------------------------

def _retarget_okx(ex):
    ex.userAgent = UA
    ex.headers = {**(ex.headers or {})}
    # HTTP
    ex.urls["api"] = {"public": "https://okx.mltech.ai", "private": "https://okx.mltech.ai"}
    # WS
    ex.urls["ws"] = {"public": "wss://wsokx.mltech.ai", "private": "wss://wsokx.mltech.ai"}
    return ex

def _retarget_bybit(ex):
    ex.userAgent = UA
    ex.headers = {**(ex.headers or {})}
    ex.urls["api"] = {"public": "https://bybit.mltech.ai", "private": "https://bybit.mltech.ai"}
    ex.urls["ws"]  = {"public": "wss://wsbybit.mltech.ai", "private": "wss://wsbybit.mltech.ai"}
    return ex

def _retarget_binance_spot_margin(ex):
    ex.userAgent = UA
    ex.headers = {**(ex.headers or {})}
    # SAPI (spot & margin)
    if "api" in ex.urls and isinstance(ex.urls["api"], dict):
        ex.urls["api"]["sapi"] = "https://binance-sapi-1.mltech.ai"
    return ex

def _retarget_binance_usdm(ex):
    ex.userAgent = UA
    ex.headers = {**(ex.headers or {})}
    if "api" in ex.urls and isinstance(ex.urls["api"], dict):
        ex.urls["api"]["fapi"] = "https://binance-fapi-1.mltech.ai"
    return ex

def _retarget_binance_coinm(ex):
    ex.userAgent = UA
    ex.headers = {**(ex.headers or {})}
    if "api" in ex.urls and isinstance(ex.urls["api"], dict):
        ex.urls["api"]["dapi"] = "https://binance-dapi-1.mltech.ai"
    return ex

# ---- patch subclasses ------------------------------------------------------

def _wrap_class(cls, apply_fn):
    class Wrapped(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_fn(self)
    Wrapped.__name__ = cls.__name__
    return Wrapped

# Only patch what Passivbot commonly uses
if hasattr(ccxt, "okx"):
    ccxt.okx = _wrap_class(ccxt.okx, _retarget_okx)

if hasattr(ccxt, "bybit"):
    ccxt.bybit = _wrap_class(ccxt.bybit, _retarget_bybit)

# Binance families: spot/margin, USDM futures, COINM futures
if hasattr(ccxt, "binance"):
    ccxt.binance = _wrap_class(ccxt.binance, _retarget_binance_spot_margin)
if hasattr(ccxt, "binanceusdm"):
    ccxt.binanceusdm = _wrap_class(ccxt.binanceusdm, _retarget_binance_usdm)
if hasattr(ccxt, "binancecoinm"):
    ccxt.binancecoinm = _wrap_class(ccxt.binancecoinm, _retarget_binance_coinm)

# ---- optional: keep Binance private WS direct to Binance -------------------
# Per MLT+ instruction: create listenKey via MLT+ REST, then connect WS to Binance.
# Passivbot already handles the WS URLs internally; no need to force WS here.
# If you want to be explicit, leave WS entries as-is for Binance.

