"""
data_fetcher.py
撈取台股歷史資料，自動判斷上市(.TW) / 上櫃(.TWO)，並重採樣週線與三日線。
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict


def normalize_ticker(ticker: str) -> str:
    """
    將純數字代號轉換為 yfinance 格式。
    - 已含副檔名（.TW / .TWO）直接回傳
    - 純數字先嘗試 .TW，失敗再試 .TWO
    """
    t = ticker.strip().upper()
    if "." in t:
        return t
    if t.isdigit():
        return _resolve_tw_ticker(t)
    return t


def _resolve_tw_ticker(code: str) -> str:
    """
    嘗試 .TW → .TWO，回傳能取到資料的那個。
    若兩個都失敗，預設回傳 .TW（讓後續報錯訊息更明確）。
    """
    for suffix in [".TW", ".TWO"]:
        candidate = code + suffix
        try:
            probe = yf.download(
                candidate,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=10,
            )
            if not probe.empty:
                return candidate
        except Exception:
            continue
    return code + ".TW"   # 預設，讓外層顯示正確錯誤


def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    下載日線 OHLCV。
    回傳欄位全小寫，index 為 DatetimeIndex（名稱 'date'）。
    """
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        timeout=20,
    )
    if raw.empty:
        raise ValueError(f"無法取得 {ticker} 資料，請確認代號是否正確。")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"

    df = raw[["open", "high", "low", "close", "volume"]].dropna()
    return df


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """日線 → 週線（以週五收盤為一週終點）。"""
    weekly = daily.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return weekly.dropna(subset=["open", "close"])


def resample_3day(daily: pd.DataFrame) -> pd.DataFrame:
    """日線 → 三日線（每 3 根交易日合成一根）。"""
    rows = []
    idx_list = daily.index.tolist()
    n = len(idx_list)

    for i in range(0, n, 3):
        chunk = daily.iloc[i : min(i + 3, n)]
        if chunk.empty:
            continue
        rows.append({
            "date":   chunk.index[-1],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df3 = pd.DataFrame(rows).set_index("date")
    df3.index = pd.to_datetime(df3.index)
    return df3


def get_all_timeframes(ticker: str, period: str = "2y") -> Dict[str, object]:
    """
    一次撈取，回傳三個時間框架。
    ticker 可以是純數字（自動判斷 .TW / .TWO）或完整代號。
    """
    nt      = normalize_ticker(ticker)
    daily   = fetch_daily(nt, period)
    weekly  = resample_weekly(daily)
    three_d = resample_3day(daily)
    return {
        "ticker": nt,
        "daily":  daily,
        "weekly": weekly,
        "3d":     three_d,
    }