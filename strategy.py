"""
strategy.py
1012 極速框多週期策略 — 完整狀態機 + 手動持倉分析 + 極速框回踩再進場
最終版本：含交易日記 (trade_log)、MTF 共振、聽牌偵測、3D 階梯線
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# § 1  基礎指標
# ─────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, ma_fast: int = 10, ma_slow: int = 200) -> pd.DataFrame:
    df = df.copy()
    df["body_high"] = df[["open", "close"]].max(axis=1)
    df["body_low"]  = df[["open", "close"]].min(axis=1)
    df[f"ma{ma_fast}"] = df["close"].rolling(ma_fast,  min_periods=ma_fast).mean()
    df["ma55"]          = df["close"].rolling(55,       min_periods=55).mean()
    df[f"ma{ma_slow}"] = df["close"].rolling(ma_slow,  min_periods=ma_slow).mean()
    df["vol_ma20"]      = df["volume"].rolling(20,      min_periods=20).mean()
    return df


# ─────────────────────────────────────────────────────────────────────
# § 2  週黑吞偵測
# ─────────────────────────────────────────────────────────────────────

def build_weekly_washout(weekly: pd.DataFrame) -> pd.Series:
    w = weekly.copy()
    w["body_low"] = w[["open", "close"]].min(axis=1)
    prev_bl = w["body_low"].shift(1)
    return (w["close"] < prev_bl).rename("weekly_washout")


def map_weekly_to_daily(weekly_signal: pd.Series,
                         daily_index: pd.DatetimeIndex) -> pd.Series:
    return weekly_signal.reindex(daily_index, method="ffill").fillna(False)


# ─────────────────────────────────────────────────────────────────────
# § 2.5  MTF 趨勢預計算（供 1012 濾網使用，必須在狀態機前執行）
# ─────────────────────────────────────────────────────────────────────

def _pre_calc_mtf_trends(
    daily: pd.DataFrame,
    three_d: pd.DataFrame,
    weekly: pd.DataFrame,
    ma_fast: int = 10,
) -> pd.DataFrame:
    """
    在狀態機執行前預先計算三週期趨勢布林值。
    結果寫入 daily DataFrame，供 mark_1012_candidates 的 MTF 濾網使用。

    新增欄位：
        Daily_Trend_Up  : bool
        3D_Trend_Up     : bool
        Weekly_Trend_Up : bool
    """
    maf = f"ma{ma_fast}"
    df  = daily.copy()

    # ── 日線（直接用現有均線）────────────────────────────────────────
    df["Daily_Trend_Up"] = df["close"] > df[maf]

    # ── 3D 線（shift(1) 防前視偏誤）─────────────────────────────────
    if three_d is not None and not three_d.empty:
        td = three_d.copy()
        td["_3d_ma"]    = td["close"].rolling(ma_fast, min_periods=1).mean()
        td["_3d_trend"] = td["close"] > td["_3d_ma"]
        df["3D_Trend_Up"] = (
            td["_3d_trend"]
            .shift(1)
            .reindex(df.index, method="ffill")
            .fillna(False)
            .values
        )
    else:
        df["3D_Trend_Up"] = False

    # ── 週線（shift(1) 防前視偏誤）──────────────────────────────────
    if weekly is not None and not weekly.empty:
        wk = weekly.copy()
        wk["_wk_ma"]    = wk["close"].rolling(ma_fast, min_periods=1).mean()
        wk["_wk_trend"] = wk["close"] > wk["_wk_ma"]
        df["Weekly_Trend_Up"] = (
            wk["_wk_trend"]
            .shift(1)
            .reindex(df.index, method="ffill")
            .fillna(False)
            .values
        )
    else:
        df["Weekly_Trend_Up"] = False

    return df


# ─────────────────────────────────────────────────────────────────────
# § 3  1012 候選條件（向量化）
# ─────────────────────────────────────────────────────────────────────

def mark_1012_candidates(df: pd.DataFrame, ma_fast: int = 10, ma_slow: int = 200) -> pd.DataFrame:
    maf = f"ma{ma_fast}"
    mas = f"ma{ma_slow}"

    c_t2   = df["close"].shift(2)
    maf_t2 = df[maf].shift(2)
    bh_t2  = df["body_high"].shift(2)
    o_t1   = df["open"].shift(1)
    c_t1   = df["close"].shift(1)
    maf_t1 = df[maf].shift(1)
    bh_t1  = df["body_high"].shift(1)
    lo_t1  = df["low"].shift(1)
    hi_t1  = df["high"].shift(1)

    cond_t2 = c_t2 > maf_t2
    cond_t1 = (o_t1 > maf_t1) & (c_t1 > maf_t1) & (c_t1 <= bh_t2)
    cond_t0 = (
        (df["open"]  > df[maf])          &
        (df["close"] > df[maf])          &
        (df["close"] > df["ma55"])       &
        (df["close"] > df[mas])          &
        (df["close"] > bh_t1)            &
        (df["volume"] > df["vol_ma20"])
    )

    # ── MTF 三週期共振濾網（需 _pre_calc_mtf_trends 先執行）─────────
    cond_mtf = (
        df.get("Daily_Trend_Up",  pd.Series(True, index=df.index)) &
        df.get("3D_Trend_Up",     pd.Series(True, index=df.index)) &
        df.get("Weekly_Trend_Up", pd.Series(True, index=df.index))
    )

    df["_1012_raw"] = cond_t2 & cond_t1 & cond_t0 & cond_mtf
    df["_t1_low"]   = np.where(df["_1012_raw"], lo_t1, np.nan)
    df["_t1_high"]  = np.where(df["_1012_raw"], hi_t1, np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────
# § 4  狀態機輔助
# ─────────────────────────────────────────────────────────────────────

def _init_state() -> dict:
    return {
        "in_position": False, "entry_price": None, "stop_loss": None,
        "sb_active": False, "sb_p0": None, "sb_p1": None, "sb_target": None,
        "washout_locked": False, "v_monitor": False,
        "v_ref_price": None, "v_blocked": False,
    }


def _enter_position(state, close, low, t1_low, t1_high, sb_ratio):
    state.update({
        "in_position": True, "entry_price": close, "stop_loss": low,
        "v_monitor": False, "v_blocked": False, "washout_locked": False,
        "sb_active": True, "sb_p0": t1_low, "sb_p1": t1_high,
        "sb_target": t1_low + (t1_high - t1_low) * sb_ratio,
    })
    return state


def _exit_position(state):
    ep = state["entry_price"]
    state.update({
        "v_monitor": True, "v_ref_price": ep,
        "in_position": False, "sb_active": False,
        "entry_price": None, "stop_loss": None,
        "sb_p0": None, "sb_p1": None, "sb_target": None,
    })
    return state


def _close_trade(trade_log: list, open_trade: dict,
                 exit_date, exit_price: float, exit_reason: str) -> dict:
    """結算一筆交易並寫入 trade_log，回傳空的 open_trade。"""
    if not open_trade:
        return {}
    ep   = open_trade["entry_price"]
    sl   = open_trade.get("stop_loss")
    pnl  = (exit_price - ep) / ep * 100
    risk = (ep - sl) if (sl and sl < ep) else ep * 0.03
    r_multiple = (exit_price - ep) / risk if risk > 0 else 0

    trade_log.append({
        "進場日期":  open_trade["entry_date"].strftime("%Y-%m-%d"),
        "出場日期":  exit_date.strftime("%Y-%m-%d"),
        "進場類型":  open_trade["entry_type"],
        "進場價":    round(ep, 2),
        "出場價":    round(exit_price, 2),
        "出場原因":  exit_reason,
        "報酬率(%)": round(pnl, 2),
        "R倍數":     round(r_multiple, 2),
        "勝負":      "✅ 獲利" if pnl > 0 else "❌ 虧損",
    })
    return {}


# ─────────────────────────────────────────────────────────────────────
# § 5  主策略狀態機
# ─────────────────────────────────────────────────────────────────────

def run_strategy(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    ma_fast: int = 10,
    ma_slow: int = 200,
    sb_ratio: float = 5.8,
    three_d: pd.DataFrame = None,
    timeframe: str = "1D",
) -> tuple:
    """
    執行完整策略狀態機。

    Returns
    -------
    (df, trade_log)
        df        : pd.DataFrame — 帶有所有訊號欄的日線資料
        trade_log : list[dict]   — 歷史交易紀錄（供回測頁面使用）
    """
    df = add_indicators(daily, ma_fast, ma_slow)

    # ── 週期選擇：決定極速框參考高低點的基底資料 ──────────────────────
    _tf_map = {
        "1D": daily,
        "3D": three_d if three_d is not None and not three_d.empty else daily,
        "1W": weekly,
    }
    sb_base_df = _tf_map.get(timeframe, daily)

    if timeframe != "1D" and sb_base_df is not daily:
        sb_base_aligned_high = (
            sb_base_df["high"]
            .shift(1)
            .reindex(daily.index, method="ffill")
        )
        sb_base_aligned_low = (
            sb_base_df["low"]
            .shift(1)
            .reindex(daily.index, method="ffill")
        )
        df["_sb_ref_high"] = sb_base_aligned_high
        df["_sb_ref_low"]  = sb_base_aligned_low
    else:
        df["_sb_ref_high"] = df["high"]
        df["_sb_ref_low"]  = df["low"]

    # ── MTF 趨勢預計算（必須在 mark_1012_candidates 之前）────────────
    df = _pre_calc_mtf_trends(df, three_d, weekly, ma_fast)

    df = mark_1012_candidates(df, ma_fast, ma_slow)

    # ── 週黑吞（shift(1) 防前視偏誤）────────────────────────────────
    wo_raw         = build_weekly_washout(weekly)
    wo_raw_shifted = wo_raw.shift(1)
    wo_daily       = map_weekly_to_daily(wo_raw_shifted, df.index)
    df["weekly_washout"] = wo_daily.fillna(False)
    df["_wo_trigger"]    = df["weekly_washout"] & ~df["weekly_washout"].shift(1).fillna(False)

    # ── 初始化輸出欄位 ────────────────────────────────────────────────
    for c in ["signal_1012", "speed_box_ok", "speed_box_fail", "sb_stop",
              "weekly_tp", "weekly_sl", "v_reload", "v_blocked",
              "in_position", "sb_active"]:
        df[c] = False
    for c in ["entry_price", "stop_loss_price",
              "sb_param0", "sb_param1", "sb_target",
              "sb_tp1", "sb_tp2", "sb_tp3"]:
        df[c] = np.nan
    df["Alert_Message"] = ""

    # ── 交易日記 ──────────────────────────────────────────────────────
    trade_log   = []
    _open_trade = {}

    state     = _init_state()
    maf_col   = f"ma{ma_fast}"
    body_lows = df["body_low"].values

    # ═════════════════════════════════════════════════════════════════
    # 狀態機主迴圈
    # ═════════════════════════════════════════════════════════════════
    for i in range(len(df)):
        idx   = df.index[i]
        row   = df.iloc[i]
        close = float(row["close"])
        open_ = float(row["open"])
        low   = float(row["low"])
        ma_val = row[maf_col]

        # ── A. 週黑吞觸發 ─────────────────────────────────────────────
        if row["_wo_trigger"]:
            if state["in_position"]:
                if close > state["entry_price"]:
                    df.at[idx, "weekly_tp"] = True
                    _open_trade = _close_trade(
                        trade_log, _open_trade, idx, close, "週K停利"
                    )
                else:
                    df.at[idx, "weekly_sl"] = True
                    _open_trade = _close_trade(
                        trade_log, _open_trade, idx, close, "週K停損"
                    )
                state = _exit_position(state)
            else:
                state["washout_locked"] = True

        # ── B. 週黑吞解除 ─────────────────────────────────────────────
        if not row["weekly_washout"]:
            state["washout_locked"] = False

        # ── C. 極速框追蹤 ─────────────────────────────────────────────
        if state["sb_active"] and state["in_position"]:
            df.at[idx, "sb_active"]  = True
            df.at[idx, "sb_param0"]  = state["sb_p0"]
            df.at[idx, "sb_param1"]  = state["sb_p1"]
            df.at[idx, "sb_target"]  = state["sb_target"]

            _dist = state["sb_p1"] - state["sb_p0"]
            df.at[idx, "sb_tp1"] = state["sb_p0"] + _dist * 2.618
            df.at[idx, "sb_tp2"] = state["sb_p0"] + _dist * 4.8
            df.at[idx, "sb_tp3"] = state["sb_p0"] + _dist * 5.8

            is_black = close < open_
            if is_black:
                df.at[idx, "speed_box_fail"] = True
                state["sb_active"] = False
                # 極速框失效，持倉繼續但不結算（等待後續出場訊號）

            elif close >= state["sb_target"]:
                df.at[idx, "speed_box_ok"] = True
                state["sb_active"] = False
                _open_trade = _close_trade(
                    trade_log, _open_trade, idx, close, "極速框TP3達標"
                )
                # 極速框成立視為一個里程碑，但策略不強制出場
                # 若想改為自動出場，可在此呼叫 state = _exit_position(state)

            elif close < state["sb_p0"]:
                df.at[idx, "sb_stop"] = True
                state["sb_active"] = False
                _open_trade = _close_trade(
                    trade_log, _open_trade, idx, close, "極速框底線停損"
                )
                state = _exit_position(state)

        # ── D. 1012 進場訊號 ──────────────────────────────────────────
        if (not state["in_position"] and not state["washout_locked"]
                and row["_1012_raw"]
                and pd.notna(row["_t1_low"]) and pd.notna(row["_t1_high"])):

            df.at[idx, "signal_1012"] = True
            ref_low  = float(row.get("_sb_ref_low",  row["_t1_low"]))
            ref_high = float(row.get("_sb_ref_high", row["_t1_high"]))
            state = _enter_position(state, close, low, ref_low, ref_high, sb_ratio)

            # 記錄進場
            _open_trade = {
                "entry_date":  idx,
                "entry_price": close,
                "entry_type":  "1012首次進場",
                "stop_loss":   low,
                "sb_p0":       ref_low,
                "sb_p1":       ref_high,
            }

            # 生成快訊文案
            _d   = ref_high - ref_low
            _tp1 = round(ref_low + _d * 2.618, 2)
            _tp2 = round(ref_low + _d * 4.8,   2)
            _tp3 = round(ref_low + _d * 5.8,   2)
            df.at[idx, "Alert_Message"] = (
                f"🚀 Long Entry Triggered! | "
                f"SL (p0): {round(ref_low, 2)} | "
                f"Entry (p1): {round(ref_high, 2)} | "
                f"TP1: {_tp1} | TP2: {_tp2} | TP3: {_tp3}"
            )

        # ── E. V轉 Reload 監控 ────────────────────────────────────────
        if state["v_monitor"] and not state["in_position"] and pd.notna(ma_val):
            if close < float(ma_val) and i >= 2:
                bl_0, bl_1, bl_2 = body_lows[i], body_lows[i-1], body_lows[i-2]
                if bl_0 < bl_1 < bl_2:
                    state["v_blocked"] = True
            df.at[idx, "v_blocked"] = state["v_blocked"]

            if (not state["v_blocked"] and state["v_ref_price"] is not None
                    and close >= state["v_ref_price"]):
                df.at[idx, "v_reload"] = True
                ref_price = state["v_ref_price"]
                state["in_position"]  = True
                state["entry_price"]  = ref_price
                state["v_monitor"]    = False

                _open_trade = {
                    "entry_date":  idx,
                    "entry_price": ref_price,
                    "entry_type":  "V轉Reload",
                    "stop_loss":   ref_price * 0.97,
                    "sb_p0":       None,
                    "sb_p1":       None,
                }

        # ── F. 記錄當日持倉狀態 ───────────────────────────────────────
        df.at[idx, "in_position"] = state["in_position"]
        if state["in_position"] and state["entry_price"] is not None:
            df.at[idx, "entry_price"]     = state["entry_price"]
            df.at[idx, "stop_loss_price"] = state["stop_loss"] or np.nan

    # ── 清理暫存欄位 ──────────────────────────────────────────────────
    df.drop(
        columns=["_1012_raw", "_t1_low", "_t1_high",
                 "_wo_trigger", "_sb_ref_high", "_sb_ref_low"],
        errors="ignore",
        inplace=True,
    )

    # ── 若有未結算的持倉，以最後收盤價結算（標記為未平倉）────────────
    if _open_trade and state["in_position"]:
        _close_trade(
            trade_log, _open_trade,
            df.index[-1], float(df["close"].iloc[-1]),
            "持倉中（未結算）"
        )

    # ── 極速框回踩偵測 ────────────────────────────────────────────────
    if three_d is not None and not three_d.empty:
        df = detect_sb_pullback(df, three_d, ma_fast)

    # ── 聽牌偵測 ──────────────────────────────────────────────────────
    df = mark_on_deck(df, ma_fast, ma_slow)

    # ── MTF 分數 + 3D 階梯線 ─────────────────────────────────────────
    if three_d is not None and not three_d.empty:
        df = mark_mtf_trend(df, three_d, weekly, ma_fast)
    else:
        df["MTF_Score"]    = (
            df["Daily_Trend_Up"].astype(int) +
            df["3D_Trend_Up"].astype(int) +
            df["Weekly_Trend_Up"].astype(int)
        )
        df["Prev_3D_High"] = np.nan
        df["Prev_3D_Low"]  = np.nan

    return df, trade_log


# ─────────────────────────────────────────────────────────────────────
# § 6  極速框回踩再進場偵測（3D K 線邏輯）
# ─────────────────────────────────────────────────────────────────────

def detect_sb_pullback(
    daily: pd.DataFrame,
    three_d: pd.DataFrame,
    ma_fast: int = 10,
) -> pd.DataFrame:
    """
    極速框成立後，偵測健康回踩的再進場訊號。

    新增欄位：
        sb_pb_monitoring : bool — 回踩監控中
        sb_pb_black_3d   : bool — 3D K 出現黑吞（回踩啟動）
        sb_pb_ma_break   : bool — 日線收破 MA10（回踩失敗警告）
        sb_pb_dead       : bool — 3D K 收破 param0（趨勢破壞）
        sb_pb_entry      : bool — 回踩進場訊號
        sb_pb_entry_type : str  — "黑吞+紅吞" / "3D底底高"
    """
    maf   = f"ma{ma_fast}"
    daily = daily.copy()

    daily["sb_pb_monitoring"] = False
    daily["sb_pb_black_3d"]   = False
    daily["sb_pb_ma_break"]   = False
    daily["sb_pb_dead"]       = False
    daily["sb_pb_entry"]      = False
    daily["sb_pb_entry_type"] = ""

    td = three_d.copy()
    td["body_high"] = td[["open", "close"]].max(axis=1)
    td["body_low"]  = td[["open", "close"]].min(axis=1)
    td["is_black"]  = td["close"] < td["open"]
    td["is_red"]    = td["close"] >= td["open"]

    sb_ok_dates = daily[daily.get("speed_box_ok", False) == True].index.tolist()
    if not sb_ok_dates:
        return daily

    for sb_date in sb_ok_dates:
        before    = daily.loc[:sb_date]
        p0_series = before["sb_param0"].dropna()
        if p0_series.empty:
            continue
        param0 = float(p0_series.iloc[-1])

        td_after = td[td.index > sb_date]
        if td_after.empty:
            continue

        td_list     = td_after.index.tolist()
        in_pullback = False
        dead        = False
        entry_found = False
        lows_in_pb  = []

        for i, td_date in enumerate(td_list):
            if dead or entry_found:
                break

            prev_td_date    = td_list[i - 1] if i > 0 else sb_date
            row             = td.loc[td_date]
            daily_mask      = (daily.index > prev_td_date) & (daily.index <= td_date)
            daily_in_period = daily[daily_mask]

            # 死亡界線
            if float(row["close"]) < param0:
                daily.loc[daily_mask, "sb_pb_dead"] = True
                dead = True
                break

            # 黑吞啟動回踩
            if not in_pullback and bool(row["is_black"]):
                in_pullback = True
                daily.loc[daily_mask, "sb_pb_black_3d"] = True
                lows_in_pb = [float(row["close"])]
                continue

            if not in_pullback:
                continue

            # 回踩監控
            daily.loc[daily_mask, "sb_pb_monitoring"] = True
            lows_in_pb.append(float(row["close"]))

            for d_idx in daily_in_period.index:
                d = daily.loc[d_idx]
                if pd.notna(d.get(maf)) and float(d["close"]) < float(d[maf]):
                    daily.at[d_idx, "sb_pb_ma_break"] = True

            ma_broken_in_pb = (daily.loc[daily_mask, "sb_pb_ma_break"].any()
                               if daily_mask.any() else False)

            # 進場訊號 1：黑吞 + 紅吞
            if i >= 1:
                prev_row = td.loc[td_list[i - 1]]
                if bool(prev_row["is_black"]) and bool(row["is_red"]) and not ma_broken_in_pb:
                    last_daily = daily_in_period.index[-1] if not daily_in_period.empty else None
                    if last_daily is not None:
                        daily.at[last_daily, "sb_pb_entry"]      = True
                        daily.at[last_daily, "sb_pb_entry_type"] = "黑吞+紅吞"
                        entry_found = True
                        break

            # 進場訊號 2：3D 底底高
            if len(lows_in_pb) >= 3 and not ma_broken_in_pb:
                l0, l1, l2 = lows_in_pb[-3], lows_in_pb[-2], lows_in_pb[-1]
                if l2 > l1 > l0:
                    last_daily = daily_in_period.index[-1] if not daily_in_period.empty else None
                    if last_daily is not None:
                        daily.at[last_daily, "sb_pb_entry"]      = True
                        daily.at[last_daily, "sb_pb_entry_type"] = "3D底底高"
                        entry_found = True
                        break

    return daily


# ─────────────────────────────────────────────────────────────────────
# § 聽牌偵測（On-Deck）
# ─────────────────────────────────────────────────────────────────────

def mark_on_deck(df: pd.DataFrame, ma_fast: int = 10, ma_slow: int = 200) -> pd.DataFrame:
    """
    偵測「1012 聽牌」：T-2 + T-1 條件已滿足，明日可能觸發進場。

    新增欄位：
        Is_On_Deck           : bool  — 今日為 T-1（正在聽牌）
        Target_Trigger_Price : float — 明日收盤需超越的最低門檻
    """
    maf = f"ma{ma_fast}"
    mas = f"ma{ma_slow}"
    df  = df.copy()

    c_prev   = df["close"].shift(1)
    maf_prev = df[maf].shift(1)
    bh_prev  = df["body_high"].shift(1)

    cond_t2 = c_prev > maf_prev

    cond_t1 = (
        (df["open"]  > df[maf]) &
        (df["close"] > df[maf]) &
        (df["close"] <= bh_prev)
    )

    on_deck = cond_t2 & cond_t1

    trigger_price = pd.concat(
        [df[maf], df["ma55"], df[mas], df["body_high"]], axis=1
    ).max(axis=1)

    df["Is_On_Deck"]           = on_deck
    df["Target_Trigger_Price"] = np.where(on_deck, trigger_price, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────
# § MTF 多週期共振 + 3D 階梯支撐壓力
# ─────────────────────────────────────────────────────────────────────

def mark_mtf_trend(
    daily: pd.DataFrame,
    three_d: pd.DataFrame,
    weekly: pd.DataFrame,
    ma_fast: int = 10,
) -> pd.DataFrame:
    """
    補充 MTF 共振分數與 3D 階梯支撐壓力線。
    （三週期趨勢布林值已由 _pre_calc_mtf_trends 計算完畢）

    新增 / 補充欄位：
        MTF_Score    : int   — 三燈加總（0~3）
        Prev_3D_High : float — 前一根 3D K 最高點
        Prev_3D_Low  : float — 前一根 3D K 最低點
    """
    df = daily.copy()

    df["MTF_Score"] = (
        df["Daily_Trend_Up"].astype(int) +
        df["3D_Trend_Up"].astype(int) +
        df["Weekly_Trend_Up"].astype(int)
    )

    td_hl = three_d[["high", "low"]].copy()
    td_hl["Prev_3D_High"] = td_hl["high"].shift(1)
    td_hl["Prev_3D_Low"]  = td_hl["low"].shift(1)

    df["Prev_3D_High"] = (
        td_hl["Prev_3D_High"]
        .reindex(df.index, method="ffill")
    )
    df["Prev_3D_Low"] = (
        td_hl["Prev_3D_Low"]
        .reindex(df.index, method="ffill")
    )

    return df


# ─────────────────────────────────────────────────────────────────────
# § 7  今日快照
# ─────────────────────────────────────────────────────────────────────

def get_today_summary(df: pd.DataFrame, ticker: str) -> dict:
    last = df.iloc[-1]

    def _f(key):
        v = last.get(key, np.nan)
        return float(v) if pd.notna(v) else None

    return {
        "ticker": ticker,
        "date":   df.index[-1].strftime("%Y-%m-%d"),
        "close":  float(last["close"]),
        # 核心訊號
        "signal_1012":      bool(last.get("signal_1012",    False)),
        "speed_box_ok":     bool(last.get("speed_box_ok",   False)),
        "speed_box_fail":   bool(last.get("speed_box_fail", False)),
        "sb_stop":          bool(last.get("sb_stop",        False)),
        "weekly_tp":        bool(last.get("weekly_tp",      False)),
        "weekly_sl":        bool(last.get("weekly_sl",      False)),
        "v_reload":         bool(last.get("v_reload",       False)),
        "v_blocked":        bool(last.get("v_blocked",      False)),
        "in_position":      bool(last.get("in_position",    False)),
        "sb_active":        bool(last.get("sb_active",      False)),
        # 回踩訊號
        "sb_pb_monitoring": bool(last.get("sb_pb_monitoring", False)),
        "sb_pb_entry":      bool(last.get("sb_pb_entry",      False)),
        "sb_pb_entry_type": str(last.get("sb_pb_entry_type",  "")),
        "sb_pb_dead":       bool(last.get("sb_pb_dead",       False)),
        "sb_pb_ma_break":   bool(last.get("sb_pb_ma_break",   False)),
        # 關鍵價位
        "entry_price": _f("entry_price"),
        "stop_loss":   _f("stop_loss_price"),
        "sb_param0":   _f("sb_param0"),
        "sb_param1":   _f("sb_param1"),
        "sb_target":   _f("sb_target"),
        # 聽牌
        "is_on_deck":           bool(last.get("Is_On_Deck", False)),
        "target_trigger_price": _f("Target_Trigger_Price"),
        # MTF 共振
        "daily_trend_up":   bool(last.get("Daily_Trend_Up",  False)),
        "three_d_trend_up": bool(last.get("3D_Trend_Up",     False)),
        "weekly_trend_up":  bool(last.get("Weekly_Trend_Up", False)),
        "mtf_score":        int(last.get("MTF_Score", 0)),
        # 快訊文案
        "alert_message": str(last.get("Alert_Message", "")),
    }


# ─────────────────────────────────────────────────────────────────────
# § 8  手動持倉分析（含 R 倍數停利）
# ─────────────────────────────────────────────────────────────────────

def analyze_manual_position(
    df: pd.DataFrame,
    ticker: str,
    entry_price: float,
    shares: int = 1000,
    ma_fast: int = 10,
    sb_ratio: float = 5.8,
) -> dict:
    maf_col = f"ma{ma_fast}"
    last    = df.iloc[-1]
    close   = float(last["close"])
    pnl_pct = (close - entry_price) / entry_price * 100

    sb_p0       = None
    sb_p1       = None
    sb_target   = None
    sb_achieved = False

    signal_rows = df[df["signal_1012"] == True]
    if not signal_rows.empty:
        last_sig_idx = signal_rows.index[-1]
        after_sig    = df.loc[last_sig_idx:]
        sb_rows = after_sig[after_sig["sb_param0"].notna()]
        if not sb_rows.empty:
            sb_p0 = float(sb_rows["sb_param0"].iloc[0])
            sb_p1 = float(sb_rows["sb_param1"].iloc[0]) if sb_rows["sb_param1"].notna().any() else None
            if sb_rows["sb_target"].notna().any():
                sb_target = float(sb_rows["sb_target"].dropna().iloc[0])
        if "speed_box_ok" in after_sig.columns:
            sb_achieved = bool(after_sig["speed_box_ok"].any())

    ma10_now      = float(last[maf_col]) if pd.notna(last.get(maf_col)) else None
    recent_low_5  = float(df["low"].tail(5).min())
    recent_low_10 = float(df["low"].tail(10).min())
    recent_low_20 = float(df["low"].tail(20).min())

    if sb_achieved:
        if ma10_now:
            recommended_stop = round(ma10_now * 0.99, 2)
            stop_basis = f"【移動停損】MA{ma_fast} 下方 1%（極速框已成立）"
        else:
            recommended_stop = round(recent_low_5 * 0.99, 2)
            stop_basis = "【移動停損】近 5 日低點下方 1%"
    elif sb_p0 and sb_p0 < entry_price:
        recommended_stop = round(sb_p0, 2)
        stop_basis = "極速框底線（T-1 最低點）"
    elif ma10_now and ma10_now < entry_price:
        recommended_stop = round(ma10_now * 0.99, 2)
        stop_basis = f"MA{ma_fast} 下方 1% 緩衝"
    else:
        recommended_stop = round(recent_low_10 * 0.99, 2)
        stop_basis = "近 10 日最低點下方 1%"

    R = entry_price - recommended_stop

    targets = {}
    if R > 0:
        for mult in [5, 10, 15, 20]:
            tp = round(entry_price + mult * R, 2)
            achieved_mark = "✅ " if close >= tp else ""
            targets[f"1:{mult}  {achieved_mark}{tp}"] = tp
    else:
        targets["近 20 日高點"] = round(float(df["high"].tail(20).max()), 2)

    if sb_target:
        label = f"極速框 5.8（{'✅已達成' if sb_achieved else '追蹤中'}）{round(sb_target,2)}"
        targets[label] = round(sb_target, 2)

    current_r = round((close - entry_price) / R, 2) if R > 0 else None

    next_tp = None
    if R > 0:
        for mult in [5, 10, 15, 20]:
            tp = entry_price + mult * R
            if close < tp:
                next_tp = tp
                break
    rr_to_next = round((next_tp - entry_price) / R, 2) if next_tp and R > 0 else None

    washout_now   = bool(last.get("weekly_washout", False))
    weekly_sl     = bool(last.get("weekly_sl",      False))
    sb_stop_now   = bool(last.get("sb_stop",        False))
    pb_monitoring = bool(last.get("sb_pb_monitoring", False))
    pb_entry      = bool(last.get("sb_pb_entry",      False))
    pb_dead       = bool(last.get("sb_pb_dead",       False))
    pb_ma_break   = bool(last.get("sb_pb_ma_break",   False))
    pb_entry_type = str(last.get("sb_pb_entry_type",  ""))

    action, level = _suggest_action(
        close, entry_price, recommended_stop,
        washout_now, pnl_pct, weekly_sl, sb_stop_now,
        current_r, pb_entry, pb_dead, pb_ma_break,
    )

    return {
        "ticker":           ticker,
        "entry_price":      entry_price,
        "shares":           shares,
        "close":            close,
        "pnl_pct":          round(pnl_pct, 2),
        "pnl_amount":       round((close - entry_price) * shares, 0),
        "recommended_stop": recommended_stop,
        "stop_basis":       stop_basis,
        "sb_achieved":      sb_achieved,
        "sb_param0":        round(sb_p0, 2) if sb_p0 else None,
        "ma10_now":         round(ma10_now, 2) if ma10_now else None,
        "recent_low_5":     round(recent_low_5, 2),
        "recent_low_10":    round(recent_low_10, 2),
        "recent_low_20":    round(recent_low_20, 2),
        "R":                round(R, 2) if R > 0 else None,
        "targets":          targets,
        "sb_target":        round(sb_target, 2) if sb_target else None,
        "current_r":        current_r,
        "rr_to_next":       rr_to_next,
        "weekly_washout":   washout_now,
        "pb_monitoring":    pb_monitoring,
        "pb_entry":         pb_entry,
        "pb_entry_type":    pb_entry_type,
        "pb_dead":          pb_dead,
        "pb_ma_break":      pb_ma_break,
        "action":           action,
        "action_level":     level,
    }


def _suggest_action(close, entry, stop, washout, pnl,
                    weekly_sl, sb_stop, current_r=None,
                    pb_entry=False, pb_dead=False, pb_ma_break=False):
    if weekly_sl or sb_stop:
        return "⚠️ 策略訊號觸發，建議立即出場", "danger"
    if close < stop:
        return "🔴 已跌破停損線，建議出場", "danger"
    if washout and close <= entry:
        return "🔴 週黑吞確認 + 虧損，建議出場", "danger"
    if washout and close > entry:
        return "🟡 週黑吞確認，考慮週K停利出場", "warning"
    if pb_dead:
        return "💀 3D K 收破 param0，回踩趨勢破壞，停止找買點", "danger"
    if pb_ma_break:
        return "🟠 回踩中 MA10 被收破，回踩失敗，謹慎", "warning"
    if pb_entry:
        return "🟣 極速框回踩進場訊號，可考慮加碼 / 再進場", "success"
    if current_r is not None:
        if current_r >= 15:
            return f"🟢 現處 {current_r}R，可考慮分批停利（1:15~1:20）", "success"
        if current_r >= 10:
            return f"🟢 現處 {current_r}R，持倉追蹤，注意 1:15 目標", "success"
        if current_r >= 5:
            return f"🟢 現處 {current_r}R，已過 1:5，持倉等待下一目標", "success"
        if current_r >= 0:
            return f"🔵 現處 {current_r}R，持倉觀察，目標 1:5", "info"
        if current_r >= -1:
            return f"🟡 現處 {current_r}R，小幅虧損，注意停損線", "warning"
    if pnl >= 0:
        return "🔵 獲利中，持倉觀察", "info"
    if pnl >= -3:
        return "🟡 小幅虧損，注意停損線", "warning"
    return "🔴 虧損擴大，評估是否停損", "danger"
