from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import data_fetcher
import strategy

app = FastAPI()

class ScanRequest(BaseModel):
    tickers: List[str]

@app.post("/scan")
def scan_stocks(req: ScanRequest):
    reports = []
    for t in req.tickers:
        try:
            # 抓取資料
            data = data_fetcher.get_all_timeframes(t)
            # 執行策略
            df = strategy.run_strategy(data["daily"], data["weekly"], three_d=data["3d"])
            summ = strategy.get_today_summary(df, data["ticker"])
            
            # 根據狀態組合訊息
            status_text = f"【{summ['ticker']}】\n"
            if summ['signal_1012']:
                status_text += f"🟢 1012 進場確認！\n快訊：{summ['alert_message']}"
            elif summ['is_on_deck']:
                status_text += f"👀 聽牌雷達中！明日觸發價：{summ['target_trigger_price']}"
            elif summ['sb_pb_entry']:
                status_text += f"🟣 極速框回踩進場！型態：{summ['sb_pb_entry_type']}"
            elif summ['sb_stop'] or summ['weekly_sl']:
                status_text += f"🔴 觸發停損，建議出場！"
            else:
                status_text += "⚪ 目前處於觀察區間，無特別訊號。"
            
            reports.append(status_text)
        except Exception as e:
            reports.append(f"❌ {t} 處理失敗: {str(e)}")
            
    return {"message": "\n\n".join(reports)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)