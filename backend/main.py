from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import os

try:
    from backend.ai_engine import (
        clean_data, generate_synthetic_data, create_sample_data,
        train_and_forecast, calculate_rul, calculate_health_score,
        generate_ai_insights
    )
except ImportError:
    from ai_engine import (
        clean_data, generate_synthetic_data, create_sample_data,
        train_and_forecast, calculate_rul, calculate_health_score,
        generate_ai_insights
    )

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Industrial AI API", version="1.0.0")

# Configure CORS — ruxsat berilgan domenlar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Barcha domenlarga ruxsat berish (Vercel uchun muammosiz)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

import sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Frontend static files mounting (agar dist papka mavjud bo'lsa)
base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dist_dir = os.path.join(base_dir, "frontend", "dist")

import json

CONFIG_FILE = "config.json"

def get_app_config():
    default_config = {
        "app_name": "Kiber AI",
        "badge": "KIBER AI v2.3",
        "subtitle": "Uskunalar holatini bashorat qilish va monitoring tizimi. Uzoq muddatli tahlil uchun Bidirectional LSTM neyron tarmoqlaridan foydalaniladi.",
        "page_title": "Kiber AI Prognozi | Bashoratli monitoring tizimi"
    }
    
    paths_to_check = [
        os.path.join(os.getcwd(), CONFIG_FILE),
        os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd(), CONFIG_FILE),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_FILE),
        os.path.join(base_dir, CONFIG_FILE)
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
    return default_config

@app.get("/api/config")
def read_config():
    return get_app_config()

if os.path.exists(dist_dir) and os.path.exists(os.path.join(dist_dir, "index.html")):
    assets_dir = os.path.join(dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_spa_root():
        return FileResponse(os.path.join(dist_dir, "index.html"))
else:
    @app.get("/")
    def read_root():
        return {"status": "ok", "message": "Industrial AI API is running"}

@app.post("/api/forecast")
async def run_forecast(
    file: UploadFile = File(None),
    use_demo: bool = Form(True),
    forecast_years: int = Form(5),
    epochs: int = Form(30),
    degradation: float = Form(0.0003),
    eq_type: str = Form("Motor"),
    vyahh: float = Form(7.0),
    vxahh: float = Form(6.0)
):
    try:
        df = None
        if file is not None and file.filename:
            contents = await file.read()
            if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                df = pd.read_excel(io.BytesIO(contents))
            elif file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            else:
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload Excel or CSV.")
        
        if df is None and use_demo:
            df = create_sample_data()
            
        if df is None:
            raise HTTPException(status_code=400, detail="No data provided")

        # Clean data
        df = clean_data(df)
        
        # Identify feature columns (Improved for real-world robustness)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_cols = []
        # Bug 3 fix: Aniqroq parametr moslashtirish logikasi
        # AHH (alarm) ustunlarini avval ajratib olamiz — ular sensor emas, chegara qiymatlari
        ahh_keywords = ['AHH', 'ALARM', 'LIMIT', 'THRESHOLD', 'CHEGARA']
        sensor_cols = []
        for col in numeric_cols:
            col_upper = col.upper()
            if any(kw in col_upper for kw in ahh_keywords):
                continue  # Bu AHH/alarm ustuni, sensor emas — o'tkazib yuboramiz
            sensor_cols.append(col)
        
        # Mapping for common industrial sensor names
        mapping = {
            'VYI': ['VYI', 'VIB_Y', 'VIBRATION_Y'],
            'VXI': ['VXI', 'VIB_X', 'VIBRATION_X'],
            'T': ['TEMP', 'TEMPERATURE', 'HARORAT'],
            'Speed': ['SPEED', 'RPM', 'TEZLIK', 'FREQUENCY'],
            'Current': ['CURRENT', 'AMPS', 'TOK']
        }
        
        for key, aliases in mapping.items():
            for col in sensor_cols:
                col_upper = col.upper()
                # Ustun nomining boshlanishida alias bormi tekshirish (prefiks matching)
                matched = False
                for alias in aliases:
                    if col_upper.startswith(alias) or col_upper == alias:
                        matched = True
                        break
                if matched:
                    feature_cols.append(col)
                    break
        
        # 'T' uchun alohida tekshirish (qisqa nom, faqat to'liq mos kelsa)
        if not any('TEMP' in c.upper() or 'HARORAT' in c.upper() for c in feature_cols):
            for col in sensor_cols:
                col_stripped = col.strip().upper()
                if col_stripped == 'T':
                    feature_cols.append(col)
                    break
        
        # Fallback if no standard names found — barcha sensor ustunlarini olish
        if not feature_cols:
            feature_cols = sensor_cols[:5]
        
        # Ensure unique columns
        feature_cols = list(dict.fromkeys(feature_cols))

        # Bug 1 fix: Asl tarixiy dataning oxirgi sanasini saqlaymiz
        original_last_date = None
        if 'Time' in df.columns:
            original_last_date = pd.to_datetime(df['Time'].iloc[-1])

        # Generate forecast
        # FIX: Use the actual forecast_years parameter for synthetic data extension
        total_target_days = len(df) + (forecast_years * 365)
        extended_df = generate_synthetic_data(df, target_days=total_target_days, degradation_factor=degradation)
        
        forecast_df, _, _, is_fallback = train_and_forecast(
            extended_df, feature_cols, forecast_years=forecast_years,
            seq_length=min(30, len(extended_df) // 3), epochs=epochs,
            original_last_date=original_last_date
        )

        # Health metrics
        latest = df[feature_cols].iloc[-1]
        thresholds = [vyahh if 'VY' in c.upper() else (vxahh if 'VX' in c.upper() else 100) for c in feature_cols]
        health_score = calculate_health_score(latest.values, thresholds)

        rul_vy_days, rul_vx_days = None, None
        if forecast_df is not None:
            # Ustun nomlarini prefiks bo'yicha topish (masalan, 'VYI-****** mm/s')
            vy_col = next((c for c in forecast_df.columns if c.upper().startswith('VYI')), None)
            vx_col = next((c for c in forecast_df.columns if c.upper().startswith('VXI')), None)
            if vy_col:
                rul_vy_days, _ = calculate_rul(forecast_df, vy_col, vyahh)
            if vx_col:
                rul_vx_days, _ = calculate_rul(forecast_df, vx_col, vxahh)

        insights = generate_ai_insights(health_score, rul_vy_days, rul_vx_days, forecast_df, feature_cols, eq_type)

        # Formatting response
        if 'Time' in df.columns:
            df['Time'] = df['Time'].astype(str)
        historical_data = df.fillna("").to_dict(orient='records')
        
        forecast_data = []
        if forecast_df is not None:
            forecast_df_reset = forecast_df.reset_index()
            if 'Time' not in forecast_df_reset.columns and 'index' in forecast_df_reset.columns:
                forecast_df_reset.rename(columns={'index': 'Time'}, inplace=True)
            forecast_df_reset['Time'] = forecast_df_reset['Time'].astype(str)
            forecast_data = forecast_df_reset.fillna("").to_dict(orient='records')

        return {
            "status": "success",
            "health_score": float(health_score),
            "rul_vy_days": int(rul_vy_days) if rul_vy_days is not None else None,
            "rul_vx_days": int(rul_vx_days) if rul_vx_days is not None else None,
            "insights": insights,
            "historical_data": historical_data,
            "forecast_data": forecast_data,
            "feature_cols": feature_cols,
            "forecast_horizon": forecast_years,
            "eq_type": eq_type,
            "is_fallback": is_fallback
        }

    except Exception as e:
        logger.error(f"Error in /api/forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
