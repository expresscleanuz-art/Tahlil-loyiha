"""
AI Engine - LSTM Model & Data Processing for Predictive Maintenance
Multivariate LSTM architecture for industrial equipment health forecasting
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Phase 1: Data Engineering
# ============================================================

def clean_data(df):
    """Handle missing values with interpolation and trend-based filling.
    Also parses Time column to proper datetime format."""
    df = df.copy()
    
    # Time ustunini datetime ga o'tkazish (turli formatlarni qo'llab-quvvatlash)
    if 'Time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Time']):
        try:
            # dayfirst=True: Yevropa formati (kun.oy.yil) uchun — real CSV fayllar shu formatda
            df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
        except Exception:
            try:
                # Fallback: mixed format
                df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=True)
            except Exception:
                pass  # Time parse bo'lmasa, string sifatida qoldirish
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            df[col] = df[col].ffill().bfill()
    return df


def generate_synthetic_data(df, target_days=3650, degradation_factor=0.0003):
    """
    Extend historical data with synthetic degradation trends.
    Simulates mechanical wear and tear over time.
    """
    n_existing = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_needed = max(0, target_days - n_existing)
    if n_needed == 0:
        return df

    last_vals = df[numeric_cols].iloc[-1].values
    trends = np.zeros(len(numeric_cols))
    if n_existing > 10:
        window = min(50, n_existing // 2)
        early = df[numeric_cols].iloc[:window].mean().values
        late = df[numeric_cols].iloc[-window:].mean().values
        trends = (late - early) / n_existing

    synthetic_rows = []
    for i in range(1, n_needed + 1):
        deg = 1 + degradation_factor * (i / 365)
        noise = np.random.normal(0, 0.02, len(numeric_cols))
        new_vals = last_vals + trends * i * deg + noise * np.abs(last_vals + 0.01)
        synthetic_rows.append(new_vals)

    syn_df = pd.DataFrame(synthetic_rows, columns=numeric_cols)

    if 'Time' in df.columns:
        last_time = pd.to_datetime(df['Time'].iloc[-1])
        syn_df['Time'] = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=n_needed, freq='D')
    elif df.index.name == 'Time':
        last_time = df.index[-1]
        syn_df.index = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=n_needed, freq='D')
        syn_df.index.name = 'Time'

    return pd.concat([df, syn_df], ignore_index=True)


def create_sample_data():
    """Generate sample vibration analytics data for demo purposes"""
    np.random.seed(42)
    days = 730  # 2 years of daily data
    time_idx = pd.date_range(start='2023-01-01', periods=days, freq='D')
    t = np.arange(days)

    # Simulate gradual degradation with seasonal variation
    base_vib_y = 2.5 + 0.002 * t + 0.3 * np.sin(2 * np.pi * t / 365)
    base_vib_x = 2.0 + 0.0015 * t + 0.25 * np.sin(2 * np.pi * t / 365 + 0.5)
    speed = 3000 - 0.5 * t + np.random.normal(0, 15, days)
    temp = 45 + 0.01 * t + 5 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 1.5, days)
    current = 15 + 0.005 * t + np.random.normal(0, 0.5, days)

    df = pd.DataFrame({
        'Time': time_idx,
        'VYI': base_vib_y + np.random.normal(0, 0.15, days),
        'VXI': base_vib_x + np.random.normal(0, 0.12, days),
        'Speed': speed,
        'T': temp,
        'Current': current
    })
    return df


# ============================================================
# Phase 2: Predictive Model (GradientBoosting + Sliding Window)
# ============================================================

def _create_lag_features(data, seq_length=30):
    """
    Convert time series to supervised learning with lag features.
    Creates rolling statistics (mean, std, min, max) over multiple windows.
    """
    from sklearn.preprocessing import MinMaxScaler
    
    n_samples, n_features = data.shape
    X, y = [], []
    
    for i in range(seq_length, n_samples):
        window = data[i - seq_length:i]
        features = []
        
        # Raw lag values (last 5 steps)
        for lag in [1, 3, 5, 10, seq_length]:
            if lag <= seq_length:
                features.extend(data[i - lag].tolist())
        
        # Rolling statistics over different windows
        for w in [7, 14, seq_length]:
            w_data = data[max(0, i - w):i]
            features.extend(np.mean(w_data, axis=0).tolist())
            features.extend(np.std(w_data, axis=0).tolist())
            features.extend(np.max(w_data, axis=0).tolist())
            features.extend(np.min(w_data, axis=0).tolist())
        
        # Trend: difference between recent and older values
        mid = seq_length // 2
        recent = np.mean(data[i - mid:i], axis=0)
        older = np.mean(data[i - seq_length:i - mid], axis=0)
        features.extend((recent - older).tolist())
        
        # Position/time index (normalized)
        features.append(i / n_samples)
        
        X.append(features)
        y.append(data[i].tolist())
    
    return np.array(X), np.array(y)


def train_and_forecast(df, feature_cols, forecast_years=10, seq_length=30, epochs=30, original_last_date=None):
    """
    Train GradientBoosting model with sliding window features and generate multi-year forecast.
    Uses scikit-learn — no TensorFlow required.
    Returns: forecast_df, scaler, training_history, is_fallback
    """
    from sklearn.multioutput import MultiOutputRegressor
    
    from sklearn.linear_model import Ridge
    
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y = _create_lag_features(scaled, seq_length)
    if len(X) < 20:
        return None, scaler, None, False
    
    # Train fast Ridge regression model
    model = MultiOutputRegressor(
        Ridge(alpha=1.0, random_state=42)
    )
    
    model.fit(X, y)
    
    # Generate forecast iteratively
    forecast_days = forecast_years * 365
    # Keep a rolling buffer of recent scaled data
    buffer = scaled.copy()
    predictions = []
    
    for step in range(forecast_days):
        n_buf = len(buffer)
        window = buffer[max(0, n_buf - seq_length):n_buf]
        
        features = []
        
        # Lag values
        for lag in [1, 3, 5, 10, seq_length]:
            idx = min(lag, len(window))
            features.extend(window[-idx].tolist())
        
        # Rolling statistics
        for w in [7, 14, seq_length]:
            w_data = window[max(0, len(window) - w):]
            features.extend(np.mean(w_data, axis=0).tolist())
            features.extend(np.std(w_data, axis=0).tolist())
            features.extend(np.max(w_data, axis=0).tolist())
            features.extend(np.min(w_data, axis=0).tolist())
        
        # Trend
        mid = len(window) // 2
        if mid > 0:
            recent = np.mean(window[mid:], axis=0)
            older = np.mean(window[:mid], axis=0)
            features.extend((recent - older).tolist())
        else:
            features.extend(np.zeros(len(feature_cols)).tolist())
        
        # Normalized time position (extrapolating beyond training)
        features.append((len(scaled) + step) / len(scaled))
        
        X_pred = np.array([features])
        pred = model.predict(X_pred)[0]
        
        # Clip predictions to reasonable range
        pred = np.clip(pred, 0.0, 1.5)
        
        predictions.append(pred)
        buffer = np.vstack([buffer, pred.reshape(1, -1)])
        
        # Keep buffer manageable (last 2x seq_length)
        if len(buffer) > seq_length * 2:
            buffer = buffer[-seq_length * 2:]
    
    forecast_scaled = np.array(predictions)
    forecast_vals = scaler.inverse_transform(forecast_scaled)
    
    # Bug 2 fix: Manfiy qiymatlarni 0 ga almashtirish (inverse_transform dan keyin himoya)
    forecast_vals = np.maximum(forecast_vals, 0.0)
    
    # Bug 1 fix: Prognoz sanalarini asl tarixiy dataning oxirgi sanasidan boshlash
    if original_last_date is not None:
        last_date = pd.to_datetime(original_last_date)
    elif 'Time' in df.columns:
        last_date = pd.to_datetime(df['Time'].iloc[-1])
    else:
        last_date = pd.Timestamp.now()
    
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame(forecast_vals, columns=feature_cols, index=forecast_dates)
    forecast_df.index.name = 'Time'
    
    return forecast_df, scaler, None, False



# ============================================================
# RUL & Health Calculations
# ============================================================

def calculate_rul(forecast_df, col, threshold):
    """Calculate Remaining Useful Life - when signal crosses threshold"""
    if forecast_df is None:
        return None, None
    vals = forecast_df[col].values
    crossings = np.where(vals >= threshold)[0]
    if len(crossings) > 0:
        rul_days = crossings[0]
        rul_date = forecast_df.index[crossings[0]]
        return rul_days, rul_date
    return None, None


def calculate_health_score(current_vals, thresholds):
    """
    Calculate equipment health score (0-100%).
    Based on how close current readings are to critical thresholds.
    """
    scores = []
    for val, thresh in zip(current_vals, thresholds):
        if thresh > 0:
            ratio = val / thresh
            score = max(0, min(100, (1 - ratio) * 100 + 20))
            scores.append(score)
    return np.mean(scores) if scores else 50.0


def generate_ai_insights(health_score, rul_days_vy, rul_days_vx, forecast_df, feature_cols, eq_type="Motor"):
    """Generate human-readable AI diagnostic report"""
    insights = []

    # Health status
    if health_score >= 80:
        insights.append(f"✅ **{eq_type} holati: YAXSHI** — Uskuna normal parametrlarda ishlayapti.")
    elif health_score >= 60:
        insights.append(f"⚠️ **{eq_type} holati: EHTIYOT** — Dastlabki eskirish belgilari aniqlandi.")
    elif health_score >= 40:
        insights.append(f"🔶 **{eq_type} holati: OGOHLANTIRISH** — Sezilarli eskirish aniqlandi. Texnik xizmatni rejalashtiring.")
    else:
        insights.append(f"🔴 **{eq_type} holati: KRITIK** — Darhol tekshiruv talab qilinadi!")

    # RUL insights
    if rul_days_vy is not None:
        years = rul_days_vy / 365
        if years < 1:
            insights.append(f"🚨 **Y-o'qi tebranishi** **{rul_days_vy} kundan** so'ng ({years:.1f} yil) xavfli chegaraga yetadi. Zudlik bilan podshipniklarni tekshirish kerak.")
        elif years < 3:
            q = ((rul_days_vy % 365) // 91) + 1
            y = pd.Timestamp.now().year + int(years)
            insights.append(f"⚠️ **Y-o'qi tebranishi** **{y}-yil {q}-choragida** xavfli chegaraga yetishi kutilmoqda. Podshipniklarni almashtirishni rejalashtiring.")
        else:
            insights.append(f"📊 **Y-o'qi tebranishi** keyingi **{years:.1f} yil** davomida me'yorda bo'lishi kutilmoqda.")
    else:
        insights.append("📊 **Y-o'qi tebranishi** — Prognoz davrida xavfli chegaradan o'tish kutilmayapti.")

    if rul_days_vx is not None:
        years = rul_days_vx / 365
        if years < 2:
            insights.append(f"⚠️ **X-o'qi tebranishi** **{years:.1f} yilda** xavfli chegaraga yaqinlashadi. Doimiy nazorat qiling.")

    # Trend analysis — ustun nomlarini prefiks bo'yicha topish
    if forecast_df is not None:
        temp_col = next((c for c in feature_cols if c.upper().startswith('T') and c in forecast_df.columns and 'VY' not in c.upper() and 'VX' not in c.upper() and 'TOK' not in c.upper()), None)
        if temp_col is None:
            temp_col = next((c for c in feature_cols if any(kw in c.upper() for kw in ['TEMP', 'HARORAT']) and c in forecast_df.columns), None)
        if temp_col is not None:
            temp_trend = forecast_df[temp_col].iloc[-1] - forecast_df[temp_col].iloc[0]
            if temp_trend > 10:
                insights.append("🌡️ **Harorat** o'sish tendensiyasini ko'rsatmoqda. Sovutish tizimi va moylashni tekshiring.")

        curr_col = next((c for c in feature_cols if any(kw in c.upper() for kw in ['CURRENT', 'AMPS', 'TOK']) and c in forecast_df.columns), None)
        if curr_col is not None:
            curr_trend = forecast_df[curr_col].iloc[-1] - forecast_df[curr_col].iloc[0]
            if curr_trend > 5:
                insights.append("⚡ **Tok sarfi** oshayapti — motor cho'lg'amida yoki mexanik yuklamada muammo bo'lishi mumkin.")

    # Maintenance recommendations
    if health_score < 70:
        insights.append("\n**🔧 Tavsiya etiladigan choralar:**")
        insights.append("1. Tebranish spektrini chuqur tahlil qilish")
        insights.append("2. Podshipnik holati va moylanishini tekshirish")
        insights.append("3. Valning markazlashuvi va balansini tekshirish")
        insights.append("4. Dvigatelning tok sarfidagi o'zgarishlarni tekshirish")

    return insights
