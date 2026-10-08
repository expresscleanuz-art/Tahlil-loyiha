"""
AI Engine - PyTorch Bidirectional LSTM (BiLSTM) Model & Data Processing for Predictive Maintenance
Multivariate deep learning architecture for industrial equipment health forecasting
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

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
# Phase 2: Predictive Deep Learning Model (PyTorch BiLSTM)
# ============================================================



if TORCH_AVAILABLE:
    class BiLSTMForecaster(nn.Module):
        """
        Bidirectional LSTM (BiLSTM) Deep Neural Network architecture.
        Processes multivariate time-series sequences forward and backward
        to model complex sensor correlations and degrade trajectory patterns.
        """
        def __init__(self, input_size, hidden_size=48, num_layers=2, dropout=0.2):
            super(BiLSTMForecaster, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size)
            )

        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
            out, _ = self.lstm(x)
            # Oxirgi vaqt qadami holati
            last_out = out[:, -1, :]
            return self.fc(last_out)


def _train_bilstm(scaled_data, seq_length, epochs, n_features):
    """PyTorch BiLSTM modelini o'qitish"""
    n_samples = len(scaled_data)
    X, y = [], []
    for i in range(seq_length, n_samples):
        X.append(scaled_data[i - seq_length:i])
        y.append(scaled_data[i])
        
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    
    torch.manual_seed(42)
    model = BiLSTMForecaster(input_size=n_features, hidden_size=48, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Interfeysdan tanlangan aniq epochs davrida o'qitish
    model.train()
    history = []
    actual_epochs = max(5, int(epochs))
    for epoch in range(actual_epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        
    return model, history


def train_and_forecast(df, feature_cols, forecast_years=10, seq_length=30, epochs=30, degradation_factor=0.0003, original_last_date=None):
    """
    Train PyTorch Bidirectional LSTM (BiLSTM) model on sensor time-series data
    using user-specified epochs and generate multi-year predictive forecast.
    Falls back gracefully to linear trend if PyTorch is unavailable.
    Returns: forecast_df, scaler, training_history, is_fallback
    """
    forecast_years = int(getattr(forecast_years, 'default', forecast_years))
    epochs = int(getattr(epochs, 'default', epochs))
    degradation_factor = float(getattr(degradation_factor, 'default', degradation_factor))

    data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    n_samples, n_features = scaled.shape
    
    seq_length = min(seq_length, max(5, n_samples // 3))
    forecast_days = int(forecast_years * 365)
    
    # PyTorch BiLSTM orqali o'qitish va bashorat
    if TORCH_AVAILABLE and n_samples >= seq_length + 5:
        try:
            model, history = _train_bilstm(scaled, seq_length, epochs, n_features)
            
            # Ko'p qadamli avtoregressiv bashorat (Autoregressive Rollout)
            model.eval()
            current_window = scaled[-seq_length:].copy()
            predictions = []
            
            with torch.no_grad():
                for step in range(forecast_days):
                    inp = torch.tensor(current_window.reshape(1, seq_length, n_features), dtype=torch.float32)
                    next_pred = model(inp).numpy()[0]
                    
                    # Mexanik eskirish (degradation prior) koeffitsiyenti
                    if degradation_factor > 0:
                        drift = 1.0 + degradation_factor * (step / 365.0)
                        next_pred = next_pred * drift
                    
                    next_pred = np.clip(next_pred, 0.0, 2.0)
                    predictions.append(next_pred)
                    
                    # Darchani keyingi qadamga siljitish
                    current_window = np.vstack([current_window[1:], next_pred])
            
            forecast_scaled = np.array(predictions)
            forecast_vals = scaler.inverse_transform(forecast_scaled)
            forecast_vals = np.maximum(forecast_vals, 0.0)
            
            if original_last_date is not None:
                last_date = pd.to_datetime(original_last_date)
            elif 'Time' in df.columns:
                last_date = pd.to_datetime(df['Time'].iloc[-1])
            else:
                last_date = pd.Timestamp.now()
                
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
            forecast_df = pd.DataFrame(forecast_vals, columns=feature_cols, index=forecast_dates)
            forecast_df.index.name = 'Time'
            
            logger.info(f"BiLSTM training complete: {len(history)} epochs, final loss={history[-1]:.6f}")
            return forecast_df, scaler, history, False
            
        except Exception as e:
            logger.warning(f"BiLSTM execution failed, switching to fallback: {e}")
            
    # Fallback rejim (agar torch bo'lmasa yoki xatolik yuz bersa)
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    
    X, y = [], []
    for i in range(seq_length, n_samples):
        X.append(scaled[i - seq_length:i].flatten())
        y.append(scaled[i])
        
    if len(X) < 5:
        return None, scaler, None, True
        
    model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
    model.fit(np.array(X), np.array(y))
    
    current_window = scaled[-seq_length:].copy()
    predictions = []
    for step in range(forecast_days):
        x_in = current_window.flatten().reshape(1, -1)
        next_pred = model.predict(x_in)[0]
        if degradation_factor > 0:
            next_pred = next_pred * (1.0 + degradation_factor * (step / 365.0))
        next_pred = np.clip(next_pred, 0.0, 2.0)
        predictions.append(next_pred)
        current_window = np.vstack([current_window[1:], next_pred])
        
    forecast_scaled = np.array(predictions)
    forecast_vals = scaler.inverse_transform(forecast_scaled)
    forecast_vals = np.maximum(forecast_vals, 0.0)
    
    if original_last_date is not None:
        last_date = pd.to_datetime(original_last_date)
    elif 'Time' in df.columns:
        last_date = pd.to_datetime(df['Time'].iloc[-1])
    else:
        last_date = pd.Timestamp.now()
        
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame(forecast_vals, columns=feature_cols, index=forecast_dates)
    forecast_df.index.name = 'Time'
    
    return forecast_df, scaler, None, True



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
