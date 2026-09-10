import React, { useState } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import { 
  Settings, Upload, Activity, AlertTriangle, CheckCircle, 
  Menu, Factory, Zap, Thermometer, Gauge, ChevronRight, ChevronLeft, BrainCircuit
} from 'lucide-react';
import './App.css';

// --- New Health Gauge Component ---
function HealthGauge({ score }) {
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  
  let color = 'var(--success)';
  if (score < 40) color = 'var(--danger)';
  else if (score < 70) color = 'var(--warning)';

  return (
    <div className="health-gauge-container">
      <svg className="health-gauge-svg" width="160" height="160">
        <circle className="health-gauge-bg" cx="80" cy="80" r={radius} />
        <circle 
          className="health-gauge-fill" 
          cx="80" cy="80" r={radius} 
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          stroke={color}
        />
      </svg>
      <div className="health-value-center">
        <span className="health-percentage">{score.toFixed(0)}%</span>
        <span className="health-label">Holati</span>
      </div>
    </div>
  );
}

function App() {
  // Sidebar State (auto close on mobile)
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);
  
  // Form State
  const [file, setFile] = useState(null);
  const [useDemo, setUseDemo] = useState(true);
  const [forecastYears, setForecastYears] = useState(5);
  const [epochs, setEpochs] = useState(30);
  const [degradation, setDegradation] = useState(0.0003);
  const [eqType, setEqType] = useState('Motor');
  const [vyahh, setVyahh] = useState(7.0);
  const [vxahh, setVxahh] = useState(6.0);

  // App State
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setUseDemo(false);
    }
  };

  const runForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      if (file && !useDemo) {
        formData.append('file', file);
      }
      formData.append('use_demo', useDemo);
      formData.append('forecast_years', forecastYears);
      formData.append('epochs', epochs);
      formData.append('degradation', degradation);
      formData.append('eq_type', eqType);
      formData.append('vyahh', vyahh);
      formData.append('vxahh', vxahh);

      const rawUrl = import.meta.env.VITE_API_URL || '';
      const apiUrl = rawUrl.replace(/\/+$/, '');
      const response = await axios.post(`${apiUrl}/api/forecast`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
      if(window.innerWidth < 768) setSidebarOpen(false); // auto close on mobile
    } catch (err) {
      let message = "Noma'lum xatolik yuz berdi";
      if (err.response?.data?.detail) {
        message = err.response.data.detail;
      } else if (err.message === "Network Error") {
        message = "Serverga ulanib bo'lmadi (Network Error). Backend (Railway/Lokal) manzili to'g'ri kiritilgani va server ishlayotganini tekshiring.";
      } else if (err.message) {
        message = err.message;
      }
      setError(message);
      // 7 soniyadan keyin xato xabarini avtomatik yashirish
      setTimeout(() => setError(null), 7000);
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (!results) return;
    
    // Bug 4 fix: Har bir parametrni alohida ustunda, to'g'ri CSV formatlash
    // Ustun nomlarini quotatsiya ichiga olish (probel yoki maxsus belgilar bo'lsa)
    const escapeCSV = (val) => {
      if (val === undefined || val === null || val === '') return '';
      const str = String(val);
      if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes(' ')) {
        return '"' + str.replace(/"/g, '""') + '"';
      }
      return str;
    };
    
    // Header — har bir parametr alohida ustunda
    const headers = ['Type', 'Time', ...results.feature_cols.map(c => escapeCSV(c))];
    let csv = headers.join(',') + '\n';
    
    // Tarixiy ma'lumotlar
    results.historical_data.forEach(row => {
      const values = [
        'Historical',
        escapeCSV(row.Time || ''),
        ...results.feature_cols.map(c => {
          const val = row[c];
          return (val !== undefined && val !== null && val !== '') ? val : '';
        })
      ];
      csv += values.join(',') + '\n';
    });
    
    // Prognoz ma'lumotlari — faqat ijobiy qiymatlarni eksport qilish
    results.forecast_data.forEach(row => {
      const values = [
        'Forecast',
        escapeCSV(row.Time || ''),
        ...results.feature_cols.map(c => {
          const val = row[c];
          if (val === undefined || val === null || val === '') return '';
          // Manfiy qiymatlarni eksportda ham filtrlash (qo'shimcha himoya)
          return Number(val) < 0 ? 0 : val;
        })
      ];
      csv += values.join(',') + '\n';
    });
    
    // BOM qo'shish (Excel uchun UTF-8 to'g'ri ko'rsatish)
    const BOM = '\uFEFF';
    const blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', `forecast_report_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  // Helper to format chart data
  const chartData = (colName) => {
    if (!results || !results.historical_data || !results.forecast_data) return [];
    
    // Merge historical and forecast data
    const dataMap = new Map();
    
    results.historical_data.forEach((row, i) => {
       const time = row.Time || `Day ${i}`;
       dataMap.set(time, { Time: time, History: row[colName] });
    });

    results.forecast_data.forEach((row, i) => {
       const time = row.Time || `Forecast Day ${i}`;
       if (dataMap.has(time)) {
           dataMap.get(time).Forecast = row[colName];
       } else {
           dataMap.set(time, { Time: time, Forecast: row[colName] });
       }
    });

    const fullData = Array.from(dataMap.values());
    
    // Optimization: If more than 500 points, sample the data for performance
    if (fullData.length > 500) {
      const step = Math.ceil(fullData.length / 500);
      return fullData.filter((_, idx) => idx % step === 0);
    }
    
    return fullData;
  };

  return (
    <div className="app-container">
      {/* Mobile Overlay */}
      <div 
         className={`mobile-overlay ${sidebarOpen ? '' : 'hidden'}`} 
         onClick={() => setSidebarOpen(false)}
      ></div>

      {/* Sidebar Toggle */}
      <div 
        className={`sidebar-toggle ${sidebarOpen ? 'open' : ''}`}
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {sidebarOpen ? <ChevronLeft /> : <Menu />}
      </div>

      {/* Sidebar Configuration */}
      <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '30px', color: '#fff' }}>
          <Settings className="text-gradient" /> Sozlamalar
        </h2>

        <div className="control-group">
          <label><Factory size={16} style={{verticalAlign:'text-bottom', marginRight:'5px'}}/> Ma'lumotlar manbai</label>
          
          <label className="custom-file-upload">
            <Upload size={16} /> {file ? (file.name.length > 20 ? file.name.substring(0, 20)+'...' : file.name) : "Ma'lumot faylini tanlang (.xlsx, .csv)"}
            <input type="file" accept=".xlsx,.xls,.csv" onChange={handleFileChange} style={{display: 'none'}} />
          </label>
          
          <label className="toggle-switch" style={{ marginTop: '20px' }}>
            <input type="checkbox" style={{display: 'none'}} checked={useDemo} onChange={(e) => setUseDemo(e.target.checked)} />
            <span className="switch"></span>
            <span style={{fontSize: '15px', color: '#fff'}}>Demo ma'lumotlardan foydalanish</span>
          </label>
        </div>

        <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '20px 0' }} />

        <div className="control-group">
          <label><AlertTriangle size={16} /> Xavfsizlik chegaralari (AHH)</label>
          <div style={{ marginTop: '10px' }}>
             <small style={{color: 'var(--text-muted)'}}>VYI AHH chegarasi (mm/s)</small>
             <input type="number" step="0.5" value={vyahh} onChange={e => setVyahh(parseFloat(e.target.value))} />
          </div>
          <div style={{ marginTop: '10px' }}>
             <small style={{color: 'var(--text-muted)'}}>VXI AHH chegarasi (mm/s)</small>
             <input type="number" step="0.5" value={vxahh} onChange={e => setVxahh(parseFloat(e.target.value))} />
          </div>
        </div>

        <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '20px 0' }} />

        <div className="control-group">
          <label><BrainCircuit size={16} /> Prognoz parametrlari</label>
          <div style={{ marginTop: '10px' }}>
             <small style={{color: 'var(--text-muted)'}}>Prognoz davri (Yillar)</small>
             <select value={forecastYears} onChange={e => setForecastYears(parseInt(e.target.value))}>
                <option value="3">3 yil</option>
                <option value="5">5 yil</option>
                <option value="7">7 yil</option>
                <option value="10">10 yil</option>
             </select>
          </div>
          <div style={{ marginTop: '15px' }}>
             <small style={{display: 'block', color: 'var(--text-muted)', marginBottom: '10px'}}>O'qitish davrlari ({epochs})</small>
             <input type="range" min="10" max="100" step="10" value={epochs} onChange={e => setEpochs(parseInt(e.target.value))} style={{width:'100%', cursor:'pointer'}} />
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className={`main-content ${sidebarOpen ? 'shifted' : ''}`}>
        <div className="header">
          <div className="project-badge">SANOAT AI v2.3</div>
          <h1 className="text-gradient">
            <BrainCircuit size={48} style={{display:'inline', verticalAlign:'middle', marginRight: '15px'}} /> 
            Sanoat AI Prognozi
          </h1>
          <p className="subtitle">Uskunalar holatini bashorat qilish va monitoring tizimi. Uzoq muddatli tahlil uchun Bidirectional LSTM neyron tarmoqlaridan foydalaniladi.</p>
          <div className="system-status">
             <span className="status-dot"></span> Tizim tayyor
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '40px' }}>
           <button 
             className="btn-primary" 
             style={{ maxWidth: '350px', fontSize: '1.2rem' }}
             onClick={runForecast}
             disabled={loading}
           >
             {loading ? <Activity className="animate-spin" /> : <Zap />}
             {loading ? "Sensorlar tahlil qilinmoqda..." : "AI prognozlashni boshlash"}
           </button>
        </div>

        {error && (
           <div className="alert danger" style={{maxWidth: '600px', margin: '0 auto 30px', animation: 'fadeInDown 0.4s ease-out'}}>
             <AlertTriangle color="var(--danger)" />
             <div style={{flex: 1}}>
               <strong>Xatolik:</strong> {error}
             </div>
             <span 
               onClick={() => setError(null)} 
               style={{cursor: 'pointer', opacity: 0.7, fontSize: '1.2rem', lineHeight: 1}}
             >✕</span>
           </div>
         )}

        {results && (
          <div className="dashboard-results">
            
            {results.is_fallback && (
              <div className="alert warning" style={{marginBottom: '20px'}}>
                <AlertTriangle color="var(--warning)"/>
                <div>
                  <strong>Oddiy rejim (Fallback):</strong> Chuqur o'rganish (LSTM) modeli ishlamadi (TensorFlow topilmadi). Prognoz chiziqli tendentsiya orqali hisoblandi.
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
               <h3 style={{display:'flex', alignItems:'center', gap:'10px'}}>
                  <Activity color="var(--primary-color)"/> Diagnostika sharhi
               </h3>
               <button className="btn-secondary" onClick={downloadCSV}>
                  <Upload size={16} /> Eksport (.csv)
               </button>
            </div>

            <div className="metrics-grid">
               {results.feature_cols.map(col => {
                 const cu = col.toUpperCase();
                 let icon = <Activity size={20}/>;
                 if(cu.startsWith('VYI') || cu.startsWith('VXI') || cu.startsWith('VIB')) icon = <Activity size={20} color="var(--warning)"/>;
                 if(cu === 'T' || cu.includes('TEMP') || cu.includes('HARORAT')) icon = <Thermometer size={20} color="var(--danger)"/>;
                 if(cu.startsWith('SPEED') || cu.startsWith('RPM') || cu.includes('TEZLIK')) icon = <Gauge size={20} color="var(--success)"/>;
                 if(cu.startsWith('CURRENT') || cu.includes('AMPS') || cu.includes('TOK')) icon = <Zap size={20} color="var(--primary-color)"/>;
                 
                 const latest = results.historical_data[results.historical_data.length - 1][col];
                 let unit = '';
                 if(cu.startsWith('VYI') || cu.startsWith('VXI') || cu.startsWith('VIB')) unit = 'mm/s';
                 else if(cu === 'T' || cu.includes('TEMP') || cu.includes('HARORAT')) unit = '°C';
                 else if(cu.startsWith('SPEED') || cu.startsWith('RPM') || cu.includes('TEZLIK')) unit = 'RPM';
                 else if(cu.startsWith('CURRENT') || cu.includes('AMPS') || cu.includes('TOK')) unit = 'A';
                 
                 return (
                   <div className="glass-card" key={col}>
                      <div className="metric-label">{icon} {col}</div>
                      <div className="metric-value">{latest ? latest.toFixed(2) : '-'} <span style={{fontSize:'1rem', color:'var(--text-muted)'}}>{unit}</span></div>
                   </div>
                 );
              })}
            </div>

            <div className="metrics-grid" style={{gridTemplateColumns: '1.2fr 2fr'}}>
                <div className="glass-card" style={{display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', textAlign:'center'}}>
                   <HealthGauge score={results.health_score || 0} />
                </div>

                <div className="glass-card">
                   <h4 style={{marginBottom:'15px', color:'var(--primary-color)', display:'flex', alignItems:'center', gap:'10px'}}>
                      <BrainCircuit size={20}/> AI diagnostika xulosalari
                   </h4>
                   {results.insights.map((insight, idx) => (
                       <div key={idx} className={`alert ${insight.includes('✅') ? 'success' : (insight.includes('🔴') || insight.includes('🚨') ? 'danger' : 'warning')}`}>
                          {insight.includes('✅') && <CheckCircle color="var(--success)"/>}
                          {(insight.includes('🔴') || insight.includes('🚨') || insight.includes('⚠️')) && <AlertTriangle color={insight.includes('🔴') ? "var(--danger)" : "var(--warning)"}/>}
                          {!insight.includes('✅') && !insight.includes('🔴') && !insight.includes('🚨') && !insight.includes('⚠️') && <Activity color="var(--primary-color)"/>}
                          <div>
                             <span dangerouslySetInnerHTML={{__html: insight.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}}></span>
                          </div>
                       </div>
                   ))}
                </div>
            </div>

            <h3 style={{marginTop: '50px', marginBottom: '25px', display:'flex', alignItems:'center', gap:'10px'}}>
               <Activity color="var(--primary-color)"/> Bashoratli prognoz tendentsiyalari
            </h3>
            
            {results.feature_cols.map(col => {
               const data = chartData(col);
               const cu = col.toUpperCase();
               let ahh = null;
               if(cu.startsWith('VYI')) ahh = vyahh;
               if(cu.startsWith('VXI')) ahh = vxahh;

               return (
                 <div className="chart-container" key={col}>
                    <h4 className="chart-title">{col} prognozi ({forecastYears} yillik davr)</h4>
                    <ResponsiveContainer width="100%" height={320} minWidth={0} minHeight={200}>
                        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis 
                            dataKey="Time" 
                            stroke="rgba(255,255,255,0.3)" 
                            tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 11}}
                            minTickGap={60}
                          />
                          <YAxis stroke="rgba(255,255,255,0.3)" tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 11}} />
                          <RechartsTooltip 
                             contentStyle={{backgroundColor: 'rgba(10,15,25,0.95)', border: '1px solid rgba(79, 172, 254, 0.2)', borderRadius: '12px', boxShadow: '0 10px 30px rgba(0,0,0,0.5)'}}
                             itemStyle={{color: '#fff', fontSize: '14px'}}
                          />
                          <Legend wrapperStyle={{paddingTop: '20px'}} />
                          <Line type="monotone" name="Tarixiy o'lchovlar" dataKey="History" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                          <Line type="monotone" name="AI prognoz tendentsiyasi" dataKey="Forecast" stroke="#f97316" strokeWidth={3} strokeDasharray="6 6" dot={false} />
                          {ahh && <ReferenceLine y={ahh} label={{ value: `Chegara: ${ahh}`, fill: '#ef4444', position: 'insideTopRight' }} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={2} />}
                        </LineChart>
                    </ResponsiveContainer>
                 </div>
               )
            })}

          </div>
        )}

      </div>
    </div>
  );
}

export default App;
