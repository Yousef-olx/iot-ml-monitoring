import { useState, useEffect } from 'react';
import Head from 'next/head';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [stats, setStats] = useState({ totalSensors: 0, totalPredictions: 0, highRiskEquipment: 0, unreadAlerts: 0 });
  const [alerts, setAlerts] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [realtimeData, setRealtimeData] = useState(null);
  const [nextEquipmentId, setNextEquipmentId] = useState(1);
  const [predictionsLimit, setPredictionsLimit] = useState(10);
  const [lastPredictionUpdate, setLastPredictionUpdate] = useState(null);
  const [formData, setFormData] = useState({
    equipmentId: '1',
    temperature: '70',
    vibration: '1.5',
    pressure: '35',
    humidity: '50'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadDashboardData();
    loadNextEquipmentId();
    loadPredictions();
    const interval = setInterval(() => {
      loadDashboardData();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    loadPredictions();
  }, [predictionsLimit]);

  const loadNextEquipmentId = async () => {
    try {
      const response = await fetch('/api/sensors/next-id');
      if (response.ok) {
        const data = await response.json();
        const nextId = data.nextId || 1;
        setNextEquipmentId(nextId);
        setFormData(prev => ({ ...prev, equipmentId: String(nextId) }));
      }
    } catch (error) {
      console.error('Error loading next equipment ID:', error);
    }
  };

  const loadDashboardData = async () => {
    try {
      const [statsRes, alertsRes, realtimeRes] = await Promise.all([
        fetch('/api/dashboard/stats'),
        fetch('/api/alerts?limit=5'),
        fetch('/api/sensors/latest')
      ]);

      if (statsRes.ok) setStats((await statsRes.json()).stats);
      if (alertsRes.ok) setAlerts((await alertsRes.json()).data);
      if (realtimeRes.ok) setRealtimeData((await realtimeRes.json()).data);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const loadPredictions = async () => {
    try {
      const limit = predictionsLimit === 'all' ? 10000 : predictionsLimit;
      const response = await fetch(`/api/predictions?limit=${limit}`);
      if (response.ok) {
        const data = await response.json();
        setPredictions(data.data);
        setLastPredictionUpdate(new Date());
      }
    } catch (error) {
      console.error('Error loading predictions:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPredictionResult(null);

    try {
      const response = await fetch('/api/sensors/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const result = await response.json();

      if (result.success && result.prediction) {
        setPredictionResult(result.prediction);
        loadDashboardData();
        loadPredictions(); // Refresh predictions list
        // Auto-increment equipment ID for next submission
        const currentId = parseInt(formData.equipmentId);
        const newId = currentId + 1;
        setNextEquipmentId(newId);
        setFormData(prev => ({ ...prev, equipmentId: String(newId) }));
      } else {
        alert('Error: ' + (result.error || 'Unknown error'));
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    loadDashboardData();
    loadNextEquipmentId();
    setPredictionResult(null);
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>IoT ML Monitoring - Next.js</title>
        <meta name="description" content="Real-time IoT monitoring with ML predictions" />
      </Head>

      <header className={styles.header}>
        <h1>🔬 IoT Equipment Monitoring Dashboard</h1>
        <p>Real-time Machine Learning Predictions with XGBoost</p>
        <button onClick={handleRefresh} className={styles.refreshBtn} title="Refresh Data">
          🔄 Refresh
        </button>
      </header>

      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <h3>Total Sensors</h3>
          <div className={styles.value}>{stats.totalSensors}</div>
        </div>
        <div className={styles.statCard}>
          <h3>Predictions</h3>
          <div className={styles.value}>{stats.totalPredictions}</div>
        </div>
        <div className={styles.statCard}>
          <h3>High Risk</h3>
          <div className={`${styles.value} ${styles.danger}`}>{stats.highRiskEquipment}</div>
        </div>
        <div className={styles.statCard}>
          <h3>Unread Alerts</h3>
          <div className={`${styles.value} ${styles.warning}`}>{stats.unreadAlerts}</div>
        </div>
      </div>

      {/* Real-time Sensor Data Display */}
      {realtimeData && (
        <div className={styles.realtimeSection}>
          <h2>📡 Real-Time Sensor Data</h2>
          <div className={styles.realtimeGrid}>
            <div className={styles.realtimeCard}>
              <div className={styles.realtimeIcon}>🌡️</div>
              <div className={styles.realtimeLabel}>Temperature</div>
              <div className={styles.realtimeValue}>{realtimeData.Temperature ? Number(realtimeData.Temperature).toFixed(1) : 'N/A'}°C</div>
            </div>
            <div className={styles.realtimeCard}>
              <div className={styles.realtimeIcon}>📳</div>
              <div className={styles.realtimeLabel}>Vibration</div>
              <div className={styles.realtimeValue}>{realtimeData.Vibration ? Number(realtimeData.Vibration).toFixed(2) : 'N/A'}</div>
            </div>
            <div className={styles.realtimeCard}>
              <div className={styles.realtimeIcon}>💨</div>
              <div className={styles.realtimeLabel}>Pressure</div>
              <div className={styles.realtimeValue}>{realtimeData.Pressure ? Number(realtimeData.Pressure).toFixed(1) : 'N/A'}</div>
            </div>
            <div className={styles.realtimeCard}>
              <div className={styles.realtimeIcon}>💧</div>
              <div className={styles.realtimeLabel}>Humidity</div>
              <div className={styles.realtimeValue}>{realtimeData.Humidity ? Number(realtimeData.Humidity).toFixed(1) : 'N/A'}%</div>
            </div>
          </div>
          <div className={styles.realtimeInfo}>
            <span>Equipment ID: <strong>{realtimeData.EquipmentID}</strong></span>
            <span>Last Update: <strong>{new Date(realtimeData.Timestamp).toLocaleString()}</strong></span>
            {realtimeData.RiskLevel && (
              <span className={styles[`risk${realtimeData.RiskLevel}`]}>
                Risk: <strong>{realtimeData.RiskLevel}</strong> ({(Number(realtimeData.FailureRisk) * 100).toFixed(2)}%)
              </span>
            )}
          </div>
        </div>
      )}

      <div className={styles.grid2}>
        <div className={styles.card}>
          <h2>📊 Add Sensor Reading</h2>
          <div className={styles.nextIdIndicator}>
            Next Equipment ID: <strong>{nextEquipmentId}</strong>
          </div>
          <form onSubmit={handleSubmit}>
            <div className={styles.formGroup}>
              <label>Equipment ID:</label>
              <input type="text" value={formData.equipmentId} onChange={(e) => setFormData({...formData, equipmentId: e.target.value})} required />
            </div>
            <div className={styles.formGroup}>
              <label>Temperature (°C):</label>
              <input type="number" step="0.1" value={formData.temperature} onChange={(e) => setFormData({...formData, temperature: e.target.value})} required />
            </div>
            <div className={styles.formGroup}>
              <label>Vibration:</label>
              <input type="number" step="0.1" value={formData.vibration} onChange={(e) => setFormData({...formData, vibration: e.target.value})} required />
            </div>
            <div className={styles.formGroup}>
              <label>Pressure:</label>
              <input type="number" step="0.1" value={formData.pressure} onChange={(e) => setFormData({...formData, pressure: e.target.value})} required />
            </div>
            <div className={styles.formGroup}>
              <label>Humidity (%):</label>
              <input type="number" step="0.1" value={formData.humidity} onChange={(e) => setFormData({...formData, humidity: e.target.value})} required />
            </div>
            <button type="submit" disabled={loading} className={styles.button}>
              {loading ? '⏳ Processing...' : '🚀 Add & Predict'}
            </button>
          </form>

          {predictionResult && (
            <div className={`${styles.predictionResult} ${styles[predictionResult.riskLevel.toLowerCase()]}`}>
              <h3>🎯 Prediction Result</h3>
              <div className={styles.riskScore}>{predictionResult.riskScore}% Risk</div>
              <p><strong>Prediction:</strong> {predictionResult.prediction}</p>
              <p><strong>Risk Level:</strong> {predictionResult.riskLevel}</p>
              <p><strong>Action:</strong> {predictionResult.recommendedAction}</p>
            </div>
          )}
        </div>

        <div className={styles.card}>
          <h2>🚨 Recent Alerts</h2>
          {alerts.length > 0 ? (
            <table className={styles.table}>
              <thead>
                <tr><th>Severity</th><th>Message</th><th>Risk</th></tr>
              </thead>
              <tbody>
                {alerts.map((alert, i) => (
                  <tr key={i}>
                    <td><span className={`${styles.badge} ${styles[alert.Severity.toLowerCase()]}`}>{alert.Severity}</span></td>
                    <td>{alert.Message}</td>
                    <td>{alert.RiskScore}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p>✓ No alerts</p>}
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <h2>📈 Recent Predictions</h2>
          <div className={styles.filterGroup}>
            <button onClick={loadPredictions} className={styles.refreshBtnSmall} title="Refresh Now">🔄</button>
            <label>Show: </label>
            <select 
              value={predictionsLimit} 
              onChange={(e) => setPredictionsLimit(e.target.value === 'all' ? 'all' : parseInt(e.target.value))}
              className={styles.filterSelect}
            >
              <option value={10}>Last 10</option>
              <option value={25}>Last 25</option>
              <option value={50}>Last 50</option>
              <option value={100}>Last 100</option>
              <option value="all">All Predictions</option>
            </select>
          </div>
        </div>
        {predictions.length > 0 ? (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Equipment</th>
                <th>Temp (°C)</th>
                <th>Vibration</th>
                <th>Pressure</th>
                <th>Humidity (%)</th>
                <th>Prediction</th>
                <th>Risk Score</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((pred, i) => {
                const predictedClass = pred.PredictedClass != null ? (pred.PredictedClass == 0 ? 'Normal' : 'Failure') : (pred.Prediction || 'N/A');
                return (
                  <tr key={i}>
                    <td>{pred.EquipmentID}</td>
                    <td>{pred.Temperature ? Number(pred.Temperature).toFixed(1) : 'N/A'}</td>
                    <td>{pred.Vibration ? Number(pred.Vibration).toFixed(2) : 'N/A'}</td>
                    <td>{pred.Pressure ? Number(pred.Pressure).toFixed(1) : 'N/A'}</td>
                    <td>{pred.Humidity ? Number(pred.Humidity).toFixed(1) : 'N/A'}</td>
                    <td><span className={`${styles.badge} ${styles[pred.RiskLevel?.toLowerCase() || 'low']}`}>{predictedClass}</span></td>
                    <td><strong>{pred.FailureRisk != null ? Number(pred.FailureRisk).toFixed(2) : (pred.RiskScore || '0')}%</strong></td>
                    <td>{pred.RecommendedAction || 'N/A'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : <p>No predictions yet</p>}
      </div>
    </div>
  );
}
