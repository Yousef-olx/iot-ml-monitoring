import { query } from '../../../lib/mysql';
import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { equipmentId, temperature, vibration, pressure, humidity } = req.body;

    if (!equipmentId || temperature === undefined || vibration === undefined || pressure === undefined) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const temp = parseFloat(temperature);
    const vib = parseFloat(vibration);
    const press = parseFloat(pressure);
    const hum = parseFloat(humidity || 50.0);
    const timestamp = new Date();

    // 0. Auto-create equipment if it doesn't exist
    await query(
      `INSERT INTO equipment (EquipmentID, Name, Type, Location, InstallationDate, Status) 
       VALUES (?, ?, ?, ?, ?, ?) 
       ON DUPLICATE KEY UPDATE EquipmentID = EquipmentID`,
      [equipmentId, `Equipment ${equipmentId}`, 'Auto', 'Unknown', timestamp.toISOString().split('T')[0], 'Active']
    );

    // 1. Save to MySQL
    const result = await query(
      `INSERT INTO sensordata (EquipmentID, Timestamp, Temperature, Vibration, Pressure, Humidity, Status) 
       VALUES (?, ?, ?, ?, ?, ?, 'Pending')`,
      [equipmentId, timestamp, temp, vib, press, hum]
    );
    const sensorId = result.insertId;

    // 2. Get ML Prediction
    let predictionData = null;
    try {
      const mlResponse = await axios.post(`${process.env.ML_API_URL}/predict`, {
        temperature: temp,
        vibration: vib,
        pressure: press,
        humidity: hum
      }, { timeout: 5000 });

      const prediction = mlResponse.data;
      predictionData = {
        prediction: prediction.prediction,
        riskScore: prediction.risk_score,
        riskLevel: prediction.risk_level,
        recommendedAction: prediction.recommended_action
      };

      // 3. Save prediction to MySQL
      await query(
        `INSERT INTO predictions (SensorID, Timestamp, PredictedClass, FailureRisk, RiskLevel, RecommendedAction) 
         VALUES (?, ?, ?, ?, ?, ?)`,
        [sensorId, timestamp, predictionData.prediction, predictionData.riskScore, predictionData.riskLevel, predictionData.recommendedAction]
      );

      // 4. Update sensor status
      await query(`UPDATE sensordata SET Status = 'Processed' WHERE SensorID = ?`, [sensorId]);

      // 5. Create alert if high risk
      if (predictionData.riskScore > 40) {
        const alertMessage = `${predictionData.riskLevel} Risk: ${predictionData.recommendedAction}`;
        
        await query(
          `INSERT INTO alerts (UserID, EquipmentID, Message, Severity, RiskScore, Timestamp, Status) 
           VALUES (?, ?, ?, ?, ?, ?, 'Unread')`,
          [1, equipmentId, alertMessage, predictionData.riskLevel, predictionData.riskScore, timestamp]
        );
      }

    } catch (mlError) {
      console.error('ML API Error:', mlError.message);
      console.error('ML API Error Details:', mlError.response?.data || mlError.code || 'Unknown error');
      console.error('ML API URL:', process.env.ML_API_URL);
      await query(`UPDATE sensordata SET Status = 'Error' WHERE SensorID = ?`, [sensorId]);
    }

    res.status(200).json({
      success: true,
      sensorId,
      sensorData: { equipmentId, temperature: temp, vibration: vib, pressure: press, humidity: hum, timestamp },
      prediction: predictionData
    });

  } catch (error) {
    console.error('Error adding sensor:', error);
    res.status(500).json({ error: error.message });
  }
}
