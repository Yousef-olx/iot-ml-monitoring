import { query } from '../../../lib/mysql';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get the most recent sensor reading
    const result = await query(`
      SELECT 
        s.SensorID,
        s.EquipmentID,
        s.Temperature,
        s.Vibration,
        s.Pressure,
        s.Humidity,
        s.Timestamp,
        p.FailureRisk,
        p.PredictedClass,
        p.RiskLevel
      FROM sensordata s
      LEFT JOIN predictions p ON s.SensorID = p.SensorID
      ORDER BY s.Timestamp DESC
      LIMIT 1
    `);

    if (result && result.length > 0) {
      return res.status(200).json({ 
        success: true, 
        data: result[0]
      });
    } else {
      return res.status(200).json({ 
        success: true, 
        data: null 
      });
    }

  } catch (error) {
    console.error('Error fetching latest sensor data:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch latest sensor data',
      details: error.message 
    });
  }
}
