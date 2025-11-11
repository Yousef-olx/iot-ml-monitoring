import { query } from '../../../lib/mysql';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const limit = parseInt(req.query.limit) || 50;

    const predictions = await query(
      `SELECT p.*, s.EquipmentID, s.Temperature, s.Vibration, s.Pressure, s.Humidity 
       FROM predictions p 
       JOIN sensordata s ON p.SensorID = s.SensorID 
       ORDER BY p.Timestamp DESC 
       LIMIT ${limit}`
    );

    res.status(200).json({
      success: true,
      data: predictions,
      count: predictions.length
    });

  } catch (error) {
    console.error('Error fetching predictions:', error);
    res.status(500).json({ error: error.message });
  }
}
