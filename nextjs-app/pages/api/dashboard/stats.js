import { query } from '../../../lib/mysql';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const [sensors] = await query(`SELECT COUNT(*) as count FROM sensordata`);
    const [predictions] = await query(`SELECT COUNT(*) as count FROM predictions`);
    const [alerts] = await query(`SELECT COUNT(*) as count FROM alerts WHERE Status = 'Unread'`);
    const [highRisk] = await query(`SELECT COUNT(*) as count FROM predictions WHERE FailureRisk > 0.7`);
    const [equipment] = await query(`SELECT COUNT(*) as count FROM equipment`);

    res.status(200).json({
      success: true,
      stats: {
        totalSensors: sensors.count,
        totalPredictions: predictions.count,
        unreadAlerts: alerts.count,
        highRiskEquipment: highRisk.count,
        totalEquipment: equipment.count
      }
    });

  } catch (error) {
    console.error('Error fetching stats:', error);
    res.status(500).json({ error: error.message });
  }
}
