import { query } from '../../../lib/mysql';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const limit = parseInt(req.query.limit) || 50;
    const equipmentId = req.query.equipmentId;

    let sql, sensors;

    if (equipmentId) {
      sql = `SELECT * FROM sensordata WHERE EquipmentID = ? ORDER BY Timestamp DESC LIMIT ${limit}`;
      sensors = await query(sql, [equipmentId]);
    } else {
      sql = `SELECT * FROM sensordata ORDER BY Timestamp DESC LIMIT ${limit}`;
      sensors = await query(sql);
    }

    res.status(200).json({
      success: true,
      data: sensors,
      count: sensors.length
    });

  } catch (error) {
    console.error('Error fetching sensors:', error);
    res.status(500).json({ error: error.message });
  }
}
