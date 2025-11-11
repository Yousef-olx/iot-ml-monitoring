import { query } from '../../../lib/mysql';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get the maximum equipment ID from the database
    const result = await query(`
      SELECT MAX(CAST(EquipmentID AS UNSIGNED)) as maxId 
      FROM sensordata
    `);

    let nextId = 1;
    if (result && result[0] && result[0].maxId) {
      nextId = parseInt(result[0].maxId) + 1;
    }

    // Ensure the equipment exists in the equipment table
    try {
      await query(
        `INSERT IGNORE INTO equipment (EquipmentID, Name, Type, Location, InstallationDate, Status) 
         VALUES (?, ?, ?, ?, ?, ?)`,
        [nextId, `Equipment ${nextId}`, 'Generic', 'Default Location', new Date().toISOString().split('T')[0], 'Active']
      );
    } catch (eqError) {
      console.log('Equipment insert warning:', eqError.message);
    }

    return res.status(200).json({ 
      success: true, 
      nextId: nextId 
    });

  } catch (error) {
    console.error('Error getting next equipment ID:', error);
    return res.status(500).json({ 
      error: 'Failed to get next equipment ID',
      details: error.message 
    });
  }
}
