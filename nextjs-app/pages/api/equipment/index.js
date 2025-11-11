import { query } from '../../../lib/mysql';

export default async function handler(req, res) {
  if (req.method === 'GET') {
    try {
      const equipment = await query(`SELECT * FROM Equipment ORDER BY CreatedAt DESC`);
      res.status(200).json({ success: true, data: equipment, count: equipment.length });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  } 
  else if (req.method === 'POST') {
    try {
      const { name, type, location, installationDate } = req.body;
      const result = await query(
        `INSERT INTO Equipment (Name, Type, Location, InstallationDate, Status) VALUES (?, ?, ?, ?, 'Active')`,
        [name, type, location || null, installationDate || new Date()]
      );
      res.status(200).json({ success: true, id: result.insertId });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  } 
  else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
