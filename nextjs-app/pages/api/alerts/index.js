import { query } from '../../../lib/mysql';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const status = req.query.status || null;
    const limit = parseInt(req.query.limit) || 100;

    let sql, alerts;

    if (status) {
      sql = `SELECT * FROM alerts WHERE Status = ? ORDER BY Timestamp DESC LIMIT ${limit}`;
      alerts = await query(sql, [status]);
    } else {
      sql = `SELECT * FROM alerts ORDER BY Timestamp DESC LIMIT ${limit}`;
      alerts = await query(sql);
    }

    res.status(200).json({
      success: true,
      data: alerts,
      count: alerts.length
    });

  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ error: error.message });
  }
}
