import mysql from 'mysql2/promise';

export default async function handler(req, res) {
  try {
    const config = {
      host: process.env.MYSQL_HOST,
      user: process.env.MYSQL_USER,
      password: process.env.MYSQL_PASSWORD,
      database: process.env.MYSQL_DATABASE,
      port: parseInt(process.env.MYSQL_PORT) || 27906,
      ssl: {
        rejectUnauthorized: false
      }
    };

    // Create a test connection
    const connection = await mysql.createConnection(config);
    
    // Test query
    const [rows] = await connection.execute('SELECT 1 as test');
    await connection.end();

    res.status(200).json({
      success: true,
      message: 'Database connection successful!',
      testQuery: rows,
      config: {
        host: config.host,
        user: config.user,
        database: config.database,
        port: config.port,
        password: '***hidden***'
      }
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      code: error.code,
      errno: error.errno
    });
  }
}
