"""
Complete Database Setup Script for IoT ML Monitoring System
Creates database, tables, and ensures all columns exist
"""

import mysql.connector
from mysql.connector import Error

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '001100Yy'
}

DATABASE_NAME = 'iot_monitoring'

def create_database():
    """Create the database if it doesn't exist"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
        print(f"✅ Database '{DATABASE_NAME}' created/verified")
        
        cursor.close()
        conn.close()
        return True
    except Error as e:
        print(f"❌ Error creating database: {e}")
        return False

def create_tables():
    """Create all tables with complete schema"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG, database=DATABASE_NAME)
        cursor = conn.cursor()
        
        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                UserID INT PRIMARY KEY AUTO_INCREMENT,
                Username VARCHAR(100) UNIQUE NOT NULL,
                Email VARCHAR(100) UNIQUE NOT NULL,
                PasswordHash VARCHAR(255),
                CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Table 'Users' created/verified")
        
        # 2. Equipment table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Equipment (
                EquipmentID INT PRIMARY KEY AUTO_INCREMENT,
                Name VARCHAR(100) NOT NULL,
                Type VARCHAR(50),
                Location VARCHAR(100),
                InstallationDate DATE,
                Status VARCHAR(20) DEFAULT 'Active'
            )
        """)
        print("✅ Table 'Equipment' created/verified")
        
        # 3. SensorData table (with Humidity)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SensorData (
                SensorID INT PRIMARY KEY AUTO_INCREMENT,
                EquipmentID INT,
                Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                Temperature DECIMAL(10,2),
                Vibration DECIMAL(10,2),
                Pressure DECIMAL(10,2),
                Humidity DECIMAL(10,2) DEFAULT 50.0,
                Status VARCHAR(20) DEFAULT 'Normal',
                FOREIGN KEY (EquipmentID) REFERENCES Equipment(EquipmentID) ON DELETE CASCADE
            )
        """)
        print("✅ Table 'SensorData' created/verified")
        
        # 4. Predictions table (with RiskScore, RiskLevel, RecommendedAction)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Predictions (
                PredictionID INT PRIMARY KEY AUTO_INCREMENT,
                SensorID INT,
                Prediction VARCHAR(50),
                RiskScore DECIMAL(5,2) DEFAULT 0,
                RiskLevel VARCHAR(20) DEFAULT 'Low',
                RecommendedAction TEXT,
                Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (SensorID) REFERENCES SensorData(SensorID) ON DELETE CASCADE
            )
        """)
        print("✅ Table 'Predictions' created/verified")
        
        # 5. Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Alerts (
                AlertID INT PRIMARY KEY AUTO_INCREMENT,
                EquipmentID INT,
                SensorID INT,
                Message TEXT,
                Severity VARCHAR(20),
                RiskScore DECIMAL(5,2),
                Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                Status VARCHAR(20) DEFAULT 'Unread',
                FOREIGN KEY (EquipmentID) REFERENCES Equipment(EquipmentID) ON DELETE CASCADE,
                FOREIGN KEY (SensorID) REFERENCES SensorData(SensorID) ON DELETE CASCADE
            )
        """)
        print("✅ Table 'Alerts' created/verified")
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        print(f"❌ Error creating tables: {e}")
        return False

def add_missing_columns():
    """Add any missing columns to existing tables"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG, database=DATABASE_NAME)
        cursor = conn.cursor()
        
        print("\n🔧 Checking for missing columns...")
        
        # Add Humidity to sensordata if missing
        try:
            cursor.execute("ALTER TABLE sensordata ADD COLUMN Humidity DECIMAL(10,2) DEFAULT 50.0 AFTER Pressure")
            print("✅ Added Humidity column to sensordata")
        except Error as e:
            if "Duplicate column" in str(e):
                print("✓ Humidity column already exists in sensordata")
            else:
                print(f"⚠️  Could not add Humidity: {e}")
        
        # Check if predictions table has old schema and rename/add columns
        cursor.execute("DESCRIBE predictions")
        columns = {col[0]: col[1] for col in cursor.fetchall()}
        
        # Add RiskScore (if FailureRisk exists, we'll use it as is)
        if 'RiskScore' not in columns and 'FailureRisk' not in columns:
            try:
                cursor.execute("ALTER TABLE predictions ADD COLUMN RiskScore DECIMAL(5,2) DEFAULT 0 AFTER SensorID")
                print("✅ Added RiskScore column to predictions")
            except Error as e:
                print(f"⚠️  Could not add RiskScore: {e}")
        elif 'FailureRisk' in columns:
            print("✓ Using FailureRisk as RiskScore in predictions")
        else:
            print("✓ RiskScore column already exists in predictions")
        
        # Add RiskLevel if missing
        if 'RiskLevel' not in columns:
            try:
                if 'FailureRisk' in columns:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN RiskLevel VARCHAR(20) DEFAULT 'Low' AFTER FailureRisk")
                elif 'RiskScore' in columns:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN RiskLevel VARCHAR(20) DEFAULT 'Low' AFTER RiskScore")
                else:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN RiskLevel VARCHAR(20) DEFAULT 'Low' AFTER SensorID")
                print("✅ Added RiskLevel column to predictions")
            except Error as e:
                print(f"⚠️  Could not add RiskLevel: {e}")
        else:
            print("✓ RiskLevel column already exists in predictions")
        
        # RecommendedAction usually exists, but check
        if 'RecommendedAction' not in columns:
            try:
                cursor.execute("ALTER TABLE predictions ADD COLUMN RecommendedAction TEXT")
                print("✅ Added RecommendedAction column to predictions")
            except Error as e:
                print(f"⚠️  Could not add RecommendedAction: {e}")
        else:
            print("✓ RecommendedAction column already exists in predictions")
        
        # Add Prediction column if missing (PredictedClass exists but we need Prediction too)
        if 'Prediction' not in columns and 'PredictedClass' not in columns:
            try:
                cursor.execute("ALTER TABLE predictions ADD COLUMN Prediction VARCHAR(50) AFTER SensorID")
                print("✅ Added Prediction column to predictions")
            except Error as e:
                print(f"⚠️  Could not add Prediction: {e}")
        elif 'PredictedClass' in columns:
            print("✓ Using PredictedClass as Prediction in predictions")
        else:
            print("✓ Prediction column already exists in predictions")
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        print(f"❌ Error adding columns: {e}")
        return False

def verify_schema():
    """Verify all tables and columns exist"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG, database=DATABASE_NAME)
        cursor = conn.cursor()
        
        print("\n📊 Verifying database schema...")
        
        # Check tables (lowercase)
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = ['users', 'equipment', 'sensordata', 'predictions', 'alerts']
        for table in required_tables:
            if table in tables:
                print(f"✓ Table '{table}' exists")
                
                # Show columns for each table
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                print(f"  Columns: {', '.join([col[0] for col in columns])}")
            else:
                print(f"❌ Table '{table}' is missing!")
        
        cursor.close()
        conn.close()
        return True
    except Error as e:
        print(f"❌ Error verifying schema: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 IoT ML Monitoring - Database Setup")
    print("="*60)
    
    # Step 1: Create database
    print("\n📦 Step 1: Creating database...")
    if not create_database():
        print("❌ Failed to create database. Exiting.")
        return
    
    # Step 2: Create tables
    print("\n🔨 Step 2: Creating tables...")
    if not create_tables():
        print("❌ Failed to create tables. Exiting.")
        return
    
    # Step 3: Add missing columns (for upgrading existing databases)
    print("\n🔧 Step 3: Updating schema...")
    add_missing_columns()
    
    # Step 4: Verify everything
    print("\n✅ Step 4: Verification...")
    verify_schema()
    
    print("\n" + "="*60)
    print("✅ DATABASE SETUP COMPLETE!")
    print("="*60)
    print("\nDatabase is ready for use.")
    print("You can now run the ML training script (predict_fast.py)")
    print("and start the Next.js application.")

if __name__ == "__main__":
    main()
