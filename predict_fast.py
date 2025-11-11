import mysql.connector
from mysql.connector import Error
from datetime import datetime
from typing import Optional, List
import hashlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class Database:
    """Database connection handler (tables managed by update_database.py)"""
    
    def __init__(self, host='localhost', user='root', password='001100Yy', database='iot_monitoring'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print(f"Connected to MySQL database: {self.database}")
                return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.original_shape = None
        self.cleaned_shape = None
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "="*60)
        print("STEP 1: CHECKING FOR MISSING VALUES (NaN)")
        print("="*60)
        missing_info = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        print("\nMissing Values Summary:")
        print(missing_info.to_string(index=False))
        total_missing = df.isnull().sum().sum()
        print(f"\n{'⚠️  Total missing values found: ' + str(total_missing) if total_missing > 0 else '✓ No missing values found!'}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy='median') -> pd.DataFrame:
        print("\n" + "="*60 + "\nSTEP 2: HANDLING MISSING VALUES\n" + "="*60)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if strategy == 'median':
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    print(f"✓ Filled {col} with median: {fill_value:.2f}")
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                    print(f"✓ Filled {col} with mean: {fill_value:.2f}")
                elif strategy == 'drop':
                    df.dropna(subset=[col], inplace=True)
                    print(f"✓ Dropped rows with missing {col}")
        print(f"\nDataset shape after handling missing values: {df.shape}")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "="*60 + "\nSTEP 3: REMOVING DUPLICATE ROWS\n" + "="*60)
        before, df = len(df), df.drop_duplicates()
        removed = before - len(df)
        print(f"{'⚠️  Removed ' + str(removed) + ' duplicate rows' if removed > 0 else '✓ No duplicate rows found'}")
        print(f"Dataset shape after removing duplicates: {df.shape}")
        return df
    
    def detect_and_remove_outliers(self, df: pd.DataFrame, columns: list, method='iqr', threshold=1.5) -> pd.DataFrame:
        print("\n" + "="*60 + "\nSTEP 4: DETECTING AND REMOVING OUTLIERS (NOISY DATA)\n" + "="*60)
        before = len(df)
        for col in columns:
            if col not in df.columns:
                continue
            if method == 'iqr':
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - threshold * IQR, Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                print(f"\n{col}: Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}, Bounds: [{lower_bound:.2f}, {upper_bound:.2f}], Outliers: {len(outliers)}")
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
                print(f"\n{col}: Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}, Outliers (|z| > {threshold}): {len(outliers)}")
                df = df[z_scores <= threshold]
        print(f"\n⚠️  Total rows removed as outliers: {before - len(df)}\nDataset shape after outlier removal: {df.shape}")
        return df
    
    def feature_engineering(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        print("\n" + "="*60 + "\nSTEP 5: FEATURE ENGINEERING\n" + "="*60)
        if all(col in df.columns for col in feature_cols):
            interactions = [
                ('temperature', 'vibration', 'temp_vib_interaction'),
                ('temperature', 'pressure', 'temp_pressure_interaction'),
                ('vibration', 'pressure', 'vib_pressure_interaction')
            ]
            for col1, col2, name in interactions:
                if col1 in feature_cols and col2 in feature_cols:
                    df[name] = df[col1] * df[col2]
                    print(f"✓ Created: {name} ({col1.capitalize()} × {col2.capitalize()})")
            
            if 'humidity' in feature_cols:
                for col in ['temperature', 'vibration', 'pressure']:
                    if col in feature_cols:
                        df[f'{col}_humidity_interaction'] = df[col] * df['humidity']
                        print(f"✓ Created: {col}_humidity_interaction ({col.capitalize()} × Humidity)")
            
            if len(feature_cols) >= 2:
                df['feature_mean'] = df[feature_cols].mean(axis=1)
                df['feature_std'] = df[feature_cols].std(axis=1)
                df['feature_range'] = df[feature_cols].max(axis=1) - df[feature_cols].min(axis=1)
                print("✓ Created: feature_mean, feature_std, feature_range")
            
            for col in feature_cols:
                df[f'{col}_squared'] = df[col] ** 2
            print(f"✓ Created: {', '.join([f'{col}_squared' for col in feature_cols])}")
            
            if all(col in feature_cols for col in ['temperature', 'vibration', 'pressure']):
                df['critical_condition'] = ((df['temperature'] > df['temperature'].quantile(0.75)) & 
                                            (df['vibration'] > df['vibration'].quantile(0.75))).astype(int)
                print("✓ Created: critical_condition (High temp + High vibration flag)")
        print(f"\nDataset shape after feature engineering: {df.shape}, New features created: {df.shape[1] - len(feature_cols) - 1}")
        return df
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        print("\n" + "="*60 + "\nSTEP 6: DATA BALANCING USING SMOTE\n" + "="*60)
        print("\nClass distribution BEFORE SMOTE:")
        print(y.value_counts().to_frame('Count'))
        print(f"\nClass imbalance ratio: {(y.value_counts().values[0] / y.value_counts().values[1]):.2f}:1")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("\nClass distribution AFTER SMOTE:")
        print(pd.Series(y_balanced).value_counts().to_frame('Count'))
        print(f"\n✓ Dataset balanced successfully! Original size: {len(y)} → Balanced size: {len(y_balanced)}")
        return X_balanced, y_balanced


class MLModel:
    def __init__(self, model_path='rf_model.pkl', scaler_path='scaler.pkl'):
        self.model = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_columns = ['temperature', 'vibration', 'pressure', 'humidity']
        self.target_column = 'failure'
        self.preprocessor = DataPreprocessor()
        self.all_feature_columns = []
    
    def load_data(self, csv_file: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_file)
            print(f"\n✓ Data loaded successfully! Shape: {df.shape}, Columns: {df.columns.tolist()}")
            print(f"\nFirst few rows:\n{df.head()}\n\nData types:\n{df.dtypes}\n\nBasic statistics:\n{df.describe()}")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame):
        print("\n" + "="*60 + "\nDATA PREPARATION AND CLEANING PIPELINE\n" + "="*60)
        self.preprocessor.original_shape = df.shape
        print(f"\nOriginal dataset shape: {df.shape}")
        df = df.rename(columns={col: col.strip().lower() for col in df.columns})
        df = self.preprocessor.check_missing_values(df)
        df = self.preprocessor.handle_missing_values(df, strategy='median')
        df = self.preprocessor.remove_duplicates(df)
        
        possible_targets = ['failure', 'faulty', 'label', 'target']
        for candidate in possible_targets:
            if candidate in df.columns:
                self.target_column = candidate
                break
        if self.target_column not in df.columns:
            raise KeyError(f"Target column not found. Expected one of {possible_targets}")
        
        available_features = [col for col in self.feature_columns if col in df.columns]
        df = self.preprocessor.detect_and_remove_outliers(df, columns=available_features, method='iqr', threshold=5.0)
        df = self.preprocessor.feature_engineering(df, available_features)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.all_feature_columns = [col for col in numeric_columns if col != self.target_column]
        X, y = df[self.all_feature_columns], df[self.target_column]
        self.preprocessor.cleaned_shape = X.shape
        
        print(f"\n{'='*60}\nDATA PREPARATION SUMMARY\n{'='*60}")
        print(f"Original: {self.preprocessor.original_shape}, Final: {self.preprocessor.cleaned_shape}")
        print(f"Rows removed: {self.preprocessor.original_shape[0] - self.preprocessor.cleaned_shape[0]}, Features created: {self.preprocessor.cleaned_shape[1] - len(available_features)}")
        return X, y
    
    def train_model(self, csv_file: str):
        print("\n" + "="*60 + "\nTRAINING XGBOOST MODEL WITH PREPROCESSING\n" + "="*60)
        df = self.load_data(csv_file)
        if df is None:
            return False
        
        X, y = self.prepare_data(df)
        X_balanced, y_balanced = self.preprocessor.apply_smote(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.10, random_state=42, stratify=y_balanced)
        
        scaler = StandardScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
        
        print(f"\n{'='*60}\nMODEL TRAINING\n{'='*60}")
        print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
        print("\nTraining XGBoost Classifier (Optimized for 98%+ accuracy)...")
        
        from xgboost import XGBClassifier
        self.model = XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=1, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
                                   use_label_encoder=False, eval_metric='logloss', early_stopping_rounds=50, scale_pos_weight=1)
        self.model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        self.scaler = scaler
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        print(f"\n{'='*60}\nXGBOOST MODEL EVALUATION RESULTS\n{'='*60}")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        print(f"\n🎯 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%) | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        if hasattr(self.model, 'best_iteration'):
            print(f"📊 Best Iteration: {self.model.best_iteration} | Total Trees: {self.model.best_iteration + 1}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📈 Confusion Matrix:\n    ┌─────────────┬─────────────┐\n    │ TN: {cm[0,0]:7d} │ FP: {cm[0,1]:7d} │")
        print(f"    ├─────────────┼─────────────┤\n    │ FN: {cm[1,0]:7d} │ TP: {cm[1,1]:7d} │\n    └─────────────┴─────────────┘")
        print(f"\n📋 Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
        print(f"\n🔍 XGBoost Training Details:\n   Features: {len(self.all_feature_columns)} | Training: {len(X_train)} | Test: {len(X_test)}")
        print(f"   Class Balance: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        feature_importance = pd.DataFrame({'feature': self.all_feature_columns, 'importance': self.model.feature_importances_}).sort_values('importance', ascending=False)
        print("\n⭐ Top 15 Most Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")
        
        joblib.dump(self.model, self.model_path)
        print(f"\n✓ Model saved to {self.model_path}")
        
        # Save the scaler
        scaler_path = 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        print(f"\n{'='*60}\nTEST SET PREDICTIONS SAMPLE (First 10)\n{'='*60}")
        sample_df = pd.DataFrame(X_test.iloc[:10])
        sample_df['Actual'], sample_df['Predicted'] = y_test[:10].values, y_pred[:10]
        sample_df['Failure_Probability'] = y_pred_proba[:10, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:10, 0]
        original_features = [col for col in self.feature_columns if col in X_test.columns]
        print(sample_df[original_features + ['Actual', 'Predicted', 'Failure_Probability']].to_string(index=False))
        return True
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        print(f"Model file not found: {self.model_path}")
        return False
    
    def predict(self, temperature: float, vibration: float, pressure: float, humidity: float = 50.0):
        if self.model is None:
            print("Model not loaded or trained")
            return None, None
        
        features = [temperature, vibration, pressure, humidity]
        input_dict = dict(zip(['temperature', 'vibration', 'pressure', 'humidity'], features))
        input_dict.update({
            'temp_vib_interaction': temperature * vibration,
            'temp_pressure_interaction': temperature * pressure,
            'vib_pressure_interaction': vibration * pressure,
            'temp_humidity_interaction': temperature * humidity,
            'vib_humidity_interaction': vibration * humidity,
            'pressure_humidity_interaction': pressure * humidity,
            'feature_mean': np.mean(features),
            'feature_std': np.std(features),
            'feature_range': max(features) - min(features),
            'temperature_squared': temperature ** 2,
            'vibration_squared': vibration ** 2,
            'pressure_squared': pressure ** 2,
            'humidity_squared': humidity ** 2,
            'critical_condition': 0
        })
        
        input_data = pd.DataFrame([input_dict])
        for col in self.all_feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[self.all_feature_columns]
        
        input_data_scaled = self.scaler.transform(input_data) if hasattr(self, 'scaler') and self.scaler else input_data
        prediction = self.model.predict(input_data_scaled)[0]
        probability = self.model.predict_proba(input_data_scaled)[0]
        failure_prob = probability[1] if len(probability) > 1 else probability[0]
        return prediction, failure_prob


class User:
    """User class for authentication and authorization"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user credentials"""
        cursor = self.db.connection.cursor(dictionary=True)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        query = "SELECT * FROM Users WHERE Email = %s AND PasswordHash = %s"
        cursor.execute(query, (email, password_hash))
        user = cursor.fetchone()
        cursor.close()
        
        if user:
            print(f"User {email} authenticated successfully")
            return user
        else:
            print("Authentication failed")
            return None
    
    def create_user(self, role: str, permissions: str, email: str, password: str) -> int:
        """Create a new user"""
        cursor = self.db.connection.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        query = """
            INSERT INTO Users (Role, Permissions, Email, PasswordHash)
            VALUES (%s, %s, %s, %s)
        """
        
        try:
            cursor.execute(query, (role, permissions, email, password_hash))
            self.db.connection.commit()
            user_id = cursor.lastrowid
            print(f"User created with ID: {user_id}")
            return user_id
        except Error as e:
            if getattr(e, 'errno', None) == 1062:
                # Duplicate entry: fetch existing user ID
                print(f"User {email} already exists. Using existing account.")
                cursor_select = self.db.connection.cursor(dictionary=True)
                cursor_select.execute("SELECT UserID FROM Users WHERE Email = %s", (email,))
                existing = cursor_select.fetchone()
                cursor_select.close()
                if existing:
                    return existing['UserID']
                return -1
            print(f"Error creating user: {e}")
            return -1
        finally:
            cursor.close()


class Equipment:
    """Equipment management class"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def add_equipment(self, equipment_type: str, installation_date: datetime) -> int:
        """Add new equipment"""
        cursor = self.db.connection.cursor()
        
        query = """
            INSERT INTO Equipment (Type, InstallationDate, LastMaintenanceDate)
            VALUES (%s, %s, %s)
        """
        
        try:
            cursor.execute(query, (equipment_type, installation_date, installation_date))
            self.db.connection.commit()
            equipment_id = cursor.lastrowid
            print(f"Equipment added with ID: {equipment_id}")
            return equipment_id
        except Error as e:
            print(f"Error adding equipment: {e}")
            return -1
        finally:
            cursor.close()
    
    def track_maintenance(self, equipment_id: int, maintenance_date: datetime) -> bool:
        """Track equipment maintenance"""
        cursor = self.db.connection.cursor()
        
        query = "UPDATE Equipment SET LastMaintenanceDate = %s WHERE EquipmentID = %s"
        
        try:
            cursor.execute(query, (maintenance_date, equipment_id))
            self.db.connection.commit()
            print(f"Maintenance tracked for Equipment ID: {equipment_id}")
            return True
        except Error as e:
            print(f"Error tracking maintenance: {e}")
            return False
        finally:
            cursor.close()
    
    def get_equipment(self, equipment_id: int) -> Optional[dict]:
        """Get equipment details"""
        cursor = self.db.connection.cursor(dictionary=True)
        query = "SELECT * FROM Equipment WHERE EquipmentID = %s"
        cursor.execute(query, (equipment_id,))
        equipment = cursor.fetchone()
        cursor.close()
        return equipment


class SensorData:
    """Sensor data collection and validation"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def validate_sensor_reading(self, temperature: float, vibration: float, pressure: float) -> bool:
        """Validate sensor readings are within acceptable ranges"""
        if not (0 <= temperature <= 150):
            print("Warning: Temperature out of range")
            return False
        if not (0 <= vibration <= 100):
            print("Warning: Vibration out of range")
            return False
        if not (0 <= pressure <= 200):
            print("Warning: Pressure out of range")
            return False
        return True
    
    def add_sensor_reading(self, equipment_id: int, timestamp: datetime, 
                          temperature: float, vibration: float, pressure: float) -> int:
        """Add new sensor reading"""
        if not self.validate_sensor_reading(temperature, vibration, pressure):
            status = 'Warning'
        else:
            status = 'Normal'
        
        cursor = self.db.connection.cursor()
        
        query = """
            INSERT INTO SensorData (EquipmentID, Timestamp, Temperature, Vibration, Pressure, Status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        try:
            cursor.execute(query, (equipment_id, timestamp, temperature, vibration, pressure, status))
            self.db.connection.commit()
            sensor_id = cursor.lastrowid
            print(f"Sensor reading added with ID: {sensor_id}")
            return sensor_id
        except Error as e:
            print(f"Error adding sensor reading: {e}")
            return -1
        finally:
            cursor.close()
    
    def get_sensor_data(self, equipment_id: int, limit: int = 10) -> List[dict]:
        """Get recent sensor data for equipment"""
        cursor = self.db.connection.cursor(dictionary=True)
        query = """
            SELECT * FROM SensorData 
            WHERE EquipmentID = %s 
            ORDER BY Timestamp DESC 
            LIMIT %s
        """
        cursor.execute(query, (equipment_id, limit))
        data = cursor.fetchall()
        cursor.close()
        return data


class Prediction:
    """Predictive maintenance and failure analysis"""
    
    def __init__(self, db: Database, ml_model: MLModel = None):
        self.db = db
        self.ml_model = ml_model
    
    def generate_prediction(self, sensor_id: int) -> bool:
        """Generate failure prediction based on sensor data using ML model"""
        cursor = self.db.connection.cursor(dictionary=True)
        
        query = "SELECT * FROM SensorData WHERE SensorID = %s"
        cursor.execute(query, (sensor_id,))
        sensor_data = cursor.fetchone()
        
        if not sensor_data:
            cursor.close()
            return False
        
        temp = sensor_data['Temperature']
        vib = sensor_data['Vibration']
        pressure = sensor_data['Pressure']
        
        # Use ML model if available
        if self.ml_model and self.ml_model.model is not None:
            prediction, failure_risk = self.ml_model.predict(temp, vib, pressure)
            predicted_class = "Failure" if prediction == 1 else "Normal"
        else:
            # Fallback to rule-based prediction
            failure_risk = 0.0
            if temp > 100:
                failure_risk += 0.3
            if vib > 70:
                failure_risk += 0.4
            if pressure > 150:
                failure_risk += 0.3
            failure_risk = min(failure_risk, 1.0)
            predicted_class = "Failure" if failure_risk > 0.5 else "Normal"
        
        # Determine recommended action
        if failure_risk > 0.7:
            recommended_action = "Immediate maintenance required"
        elif failure_risk > 0.4:
            recommended_action = "Schedule maintenance soon"
        else:
            recommended_action = "Normal operation"
        
        # Insert prediction
        insert_query = """
            INSERT INTO Predictions (SensorID, Timestamp, FailureRisk, RecommendedAction, PredictedClass)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        try:
            # Convert numpy types to Python native types
            failure_risk_py = float(failure_risk)
            predicted_class_py = str(predicted_class)
            
            cursor.execute(insert_query, (sensor_id, datetime.now(), failure_risk_py, recommended_action, predicted_class_py))
            self.db.connection.commit()
            print(f"Prediction: Class={predicted_class}, Risk={failure_risk:.4f}, Action={recommended_action}")
            return True
        except Error as e:
            print(f"Error generating prediction: {e}")
            return False
        finally:
            cursor.close()


class Alert:
    """Alert management and notification system"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def send_alert(self, user_id: int, equipment_id: int, message: str, severity: str) -> bool:
        """Send alert to user"""
        cursor = self.db.connection.cursor()
        
        query = """
            INSERT INTO Alerts (UserID, EquipmentID, Message, Severity, Timestamp, Status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        try:
            cursor.execute(query, (user_id, equipment_id, message, severity, datetime.now(), 'Unread'))
            self.db.connection.commit()
            print(f"Alert sent to User ID {user_id}: {message}")
            return True
        except Error as e:
            print(f"Error sending alert: {e}")
            return False
        finally:
            cursor.close()
    
    def get_user_alerts(self, user_id: int, unread_only: bool = False) -> List[dict]:
        """Get alerts for a user"""
        cursor = self.db.connection.cursor(dictionary=True)
        
        if unread_only:
            query = """
                SELECT * FROM Alerts 
                WHERE UserID = %s AND Status = 'Unread' 
                ORDER BY Timestamp DESC
            """
        else:
            query = "SELECT * FROM Alerts WHERE UserID = %s ORDER BY Timestamp DESC"
        
        cursor.execute(query, (user_id,))
        alerts = cursor.fetchall()
        cursor.close()
        return alerts
    
    def mark_alert_read(self, alert_id: int) -> bool:
        """Mark alert as read"""
        cursor = self.db.connection.cursor()
        query = "UPDATE Alerts SET Status = 'Read' WHERE AlertID = %s"
        
        try:
            cursor.execute(query, (alert_id,))
            self.db.connection.commit()
            return True
        except Error as e:
            print(f"Error marking alert as read: {e}")
            return False
        finally:
            cursor.close()


# Example usage and testing
if __name__ == "__main__":
    # CSV file path - CHANGE THIS TO YOUR CSV FILE PATH
    CSV_FILE = "C:\\Users\\Admin.DESKTOP-QKNRRPJ\\OneDrive - King Suad University\\MLProject\\equipment_anomaly_data (2).csv"
    
    # Initialize ML Model
    print("\n" + "="*60)
    print("INITIALIZING MACHINE LEARNING MODEL")
    print("="*60)
    
    ml_model = MLModel()
    
    # Train model if CSV file exists
    if os.path.exists(CSV_FILE):
        ml_model.train_model(CSV_FILE)
    else:
        print(f"\n⚠️  CSV file not found: {CSV_FILE}")
        print("Please provide a CSV file with columns: Temperature, Vibration, Pressure, Failure")
        print("Creating sample CSV file for demonstration...")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'Temperature': np.random.uniform(20, 140, n_samples),
            'Vibration': np.random.uniform(10, 95, n_samples),
            'Pressure': np.random.uniform(50, 190, n_samples)
        })
        
        # Create failure labels based on thresholds
        sample_data['Failure'] = (
            ((sample_data['Temperature'] > 100) & (sample_data['Vibration'] > 70)) |
            ((sample_data['Temperature'] > 110) & (sample_data['Pressure'] > 160)) |
            ((sample_data['Vibration'] > 80) & (sample_data['Pressure'] > 170))
        ).astype(int)
        
        sample_data.to_csv(CSV_FILE, index=False)
        print(f"Sample CSV created: {CSV_FILE}")
        
        # Train with sample data
        ml_model.train_model(CSV_FILE)
    
    # Initialize database
    print("\n" + "="*60)
    print("CONNECTING TO DATABASE")
    print("="*60)
    
    db = Database(host='localhost', user='root', password='001100Yy', database='iot_monitoring')
    
    if db.connect():
        # Create instances
        user_manager = User(db)
        equipment_manager = Equipment(db)
        sensor_manager = SensorData(db)
        prediction_engine = Prediction(db, ml_model)
        alert_system = Alert(db)
        
        # Example workflow
        print("\n" + "="*60)
        print("TESTING SYSTEM WITH ML PREDICTIONS")
        print("="*60)
        
        print("\n=== Creating User ===")
        user_id = user_manager.create_user(
            role='Administrator',
            permissions='read,write,delete',
            email='admin@example.com',
            password='secure_password'
        )
        
        print("\n=== Adding Equipment ===")
        equipment_id = equipment_manager.add_equipment(
            equipment_type='Industrial Motor',
            installation_date=datetime.now()
        )
        
        # Close connection
        print("\n" + "="*60)
        db.close()
    else:
        print("Failed to connect to database")
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED")
    print("="*60)