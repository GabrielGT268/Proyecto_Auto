import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MLAnalyzer:
    
    def __init__(self, data):
        self.data = data.copy()
        self.encoders = {}
        self.rf_efficiency_model = None
        self.rf_price_model = None
        self.rf_segment_model = None
        self.feature_importance_efficiency = None
        self.feature_importance_price = None
        self.feature_importance_segment = None
        
    def prepare_features(self):
        ml_data = self.data.copy()
        
        categorical_columns = ['Brand', 'VehicleClass', 'VehicleStyle', 'DriveTrain', 'Transmission']
        
        for col in categorical_columns:
            if col in ml_data.columns:
                le = LabelEncoder()
                ml_data[col] = le.fit_transform(ml_data[col].astype(str))
                self.encoders[col] = le
        
        if 'Engine_Category' in ml_data.columns:
            le = LabelEncoder()
            ml_data['Engine_Category'] = le.fit_transform(ml_data['Engine_Category'].astype(str))
            self.encoders['Engine_Category'] = le
            
        if 'Price_Category' in ml_data.columns:
            le = LabelEncoder()
            ml_data['Price_Category'] = le.fit_transform(ml_data['Price_Category'].astype(str))
            self.encoders['Price_Category'] = le
        
        return ml_data
    
    def predict_efficiency(self):
        ml_data = self.prepare_features()
        
        feature_cols = ['EngineSize', 'Cylinders', 'HorsePower', 'Weight', 'Brand', 'VehicleClass', 'DriveTrain', 'Transmission']
        available_features = [col for col in feature_cols if col in ml_data.columns]
        
        if len(available_features) < 4:
            return None
            
        X = ml_data[available_features]
        y = ml_data['MPG_Average']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.rf_efficiency_model.fit(X_train, y_train)
        
        y_pred = self.rf_efficiency_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.rf_efficiency_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_efficiency = feature_importance
        
        cv_scores = cross_val_score(self.rf_efficiency_model, X, y, cv=5, scoring='r2')
        
        predictions = self.rf_efficiency_model.predict(ml_data[available_features])
        ml_data['Predicted_MPG'] = predictions
        ml_data['MPG_Prediction_Error'] = abs(ml_data['MPG_Average'] - ml_data['Predicted_MPG'])
        
        best_predicted = ml_data.nsmallest(10, 'MPG_Prediction_Error')
        worst_predicted = ml_data.nlargest(10, 'MPG_Prediction_Error')
        
        return {
            'model_performance': {
                'mse': mse,
                'r2_score': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            },
            'feature_importance': feature_importance,
            'best_predicted_vehicles': best_predicted[['Brand', 'Model', 'MPG_Average', 'Predicted_MPG', 'MPG_Prediction_Error']],
            'worst_predicted_vehicles': worst_predicted[['Brand', 'Model', 'MPG_Average', 'Predicted_MPG', 'MPG_Prediction_Error']],
            'predictions': predictions
        }
    
    def predict_price(self):
        ml_data = self.prepare_features()
        
        feature_cols = ['EngineSize', 'Cylinders', 'HorsePower', 'Weight', 'MPG_Average', 'Brand', 'VehicleClass', 'DriveTrain']
        available_features = [col for col in feature_cols if col in ml_data.columns]
        
        if len(available_features) < 4:
            return None
            
        X = ml_data[available_features]
        y = ml_data['MSRP']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_price_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.rf_price_model.fit(X_train, y_train)
        
        y_pred = self.rf_price_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.rf_price_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_price = feature_importance
        
        predictions = self.rf_price_model.predict(ml_data[available_features])
        ml_data['Predicted_Price'] = predictions
        ml_data['Price_Prediction_Error'] = abs(ml_data['MSRP'] - ml_data['Predicted_Price'])
        ml_data['Price_Difference_Pct'] = (ml_data['Price_Prediction_Error'] / ml_data['MSRP']) * 100
        
        undervalued = ml_data[ml_data['MSRP'] < ml_data['Predicted_Price']]
        overvalued = ml_data[ml_data['MSRP'] > ml_data['Predicted_Price']]
        
        top_undervalued = undervalued.nlargest(10, 'Price_Difference_Pct')
        top_overvalued = overvalued.nlargest(10, 'Price_Difference_Pct')
        
        return {
            'model_performance': {
                'mse': mse,
                'r2_score': r2
            },
            'feature_importance': feature_importance,
            'undervalued_vehicles': top_undervalued[['Brand', 'Model', 'MSRP', 'Predicted_Price', 'Price_Difference_Pct']],
            'overvalued_vehicles': top_overvalued[['Brand', 'Model', 'MSRP', 'Predicted_Price', 'Price_Difference_Pct']],
            'predictions': predictions
        }
    
    def predict_vehicle_segment(self):
        ml_data = self.prepare_features()
        
        if 'VehicleClass' not in ml_data.columns:
            return None
            
        feature_cols = ['EngineSize', 'Cylinders', 'HorsePower', 'Weight', 'MSRP', 'MPG_Average', 'Brand']
        available_features = [col for col in feature_cols if col in ml_data.columns]
        
        X = ml_data[available_features]
        y = ml_data['VehicleClass']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_segment_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.rf_segment_model.fit(X_train, y_train)
        
        y_pred = self.rf_segment_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.rf_segment_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_segment = feature_importance
        
        predictions = self.rf_segment_model.predict(ml_data[available_features])
        prediction_proba = self.rf_segment_model.predict_proba(ml_data[available_features])
        
        ml_data['Predicted_Segment'] = predictions
        ml_data['Segment_Confidence'] = prediction_proba.max(axis=1)
        
        misclassified = ml_data[ml_data['VehicleClass'] != ml_data['Predicted_Segment']]
        
        return {
            'model_performance': {
                'accuracy': accuracy,
                'total_classes': len(np.unique(y))
            },
            'feature_importance': feature_importance,
            'misclassified_vehicles': misclassified[['Brand', 'Model', 'VehicleClass', 'Predicted_Segment', 'Segment_Confidence']].head(10),
            'predictions': predictions,
            'confidence_scores': prediction_proba.max(axis=1)
        }
    
    def find_value_outliers(self):
        if self.rf_price_model is None or self.rf_efficiency_model is None:
            return None
            
        ml_data = self.prepare_features()
        
        price_features = [col for col in ['EngineSize', 'Cylinders', 'HorsePower', 'Weight', 'MPG_Average', 'Brand', 'VehicleClass', 'DriveTrain'] if col in ml_data.columns]
        efficiency_features = [col for col in ['EngineSize', 'Cylinders', 'HorsePower', 'Weight', 'Brand', 'VehicleClass', 'DriveTrain', 'Transmission'] if col in ml_data.columns]
        
        predicted_prices = self.rf_price_model.predict(ml_data[price_features])
        predicted_efficiency = self.rf_efficiency_model.predict(ml_data[efficiency_features])
        
        ml_data['ML_Value_Score'] = predicted_efficiency / (predicted_prices / 10000)
        ml_data['Actual_Value_Score'] = ml_data['MPG_Average'] / (ml_data['MSRP'] / 10000)
        ml_data['Value_Difference'] = ml_data['ML_Value_Score'] - ml_data['Actual_Value_Score']
        
        hidden_gems = ml_data[ml_data['Value_Difference'] > 0].nlargest(10, 'Value_Difference')
        overrated = ml_data[ml_data['Value_Difference'] < 0].nsmallest(10, 'Value_Difference')
        
        return {
            'hidden_gems': hidden_gems[['Brand', 'Model', 'MSRP', 'MPG_Average', 'ML_Value_Score', 'Actual_Value_Score', 'Value_Difference']],
            'overrated_vehicles': overrated[['Brand', 'Model', 'MSRP', 'MPG_Average', 'ML_Value_Score', 'Actual_Value_Score', 'Value_Difference']],
            'ml_predictions': {
                'predicted_prices': predicted_prices,
                'predicted_efficiency': predicted_efficiency,
                'ml_value_scores': ml_data['ML_Value_Score'].values
            }
        }
    
    def get_comprehensive_ml_insights(self):
        efficiency_results = self.predict_efficiency()
        price_results = self.predict_price()
        segment_results = self.predict_vehicle_segment()
        value_outliers = self.find_value_outliers()
        
        return {
            'efficiency_prediction': efficiency_results,
            'price_prediction': price_results,
            'segment_prediction': segment_results,
            'value_analysis': value_outliers
        }
