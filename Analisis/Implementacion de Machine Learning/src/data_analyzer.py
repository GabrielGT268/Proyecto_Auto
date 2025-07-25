import pandas as pd
import numpy as np
from ml_analyzer import MLAnalyzer

class DataAnalyzer:
    
    def __init__(self, data):
        self.data = data
        self.insights = {}
        self.ml_analyzer = MLAnalyzer(data)
        print(f"Análisis iniciado con {len(self.data)} vehículos")
    
    def calculate_value_score(self):
        self.data['Value_Score'] = self.data['MPG_Average'] / (self.data['MSRP'] / 10000)
        return self.data['Value_Score']
    
    def calculate_power_to_weight_ratio(self):
        if 'Power_to_Weight_Ratio' not in self.data.columns:
            self.data['Power_to_Weight_Ratio'] = (self.data['HorsePower'] / self.data['Weight'] * 1000)
        return self.data['Power_to_Weight_Ratio']
    
    def categorize_engines(self):
        self.data['Engine_Category'] = pd.cut(
            self.data['EngineSize'], 
            bins=[0, 2.0, 3.5, 5.0, float('inf')], 
            labels=['Pequeño (<2L)', 'Mediano (2-3.5L)', 'Grande (3.5-5L)', 'Muy Grande (>5L)']
        )
        return self.data['Engine_Category']
    
    def get_price_performance_stats(self):
        required_cols = ['MSRP', 'MPG_Average']
        if not all(col in self.data.columns for col in required_cols):
            return None
        
        correlation = self.data['MSRP'].corr(self.data['MPG_Average'])
        mean_efficiency = self.data['MPG_Average'].mean()
        
        if 'Value_Score' not in self.data.columns:
            self.calculate_value_score()
        
        top_quartile_threshold = self.data['Value_Score'].quantile(0.75)
        best_value_vehicles = self.data[self.data['Value_Score'] >= top_quartile_threshold]
        
        stats = {
            'correlation_price_efficiency': correlation,
            'mean_efficiency': mean_efficiency,
            'best_value_threshold': top_quartile_threshold,
            'best_value_count': len(best_value_vehicles)
        }
        
        self.insights['price_performance'] = stats
        return stats, best_value_vehicles
    
    def get_engine_efficiency_stats(self):
        engine_cols = ['EngineSize', 'Cylinders', 'HorsePower', 'Weight']
        if not all(col in self.data.columns for col in engine_cols):
            return None
        
        engine_corr = self.data['EngineSize'].corr(self.data['MPG_Average'])
        
        self.calculate_power_to_weight_ratio()
        power_weight_corr = self.data['Power_to_Weight_Ratio'].corr(self.data['MPG_Average'])
        
        self.categorize_engines()
        engine_stats = self.data.groupby('Engine_Category', observed=False).agg({
            'MPG_Average': ['mean', 'count'],
            'MSRP': 'mean'
        }).round(2)
        engine_stats.columns = ['MPG_Mean', 'Count', 'Price_Mean']
        
        cylinder_stats = self.data.groupby('Cylinders').agg({
            'MPG_Average': 'mean',
            'MSRP': 'mean',
            'Cylinders': 'count'
        }).round(2)
        cylinder_stats.columns = ['MPG_Average', 'MSRP', 'Count']
        cylinder_stats_filtered = cylinder_stats[cylinder_stats['Count'] >= 10].sort_index()
        
        stats = {
            'engine_size_efficiency_corr': engine_corr,
            'power_weight_efficiency_corr': power_weight_corr,
            'best_engine_category': engine_stats.loc[engine_stats['MPG_Mean'].idxmax()].name,
            'engine_categories_stats': engine_stats.to_dict(),
            'cylinder_stats': cylinder_stats_filtered.to_dict()
        }
        
        self.insights['engine_efficiency'] = stats
        return stats, engine_stats, cylinder_stats_filtered
    
    def get_vehicle_segments_stats(self):
        if 'VehicleClass' not in self.data.columns:
            return None
        
        segment_stats = self.data.groupby('VehicleClass').agg({
            'MSRP': 'mean',
            'MPG_Average': 'mean',
            'VehicleClass': 'count'
        }).round(2)
        segment_stats.columns = ['MSRP', 'MPG_Average', 'Count']
        
        best_segments = segment_stats.sort_values('MPG_Average', ascending=False).head(5)
        
        segment_counts = self.data['VehicleClass'].value_counts()
        
        price_ranges = self.data.groupby('VehicleClass')['MSRP'].agg(['min', 'max', 'mean', 'count']).round(0)
        price_ranges_sorted = price_ranges.sort_values('mean')
        
        stats = {
            'best_efficiency_segments': best_segments.to_dict(),
            'segment_positioning': segment_stats.to_dict(),
            'market_distribution': segment_counts.to_dict(),
            'price_ranges': price_ranges_sorted.to_dict()
        }
        
        self.insights['vehicle_segments'] = stats
        return stats, best_segments, segment_counts, price_ranges_sorted
    
    def get_purchase_recommendations(self):
        if 'Simple_Value' not in self.data.columns:
            self.data['Simple_Value'] = self.data['MPG_Average'] / (self.data['MSRP'] / 10000)
        
        recommendations = {}
        
        if 'Price_Category' in self.data.columns:
            available_categories = self.data['Price_Category'].unique()
            
            for category in available_categories:
                category_vehicles = self.data[self.data['Price_Category'] == category]
                if len(category_vehicles) > 0:
                    best_in_category = category_vehicles.nlargest(1, 'Simple_Value')
                    top_vehicle = best_in_category.iloc[0]
                    recommendations[category] = {
                        'brand_model': f"{top_vehicle['Brand']} {top_vehicle['Model']}",
                        'price': top_vehicle['MSRP'],
                        'efficiency': top_vehicle['MPG_Average'],
                        'value_score': top_vehicle['Simple_Value'],
                        'vehicle_class': top_vehicle['VehicleClass']
                    }
        else:
            price_quantiles = self.data['MSRP'].quantile([0.33, 0.67])
            categories = {
                'Económico': self.data[self.data['MSRP'] <= price_quantiles.iloc[0]],
                'Medio': self.data[(self.data['MSRP'] > price_quantiles.iloc[0]) & 
                                 (self.data['MSRP'] <= price_quantiles.iloc[1])],
                'Premium': self.data[self.data['MSRP'] > price_quantiles.iloc[1]]
            }
            
            for category, category_data in categories.items():
                if len(category_data) > 0:
                    best_in_category = category_data.nlargest(1, 'Simple_Value')
                    top_vehicle = best_in_category.iloc[0]
                    recommendations[category] = {
                        'brand_model': f"{top_vehicle['Brand']} {top_vehicle['Model']}",
                        'price': top_vehicle['MSRP'],
                        'efficiency': top_vehicle['MPG_Average'],
                        'value_score': top_vehicle['Simple_Value'],
                        'vehicle_class': top_vehicle['VehicleClass']
                    }
        
        top_10_overall = self.data.nlargest(10, 'Simple_Value')
        
        self.insights['purchase_recommendations'] = {
            'top_10_vehicles': top_10_overall[['Brand', 'Model', 'MSRP', 'MPG_Average', 'Simple_Value', 'VehicleClass']].to_dict('records'),
            'category_recommendations': recommendations
        }
        
        return recommendations, top_10_overall
    
    def get_ml_insights(self):
        ml_results = self.ml_analyzer.get_comprehensive_ml_insights()
        
        self.insights['machine_learning'] = {
            'efficiency_model': ml_results['efficiency_prediction']['model_performance'] if ml_results['efficiency_prediction'] else None,
            'price_model': ml_results['price_prediction']['model_performance'] if ml_results['price_prediction'] else None,
            'segment_model': ml_results['segment_prediction']['model_performance'] if ml_results['segment_prediction'] else None,
            'feature_importance_efficiency': ml_results['efficiency_prediction']['feature_importance'] if ml_results['efficiency_prediction'] else None,
            'feature_importance_price': ml_results['price_prediction']['feature_importance'] if ml_results['price_prediction'] else None,
            'hidden_gems': ml_results['value_analysis']['hidden_gems'] if ml_results['value_analysis'] else None,
            'overrated_vehicles': ml_results['value_analysis']['overrated_vehicles'] if ml_results['value_analysis'] else None,
            'undervalued_vehicles': ml_results['price_prediction']['undervalued_vehicles'] if ml_results['price_prediction'] else None,
            'overvalued_vehicles': ml_results['price_prediction']['overvalued_vehicles'] if ml_results['price_prediction'] else None
        }
        
        return ml_results
    
    def get_all_insights(self):
        return self.insights
