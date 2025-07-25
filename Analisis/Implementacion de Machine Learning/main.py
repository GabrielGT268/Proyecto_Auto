import sys
import os
sys.path.append('src')

from data_preprocessing import DataProcessor
from data_analyzer import DataAnalyzer
from visualization_helper import PlotHelper
from report_generator import ReportGenerator
from ml_analyzer import MLAnalyzer
import matplotlib.pyplot as plt

def run_modular_analysis(analyzer, plotter, data, ml_analyzer=None):

    print("Mostrando resultados...")
    all_insights = {}
    
    print("\n=== [1] RELACIÓN PRECIO-RENDIMIENTO ===")
    stats, best_value_vehicles = analyzer.get_price_performance_stats()
    if stats:
        fig, axes = plotter.setup_subplot_grid(
            2, 2, figsize=(15, 12), 
            title='Análisis Principal: Precio vs Rendimiento del Combustible'
        )
        
        correlation = plotter.create_scatter_with_trendline(
            axes[0, 0], data['MSRP'], data['MPG_Average'],
            'Precio (MSRP) - USD', 'Eficiencia Promedio (MPG)',
            'Precio vs Eficiencia de Combustible'
        )
        
        if ml_analyzer:
            efficiency_results = ml_analyzer.predict_efficiency()
            if efficiency_results and 'predictions' in efficiency_results:
                axes[0, 0].scatter(data['MSRP'], efficiency_results['predictions'], 
                                 alpha=0.4, s=30, color='purple', label='Predicción Eficiencia | RandomForest')
                axes[0, 0].legend()
        
        plotter.create_histogram_with_mean(
            axes[0, 1], data['MPG_Average'],
            'Eficiencia de Combustible (MPG)', 'Número de Vehículos',
            'Distribución de Eficiencia de Combustible'
        )
        
        if 'Price_Category' in data.columns:
            price_eff_stats = data.groupby('Price_Category', observed=False).agg({
                'MPG_Average': ['mean', 'std', 'count'],
                'MSRP': 'mean'
            }).round(2)
            price_eff_stats.columns = ['MPG_Mean', 'MPG_Std', 'Count', 'Price_Mean']
            
            plotter.create_category_bar_chart(
                axes[1, 0], price_eff_stats, 'MPG_Mean',
                'Categorías de Precio', 'Eficiencia Promedio (MPG)',
                'Eficiencia por Categoría de Precio'
            )
        
        axes[1, 1].scatter(data['MSRP'], data['MPG_Average'], 
                         alpha=0.6, s=50, color='lightcoral', label='Todos los vehículos')
        axes[1, 1].scatter(best_value_vehicles['MSRP'], best_value_vehicles['MPG_Average'], 
                         alpha=0.8, s=60, color='green', label=f'Mejor valor (Top 25%)')
        
        if ml_analyzer:
            value_outliers = ml_analyzer.find_value_outliers()
            if value_outliers and 'hidden_gems' in value_outliers:
                hidden_gems = value_outliers['hidden_gems']
                if len(hidden_gems) > 0:
                    axes[1, 1].scatter(hidden_gems['MSRP'], hidden_gems['MPG_Average'], 
                                     alpha=0.9, s=80, color='gold', marker='*', 
                                     label='Carros infravalorados')
        
        axes[1, 1].set_xlabel('Precio (MSRP) - USD')
        axes[1, 1].set_ylabel('Eficiencia (MPG)')
        axes[1, 1].set_title('Identificación de Vehículos con Mejor Valor')
        axes[1, 1].legend()
        
        plotter.save_and_show_plot('price_performance_analysis.png')
        
        print(f"Correlación Precio-Eficiencia: {stats['correlation_price_efficiency']:.3f}")
        print(f"Eficiencia promedio: {stats['mean_efficiency']:.1f} MPG")
        print(f"Vehículos con mejor valor (top 25%): {stats['best_value_count']}")
        print("Análisis guardado en: results/price_performance_analysis.png")
        
        all_insights['price_performance'] = stats
    
    print("\n=== [2] FACTORES DE EFICIENCIA DEL MOTOR ===")
    engine_data = analyzer.get_engine_efficiency_stats()
    if engine_data:
        stats, engine_stats, cylinder_stats = engine_data
        
        fig, axes = plotter.setup_subplot_grid(
            2, 2, figsize=(14, 10),
            title='Factores del Motor que Influyen en la Eficiencia'
        )
        
        plotter.create_scatter_with_trendline(
            axes[0, 0], data['EngineSize'], data['MPG_Average'],
            'Tamaño del Motor (L)', 'Eficiencia Promedio (MPG)',
            'Tamaño del Motor vs Eficiencia'
        )
        
        plotter.create_scatter_with_trendline(
            axes[0, 1], data['Power_to_Weight_Ratio'], data['MPG_Average'],
            'Relación Potencia/Peso (HP/1000 lbs)', 'Eficiencia Promedio (MPG)',
            'Relación Potencia/Peso vs Eficiencia', color='coral'
        )
        
        if ml_analyzer:
            efficiency_results = ml_analyzer.predict_efficiency()
            if efficiency_results and 'feature_importance' in efficiency_results:
                feature_imp = efficiency_results['feature_importance'].head(5)
                bars = axes[1, 0].bar(range(len(feature_imp)), feature_imp['importance'], 
                                    alpha=0.8, color='orange')
                axes[1, 0].set_xticks(range(len(feature_imp)))
                axes[1, 0].set_xticklabels(feature_imp['feature'], rotation=45, ha='right')
                axes[1, 0].set_ylabel('Importancia')
                axes[1, 0].set_title('Top 5 Factores - Predicción de Eficiencia')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
            else:
                plotter.create_category_bar_chart(
                    axes[1, 0], engine_stats, 'MPG_Mean',
                    'Categorías de Motor', 'Eficiencia Promedio (MPG)',
                    'Eficiencia por Categoría de Motor', color='lightgreen'
                )
        else:
            plotter.create_category_bar_chart(
                axes[1, 0], engine_stats, 'MPG_Mean',
                'Categorías de Motor', 'Eficiencia Promedio (MPG)',
                'Eficiencia por Categoría de Motor', color='lightgreen'
            )
        
        if len(cylinder_stats) > 0:
            bars = axes[1, 1].bar(range(len(cylinder_stats)), cylinder_stats['MPG_Average'], 
                                alpha=0.7, color='skyblue')
            axes[1, 1].set_xticks(range(len(cylinder_stats)))
            axes[1, 1].set_xticklabels([f'{int(cyl)} Cil.' for cyl in cylinder_stats.index])
            axes[1, 1].set_ylabel('Eficiencia Promedio (MPG)')
            axes[1, 1].set_title(f'Eficiencia por Número de Cilindros')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plotter.save_and_show_plot('engine_efficiency_analysis.png')
        
        print(f"Correlación Tamaño Motor-Eficiencia: {stats['engine_size_efficiency_corr']:.3f}")
        print(f"Correlación Potencia/Peso-Eficiencia: {stats['power_weight_efficiency_corr']:.3f}")
        print("Análisis guardado en: results/engine_efficiency_analysis.png")
        
        all_insights['engine_efficiency'] = stats
    
    print("\n=== [3] SEGMENTOS DE VEHÍCULOS PARA DECISIÓN DE COMPRA ===")
    segment_data = analyzer.get_vehicle_segments_stats()
    if segment_data:
        stats, best_segments, segment_counts, price_ranges = segment_data
        
        fig, axes = plotter.setup_subplot_grid(
            2, 2, figsize=(15, 12),
            title='Segmentos de Vehículos: Análisis para Decisión de Compra'
        )
        
        plotter.create_segment_scatter(
            axes[0, 0], data, 'VehicleClass', 'MSRP', 'MPG_Average',
            'Precio (USD)', 'Eficiencia (MPG)',
            'Posicionamiento de Segmentos: Precio vs Eficiencia'
        )
        
        plotter.create_category_bar_chart(
            axes[0, 1], best_segments, 'MPG_Average',
            'Segmentos', 'Eficiencia Promedio (MPG)',
            'Top 5 Segmentos más Eficientes', color='green', add_labels=False
        )
        
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, 
                      autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        axes[1, 0].set_title('Distribución del Mercado por Segmento')
        
        bars = axes[1, 1].bar(range(len(price_ranges)), price_ranges['mean'], 
                             alpha=0.7, color='skyblue')
        axes[1, 1].set_xticks(range(len(price_ranges)))
        axes[1, 1].set_xticklabels(price_ranges.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Precio Promedio (USD)')
        axes[1, 1].set_title('Precio Promedio por Segmento')
        
        plotter.save_and_show_plot('vehicle_segments_analysis.png')
        
        print("Top 5 segmentos más eficientes:")
        for i, (segment, mpg) in enumerate(best_segments['MPG_Average'].items(), 1):
            print(f"  {i}. {segment}: {mpg:.1f} MPG")
        print("Análisis guardado en: results/vehicle_segments_analysis.png")
        
        all_insights['vehicle_segments'] = stats
    
    if ml_analyzer:
        print("\n=== [4] MACHINE LEARNING INSIGHTS ===")
        ml_results = ml_analyzer.get_comprehensive_ml_insights()
        if ml_results:
            fig, axes = plotter.setup_subplot_grid(
                2, 3, figsize=(18, 12),
                title='Análisis de Machine Learning: Predicciones y Patrones Ocultos'
            )
            
            if ml_results['efficiency_prediction']:
                eff_data = ml_results['efficiency_prediction']
                feature_imp = eff_data['feature_importance'].head(6)
                bars = axes[0, 0].bar(range(len(feature_imp)), feature_imp['importance'], 
                                    alpha=0.8, color='orange')
                axes[0, 0].set_xticks(range(len(feature_imp)))
                axes[0, 0].set_xticklabels(feature_imp['feature'], rotation=45, ha='right')
                axes[0, 0].set_ylabel('Importancia')
                axes[0, 0].set_title('Predicción de Eficiencia | RandomForest')
                axes[0, 0].grid(True, alpha=0.3, axis='y')
                
                if 'predictions' in eff_data:
                    axes[0, 1].scatter(data['MPG_Average'], eff_data['predictions'], 
                                     alpha=0.6, s=30, color='blue')
                    axes[0, 1].plot([data['MPG_Average'].min(), data['MPG_Average'].max()], 
                                   [data['MPG_Average'].min(), data['MPG_Average'].max()], 
                                   'r--', alpha=0.8)
                    axes[0, 1].set_xlabel('MPG Real')
                    axes[0, 1].set_ylabel('MPG Predicho')
                    axes[0, 1].set_title('Predicción vs Real - Eficiencia')
                
                r2_score = eff_data['model_performance']['r2_score']
                axes[0, 2].text(0.5, 0.5, f'R² Score\n{r2_score:.3f}', 
                               ha='center', va='center', fontsize=20, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
                axes[0, 2].set_xlim(0, 1)
                axes[0, 2].set_ylim(0, 1)
                axes[0, 2].set_title('Precisión Modelo Eficiencia')
                axes[0, 2].axis('off')
            
            if ml_results['price_prediction']:
                price_data = ml_results['price_prediction']
                feature_imp = price_data['feature_importance'].head(6)
                bars = axes[1, 0].bar(range(len(feature_imp)), feature_imp['importance'], 
                                    alpha=0.8, color='purple')
                axes[1, 0].set_xticks(range(len(feature_imp)))
                axes[1, 0].set_xticklabels(feature_imp['feature'], rotation=45, ha='right')
                axes[1, 0].set_ylabel('Importancia')
                axes[1, 0].set_title('Predicción de Precio | RandomForest')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                if not price_data['undervalued_vehicles'].empty:
                    top_undervalued = price_data['undervalued_vehicles'].head(5)
                    vehicle_names = [f"{row['Brand']} {row['Model']}"[:12] for _, row in top_undervalued.iterrows()]
                    axes[1, 1].barh(range(len(top_undervalued)), top_undervalued['Price_Difference_Pct'],
                                   alpha=0.7, color='green')
                    axes[1, 1].set_yticks(range(len(top_undervalued)))
                    axes[1, 1].set_yticklabels(vehicle_names)
                    axes[1, 1].set_xlabel('% Infravaloración')
                    axes[1, 1].set_title('Top 5 Vehículos Infravalorados')
                    axes[1, 1].grid(True, alpha=0.3, axis='x')
                
                r2_score = price_data['model_performance']['r2_score']
                axes[1, 2].text(0.5, 0.5, f'R² Score\n{r2_score:.3f}', 
                               ha='center', va='center', fontsize=20, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[1, 2].set_xlim(0, 1)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].set_title('Precisión Modelo Precio')
                axes[1, 2].axis('off')
            
            plotter.save_and_show_plot('machine_learning_analysis.png')
            
            if ml_results['efficiency_prediction']:
                print(f"Eficiencia de Modelo - R²: {ml_results['efficiency_prediction']['model_performance']['r2_score']:.3f}")
                print(f"Factor más importante: {ml_results['efficiency_prediction']['feature_importance'].iloc[0]['feature']}")
            
            if ml_results['price_prediction']:
                print(f"Modelo Precio - R²: {ml_results['price_prediction']['model_performance']['r2_score']:.3f}")
                print(f"Vehículos subvalorados identificados: {len(ml_results['price_prediction']['undervalued_vehicles'])}")
            
            print("Análisis de ML guardado en: results/machine_learning_analysis.png")
            
            all_insights['machine_learning'] = ml_results
     
    print("\n=== [4] MACHINE LEARNING INSIGHTS ===")
    ml_results = analyzer.get_ml_insights()
    if ml_results:
        fig, axes = plotter.setup_subplot_grid(
            2, 3, figsize=(18, 12),
            title='Análisis con RandomForest: Predicciones y Patrones Ocultos'
        )
        
        if ml_results['efficiency_prediction']:
            eff_data = ml_results['efficiency_prediction']
            plotter.create_feature_importance_chart(
                axes[0, 0], eff_data['feature_importance'],
                'Factores más Importantes para Eficiencia'
            )
            
            if 'predictions' in eff_data:
                plotter.create_ml_prediction_scatter(
                    axes[0, 1], data['MPG_Average'], eff_data['predictions'],
                    'Predicción de Eficiencia (RT)', 'MPG Real', 'MPG Predicho'
                )
            
            plotter.create_ml_insights_overview(
                axes[0, 2], eff_data['model_performance'],
                'Rendimiento Modelo Eficiencia'
            )
        
        if ml_results['price_prediction']:
            price_data = ml_results['price_prediction']
            plotter.create_feature_importance_chart(
                axes[1, 0], price_data['feature_importance'],
                'Factores más Importantes para Precio'
            )
            
            if 'predictions' in price_data:
                plotter.create_ml_prediction_scatter(
                    axes[1, 1], data['MSRP'], price_data['predictions'],
                    'Predicción de Precio (RT)', 'Precio Real (USD)', 'Precio Predicho (USD)'
                )
            
            if not price_data['undervalued_vehicles'].empty:
                top_undervalued = price_data['undervalued_vehicles'].head(5)
                axes[1, 2].barh(range(len(top_undervalued)), top_undervalued['Price_Difference_Pct'],
                               alpha=0.7, color='green')
                axes[1, 2].set_yticks(range(len(top_undervalued)))
                axes[1, 2].set_yticklabels([f"{row['Brand']} {row['Model']}"[:15] 
                                          for _, row in top_undervalued.iterrows()])
                axes[1, 2].set_xlabel('% Subvalorado')
                axes[1, 2].set_title('Top 5 Vehículos Subvalorados')
                axes[1, 2].grid(True, alpha=0.3, axis='x')
        
        plotter.save_and_show_plot('machine_learning_analysis.png')
        
        if ml_results['efficiency_prediction']:
            print(f"Modelo Eficiencia - R²: {ml_results['efficiency_prediction']['model_performance']['r2_score']:.3f}")
            print(f"Factor más importante: {ml_results['efficiency_prediction']['feature_importance'].iloc[0]['feature']}")
        
        if ml_results['price_prediction']:
            print(f"Modelo Precio - R²: {ml_results['price_prediction']['model_performance']['r2_score']:.3f}")
            print(f"Vehículos subvalorados identificados: {len(ml_results['price_prediction']['undervalued_vehicles'])}")
        
        if ml_results['value_analysis'] and ml_results['value_analysis']['hidden_gems'] is not None:
            print(f"Joyas ocultas encontradas: {len(ml_results['value_analysis']['hidden_gems'])}")
        
        print("Análisis ML guardado en: results/machine_learning_analysis.png")
        
        all_insights['machine_learning'] = analyzer.insights.get('machine_learning', {})
    
    print("\n=== RECOMENDACIONES DE COMPRA ===")
    
    recommendations_data = analyzer.get_purchase_recommendations()
    if recommendations_data:
        recommendations, top_10_overall = recommendations_data
        if 'Simple_Value' not in top_10_overall.columns:
            top_10_overall['Simple_Value'] = top_10_overall['MPG_Average'] / (top_10_overall['MSRP'] / 10000)
    else:
        data['Simple_Value'] = data['MPG_Average'] / (data['MSRP'] / 10000)
        top_10_overall = data.nlargest(10, 'Simple_Value')
        recommendations = {}

    fig, axes = plotter.setup_subplot_grid(
        2, 2, figsize=(16, 12),
        title='Recomendaciones de Compra: Mejores Vehículos por Categoría'
    )
    
    plotter.create_top_vehicles_chart(
        axes[0, 0], top_10_overall, 'Simple_Value',
        'Top 10 Vehículos con Mejor Valor\n(Fórmula: MPG ÷ (Precio/10000))'
    )
    
    if ml_analyzer:
        value_outliers = ml_analyzer.find_value_outliers()
        if value_outliers and 'hidden_gems' in value_outliers:
            hidden_gems = value_outliers['hidden_gems'].head(8)
            if len(hidden_gems) > 0:
                plotter.create_top_vehicles_chart(
                    axes[0, 1], hidden_gems, 'ML_Value_Score',
                    'Top 8 infravaloradas\n(Basado en predicciones de ML)'
                )
        
        if value_outliers and 'overrated_vehicles' in value_outliers:
            overrated = value_outliers['overrated_vehicles'].head(6)
            if len(overrated) > 0:
                vehicle_names = [f"{row['Brand']} {row['Model']}"[:15] for _, row in overrated.iterrows()]
                axes[1, 0].barh(range(len(overrated)), overrated['Value_Difference'].abs(),
                               alpha=0.7, color='red')
                axes[1, 0].set_yticks(range(len(overrated)))
                axes[1, 0].set_yticklabels(vehicle_names)
                axes[1, 0].set_xlabel('Diferencia de Valor')
                axes[1, 0].set_title('Vehículos Sobrevalorados')
                axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    if recommendations:
        sorted_categories = sorted(recommendations.items(), 
                                 key=lambda x: x[1]['value_score'], reverse=True)
        categories = [item[0] for item in sorted_categories]
        values = [item[1]['value_score'] for item in sorted_categories]
        
        colors = ['lightblue', 'orange', 'lightcoral', 'lightgreen', 'lavender'][:len(categories)]
        if ml_analyzer:
            bars = axes[1, 1].bar(categories, values, alpha=0.7, color=colors)
            axes[1, 1].set_ylabel('Puntuación de Valor')
            axes[1, 1].set_title(f'Mejor Vehículo por Categoría de Precio')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            for bar, cat in zip(bars, categories):
                vehicle_info = recommendations[cat]
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f"{vehicle_info['brand_model'][:20]}...\n${vehicle_info['price']:,.0f}\n{vehicle_info['efficiency']:.1f} MPG", 
                           ha='center', va='bottom', fontsize=8)
        else:
            bars = axes[0, 1].bar(categories, values, alpha=0.7, color=colors)
            axes[0, 1].set_ylabel('Puntuación de Valor')
            axes[0, 1].set_title(f'Mejor Vehículo por Categoría de Precio')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, cat in zip(bars, categories):
                vehicle_info = recommendations[cat]
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f"{vehicle_info['brand_model'][:20]}...\n${vehicle_info['price']:,.0f}\n{vehicle_info['efficiency']:.1f} MPG", 
                           ha='center', va='bottom', fontsize=8)
    
    plotter.save_and_show_plot('purchase_recommendations.png')
    
    # Preparar columnas para el diccionario final
    required_columns = ['Brand', 'Model', 'MSRP', 'MPG_Average', 'Simple_Value']
    available_columns = [col for col in required_columns if col in top_10_overall.columns]
    if 'VehicleClass' in top_10_overall.columns:
        available_columns.append('VehicleClass')
    
    all_insights['purchase_recommendations'] = {
        'top_10_vehicles': top_10_overall[available_columns].to_dict('records'),
        'category_recommendations': recommendations
    }
    
    return {
        'best_value_vehicles': best_value_vehicles,
        'recommendations': recommendations,
        'insights': all_insights
    }

def run_analysis():

    try:
        print("\nCARGANDO Y PROCESANDO DATOS...")
        print("-" * 50)
        
        processor = DataProcessor('data/Proyecto.xlsx')
        
        data = processor.load_data()
        if data is None:
            print("[X] Error: No se pudieron cargar los datos")
            return
        
        processor.clean_data()
        processor.encode_categorical_variables()
        
        print(f"DATOS CARGADOS Y PROCESADOS")
        print(f"   > Total de vehículos a analizar: {len(processor.processed_data)}")
        print("-" * 50)
        
        analyzer = DataAnalyzer(processor.processed_data)
        plotter = PlotHelper()
        ml_analyzer = MLAnalyzer(processor.processed_data)
        
        results = run_modular_analysis(analyzer, plotter, processor.processed_data, ml_analyzer)
        
        if results and 'insights' in results:
            reporter = ReportGenerator(results['insights'], processor.processed_data)
            
            if 'price_performance' in results['insights']:
                reporter.print_price_performance_summary(results['insights']['price_performance'])
            
            if 'engine_efficiency' in results['insights']:
                reporter.print_engine_efficiency_summary(results['insights']['engine_efficiency'])
            
            if 'vehicle_segments' in results['insights']:
                segment_data = analyzer.get_vehicle_segments_stats()
                if segment_data:
                    _, best_segments, _, _ = segment_data
                    reporter.print_vehicle_segments_summary(best_segments)
            
            if 'machine_learning' in results['insights']:
                reporter.print_ml_insights_summary(results['insights']['machine_learning'])
            
            if 'recommendations' in results and 'purchase_recommendations' in results['insights']:
                reporter.print_purchase_recommendations_summary(results['recommendations'], 
                                                               analyzer.get_purchase_recommendations()[1])
            
            summary = reporter.generate_executive_summary()
            reporter.print_analysis_completion_summary()
        
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n[X] ERROR DURANTE EL ANÁLISIS")
        print(f"   {str(e)}")
        return None

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')
        print("[>] Carpeta 'results' creada")
    
    results = run_analysis()
    
    if results:
        print(f"\n[>] Finalizando.. Revisa los gráficos en la carpeta 'results'")
    else:
        print(f"\n[X] Occurió un error durante el análisis.")
