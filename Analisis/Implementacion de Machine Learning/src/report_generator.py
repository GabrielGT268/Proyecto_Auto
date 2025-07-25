class ReportGenerator:
    
    def __init__(self, insights, data):
        self.insights = insights
        self.data = data
    
    def print_price_performance_summary(self, stats):
        if not stats:
            print("Datos de precio-rendimiento no disponibles")
            return
        
        correlation = stats['correlation_price_efficiency']
        print(f"Correlación Precio-Eficiencia: {correlation:.3f}")
        print(f"Eficiencia promedio: {stats['mean_efficiency']:.1f} MPG")
        print(f"Vehículos con mejor valor (top 25%): {stats['best_value_count']}")
        print("Análisis guardado en: results/price_performance_analysis.png")
    
    def print_engine_efficiency_summary(self, stats):
        if not stats:
            print("Datos del motor no disponibles")
            return
        
        print(f"Correlación Tamaño Motor-Eficiencia: {stats['engine_size_efficiency_corr']:.3f}")
        print(f"Correlación Potencia/Peso-Eficiencia: {stats['power_weight_efficiency_corr']:.3f}")
        print("Análisis guardado en: results/engine_efficiency_analysis.png")
    
    def print_vehicle_segments_summary(self, best_segments):
        if best_segments is None:
            print("Datos de segmentos no disponibles")
            return
        
        print("Top 5 segmentos más eficientes:")
        for i, (segment, mpg) in enumerate(best_segments['MPG_Average'].items(), 1):
            print(f"  {i}. {segment}: {mpg:.1f} MPG")
        print("Análisis guardado en: results/vehicle_segments_analysis.png")
    
    def print_ml_insights_summary(self, ml_insights):
        if not ml_insights:
            print("Datos de Machine Learning no disponibles")
            return
        
        print("\n=== RESUMEN MACHINE LEARNING ===")
        
        if ml_insights.get('efficiency_model'):
            eff_perf = ml_insights['efficiency_model']
            print(f"Modelo Predicción Eficiencia:")
            print(f"  - R² Score: {eff_perf['r2_score']:.3f}")
            if 'cv_mean' in eff_perf:
                print(f"  - Validación Cruzada: {eff_perf['cv_mean']:.3f} ± {eff_perf['cv_std']:.3f}")
        
        if ml_insights.get('price_model'):
            price_perf = ml_insights['price_model']
            print(f"Modelo Predicción Precio:")
            print(f"  - R² Score: {price_perf['r2_score']:.3f}")
        
        if ml_insights.get('segment_model'):
            seg_perf = ml_insights['segment_model']
            print(f"Modelo Clasificación Segmentos:")
            print(f"  - Accuracy: {seg_perf['accuracy']:.3f}")
        
        if ml_insights.get('feature_importance_efficiency') is not None:
            top_feature = ml_insights['feature_importance_efficiency'].iloc[0]
            print(f"Factor más importante para eficiencia: {top_feature['feature']} ({top_feature['importance']:.3f})")
        
        if ml_insights.get('hidden_gems') is not None and len(ml_insights['hidden_gems']) > 0:
            print(f"Joyas ocultas identificadas: {len(ml_insights['hidden_gems'])}")
            top_gem = ml_insights['hidden_gems'].iloc[0]
            print(f"  - Mejor: {top_gem['Brand']} {top_gem['Model']}")
        
        if ml_insights.get('undervalued_vehicles') is not None and len(ml_insights['undervalued_vehicles']) > 0:
            print(f"Vehículos subvalorados: {len(ml_insights['undervalued_vehicles'])}")
        
        print("Análisis ML guardado en: results/machine_learning_analysis.png")
    
    def print_purchase_recommendations_summary(self, recommendations, top_10_overall):
        if top_10_overall is None or len(top_10_overall) == 0:
            print("No se encontraron recomendaciones")
            return
        
        print(f"Mejor vehículo general: {top_10_overall.iloc[0]['Brand']} {top_10_overall.iloc[0]['Model']}")
        print(f"  - Precio: ${top_10_overall.iloc[0]['MSRP']:,.0f}")
        print(f"  - Eficiencia: {top_10_overall.iloc[0]['MPG_Average']:.1f} MPG")
        print(f"  - Puntuación de valor: {top_10_overall.iloc[0]['Simple_Value']:.1f}")
        
        if recommendations:
            print(f"\nMejores por categoría de precio ({len(recommendations)} categorías):")
            for category, vehicle in recommendations.items():
                print(f"  {category}: {vehicle['brand_model']} - ${vehicle['price']:,.0f} (Valor: {vehicle['value_score']:.1f})")
        
        print("Análisis guardado en: results/purchase_recommendations.png")
    
    def generate_executive_summary(self):
        print("\n" + "="*60)
        print("RESUMEN FINAL DE ANÁLISIS PRECIO-RENDIMIENTO DE AUTOMÓVILES")
        print("="*60)
        
        summary = {}
        
        total_vehicles = len(self.data)
        avg_price = self.data['MSRP'].mean()
        avg_efficiency = self.data['MPG_Average'].mean()
        
        print(f"\n [1] ESTADÍSTICAS GENERALES:")
        print(f"   • Total de vehículos analizados: {total_vehicles}")
        print(f"   • Precio promedio: ${avg_price:,.0f}")
        print(f"   • Eficiencia promedio: {avg_efficiency:.1f} MPG")
        
        if 'price_performance' in self.insights:
            correlation = self.insights['price_performance']['correlation_price_efficiency']
            summary['price_efficiency_correlation'] = correlation
            print(f"\n[2] RELACIÓN PRECIO-EFICIENCIA:")
            print(f"   • Correlación: {correlation:.3f}")
            
            if correlation < -0.3:
                print("   • Los vehículos más caros tienden a ser MENOS eficientes")
            elif correlation > 0.3:
                print("   • Los vehículos más caros tienden a ser MÁS eficientes")
            else:
                print("   • No hay relación fuerte entre precio y eficiencia")
        
        if 'engine_efficiency' in self.insights:
            engine_corr = self.insights['engine_efficiency']['engine_size_efficiency_corr']
            best_engine = self.insights['engine_efficiency']['best_engine_category']
            summary['engine_efficiency'] = {'correlation': engine_corr, 'best_category': best_engine}
            print(f"\n[3] FACTORES DEL MOTOR:")
            print(f"   • Correlación tamaño motor-eficiencia: {engine_corr:.3f}")
            print(f"   • Mejor categoría de motor: {best_engine}")
            
            if engine_corr < -0.5:
                print("   • Motores más pequeños son significativamente más eficientes")
                print("   • Recomendación: Priorizar motores de menor cilindrada")
        
        if 'vehicle_segments' in self.insights:
            best_segments = self.insights['vehicle_segments']['best_efficiency_segments']
            if 'MPG_Average' in best_segments:
                top_segment = list(best_segments['MPG_Average'].keys())[0]
                summary['best_segment'] = top_segment
                print(f"\n[4] SEGMENTOS DE VEHÍCULOS:")
                print(f"   • Segmento más eficiente: {top_segment}")
                print("   • Top 5 segmentos por eficiencia:")
                for i, (segment, mpg) in enumerate(list(best_segments['MPG_Average'].items())[:5], 1):
                    print(f"     {i}. {segment} ({mpg:.1f} MPG)")
        
        if 'purchase_recommendations' in self.insights:
            top_vehicle = self.insights['purchase_recommendations']['top_10_vehicles'][0]
            category_recs = self.insights['purchase_recommendations']['category_recommendations']
            
            print(f"\n[5] RECOMENDACIONES DE COMPRA:")
            print(f"   • Mejor vehículo general: {top_vehicle['Brand']} {top_vehicle['Model']}")
            print(f"     - Precio: ${top_vehicle['MSRP']:,.0f}")
            print(f"     - Eficiencia: {top_vehicle['MPG_Average']:.1f} MPG")
            print(f"     - Puntuación valor: {top_vehicle['Simple_Value']:.1f}")
            
            if category_recs:
                print(f"\n   • Mejores por categoría de precio ({len(category_recs)} categorías):")
                for category, vehicle in category_recs.items():
                    print(f"     - {category}: {vehicle['brand_model']} (${vehicle['price']:,.0f})")
        
        if 'machine_learning' in self.insights:
            ml_data = self.insights['machine_learning']
            
            print(f"\n[6] MACHINE LEARNING INSIGHTS:")
            if ml_data.get('efficiency_model'):
                print(f"   • Modelo Eficiencia R²: {ml_data['efficiency_model']['r2_score']:.3f}")
            if ml_data.get('price_model'):
                print(f"   • Modelo Precio R²: {ml_data['price_model']['r2_score']:.3f}")
            if ml_data.get('hidden_gems') is not None and len(ml_data['hidden_gems']) > 0:
                print(f"   • Joyas ocultas encontradas: {len(ml_data['hidden_gems'])}")
            if ml_data.get('undervalued_vehicles') is not None and len(ml_data['undervalued_vehicles']) > 0:
                print(f"   • Vehículos subvalorados: {len(ml_data['undervalued_vehicles'])}")
        
        summary_data = {
            'total_vehicles_analyzed': total_vehicles,
            'average_price': avg_price,
            'average_efficiency': avg_efficiency,
            'key_insights': summary
        }
        
        print("[i] Todos los gráficos disponibles en la carpeta 'results/'")
        return summary_data
    
    def print_analysis_completion_summary(self):
        print("\n" + "="*70)
        print("[i] ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*70)
        print("Archivos generados:")
        print("  • results/price_performance_analysis.png")
        print("  • results/engine_efficiency_analysis.png") 
        print("  • results/vehicle_segments_analysis.png")
        print("  • results/purchase_recommendations.png")
        print("  • results/machine_learning_analysis.png")
