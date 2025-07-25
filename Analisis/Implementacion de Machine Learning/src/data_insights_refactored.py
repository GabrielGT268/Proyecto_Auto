from data_analyzer import DataAnalyzer
from visualization_helper import PlotHelper
from report_generator import ReportGenerator
import matplotlib.pyplot as plt
import seaborn as sns

class DataInsightsAnalyzer:
    
    def __init__(self, data):

        self.data = data
        self.analyzer = DataAnalyzer(data)
        self.plotter = PlotHelper()
        self.reporter = None  # Will be initialized after analysis
        
    def analyze_price_performance_relationship(self):
        print("=== ANÁLISIS: RELACIÓN PRECIO-RENDIMIENTO ===")
        
        stats, best_value_vehicles = self.analyzer.get_price_performance_stats()
        if not stats:
            print("Datos necesarios no disponibles para este análisis")
            return None
        
        fig, axes = self.plotter.setup_subplot_grid(
            2, 2, figsize=(15, 12), 
            title='Análisis Principal: Precio vs Rendimiento del Combustible'
        )
        
        correlation = self.plotter.create_scatter_with_trendline(
            axes[0, 0], self.data['MSRP'], self.data['MPG_Average'],
            'Precio (MSRP) - USD', 'Eficiencia Promedio (MPG)',
            'Precio vs Eficiencia de Combustible'
        )
        
        self.plotter.create_histogram_with_mean(
            axes[0, 1], self.data['MPG_Average'],
            'Eficiencia de Combustible (MPG)', 'Número de Vehículos',
            'Distribución de Eficiencia de Combustible'
        )
        
        if 'Price_Category' in self.data.columns:
            price_eff_stats = self.data.groupby('Price_Category').agg({
                'MPG_Average': ['mean', 'std', 'count'],
                'MSRP': 'mean'
            }).round(2)
            price_eff_stats.columns = ['MPG_Mean', 'MPG_Std', 'Count', 'Price_Mean']
            
            self.plotter.create_category_bar_chart(
                axes[1, 0], price_eff_stats, 'MPG_Mean',
                'Categorías de Precio', 'Eficiencia Promedio (MPG)',
                'Eficiencia por Categoría de Precio'
            )
        
        axes[1, 1].scatter(self.data['MSRP'], self.data['MPG_Average'], 
                         alpha=0.6, s=50, color='lightcoral', label='Todos los vehículos')
        axes[1, 1].scatter(best_value_vehicles['MSRP'], best_value_vehicles['MPG_Average'], 
                         alpha=0.8, s=60, color='green', label=f'Mejor valor (Top 25%)')
        axes[1, 1].set_xlabel('Precio (MSRP) - USD')
        axes[1, 1].set_ylabel('Eficiencia (MPG)')
        axes[1, 1].set_title('Identificación de Vehículos con Mejor Valor')
        axes[1, 1].legend()
        
        self.plotter.save_and_show_plot('price_performance_analysis.png')
        
        return best_value_vehicles
    
    def analyze_engine_efficiency_factors(self):
        print("\n=== ANÁLISIS: FACTORES DE EFICIENCIA DEL MOTOR ===")
        
        stats, engine_stats, cylinder_stats = self.analyzer.get_engine_efficiency_stats()
        if not stats:
            print("Datos del motor no disponibles para este análisis")
            return
        
        fig, axes = self.plotter.setup_subplot_grid(
            2, 2, figsize=(14, 10),
            title='Factores del Motor que Influyen en la Eficiencia'
        )
        
        self.plotter.create_scatter_with_trendline(
            axes[0, 0], self.data['EngineSize'], self.data['MPG_Average'],
            'Tamaño del Motor (L)', 'Eficiencia Promedio (MPG)',
            'Tamaño del Motor vs Eficiencia'
        )
        
        self.plotter.create_scatter_with_trendline(
            axes[0, 1], self.data['Power_to_Weight_Ratio'], self.data['MPG_Average'],
            'Relación Potencia/Peso (HP/1000 lbs)', 'Eficiencia Promedio (MPG)',
            'Relación Potencia/Peso vs Eficiencia', color='coral'
        )
        
        self.plotter.create_category_bar_chart(
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
        else:
            axes[1, 1].text(0.5, 0.5, 'Datos de cilindros\ninsuficientes', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        self.plotter.save_and_show_plot('engine_efficiency_analysis.png')
    
    def analyze_vehicle_segments(self):
        print("\n=== ANÁLISIS: SEGMENTOS DE VEHÍCULOS PARA DECISIÓN DE COMPRA ===")
        
        segment_data = self.analyzer.get_vehicle_segments_stats()
        if not segment_data:
            print("Datos de clase de vehículo no disponibles")
            return
        
        stats, best_segments, segment_counts, price_ranges = segment_data
        
        fig, axes = self.plotter.setup_subplot_grid(
            2, 2, figsize=(15, 12),
            title='Categorías de Vehículos: Análisis para Decisión de Compra'
        )
        
        self.plotter.create_segment_scatter(
            axes[0, 0], self.data, 'VehicleClass', 'MSRP', 'MPG_Average',
            'Precio (USD)', 'Eficiencia (MPG)',
            'Posicionamiento de Categorías: Precio vs Eficiencia'
        )
        
        self.plotter.create_category_bar_chart(
            axes[0, 1], best_segments, 'MPG_Average',
            'Categorías', 'Eficiencia Promedio (MPG)',
            'Top 5 Categorías más Eficientes', color='green', add_labels=False
        )
        
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, 
                      autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        axes[1, 0].set_title('Distribución del Mercado por Categoría')
        
        bars = axes[1, 1].bar(range(len(price_ranges)), price_ranges['mean'], 
                             alpha=0.7, color='skyblue')
        axes[1, 1].set_xticks(range(len(price_ranges)))
        axes[1, 1].set_xticklabels(price_ranges.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Precio Promedio (USD)')
        axes[1, 1].set_title('Precio Promedio por Categoría')
        
        self.plotter.save_and_show_plot('vehicle_segments_analysis.png')
    
    def identify_best_purchase_recommendations(self):
        print("\n=== ANALISIS: RECOMENDACIONES DE COMPRA ===")
        
        recommendations, top_10_overall = self.analyzer.get_purchase_recommendations()
        
        fig, axes = self.plotter.setup_subplot_grid(
            1, 2, figsize=(14, 6),
            title='Recomendaciones de Compra: Mejores Vehículos por Categoría'
        )
        
        self.plotter.create_top_vehicles_chart(
            axes[0], top_10_overall, 'Simple_Value',
            'Top 10 Vehículos con Mejor Valor\n(Fórmula: MPG ÷ (Precio/10000))'
        )
        
        if recommendations:
            sorted_categories = sorted(recommendations.items(), 
                                     key=lambda x: x[1]['value_score'], reverse=True)
            categories = [item[0] for item in sorted_categories]
            values = [item[1]['value_score'] for item in sorted_categories]
            
            colors = ['lightblue', 'orange', 'lightcoral', 'lightgreen', 'lavender'][:len(categories)]
            bars = axes[1].bar(categories, values, alpha=0.7, color=colors)
            axes[1].set_ylabel('Puntuación de Valor')
            axes[1].set_title(f'Mejor Vehículo por Categoría de Precio')
            axes[1].tick_params(axis='x', rotation=45)
            
            for bar, cat in zip(bars, categories):
                vehicle_info = recommendations[cat]
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f"{vehicle_info['brand_model'][:20]}...\n${vehicle_info['price']:,.0f}\n{vehicle_info['efficiency']:.1f} MPG", 
                           ha='center', va='bottom', fontsize=8)
        else:
            axes[1].text(0.5, 0.5, 'No se encontraron\nrecomendaciones por categoría', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        
        self.plotter.save_and_show_plot('purchase_recommendations.png')
        
        return recommendations
    
    def run_focused_analysis(self):
        print("="*70)
        print("ANÁLISIS: PRECIO-RENDIMIENTO PARA DECISIÓN DE COMPRA")
        print("="*70)
        
        best_value_vehicles = self.analyze_price_performance_relationship()
        self.analyze_engine_efficiency_factors()
        self.analyze_vehicle_segments()
        recommendations = self.identify_best_purchase_recommendations()
        
        return {
            'best_value_vehicles': best_value_vehicles,
            'recommendations': recommendations,
            'insights': self.analyzer.get_all_insights()
        }
    
    def _initialize_reporter(self):
        if self.reporter is None:
            self.reporter = ReportGenerator(self.analyzer.get_all_insights(), self.data)


def main():
    print("Iniciando análisis de datos de vehículos...")
    print("="*60)

if __name__ == "__main__":
    main()
