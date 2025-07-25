import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class PlotHelper:
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    @staticmethod
    def setup_subplot_grid(rows, cols, figsize=(15, 12), title=""):
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        return fig, axes
    
    @staticmethod
    def add_correlation_text(ax, correlation, position=(0.05, 0.95)):
        ax.text(position[0], position[1], f'Correlación: {correlation:.3f}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    @staticmethod
    def create_scatter_with_trendline(ax, x_data, y_data, xlabel, ylabel, title, color='steelblue'):
        ax.scatter(x_data, y_data, alpha=0.6, s=50, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        correlation = x_data.corr(y_data)
        z = np.polyfit(x_data.dropna(), y_data[x_data.notna()], 1)
        p = np.poly1d(z)
        ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
        
        PlotHelper.add_correlation_text(ax, correlation)
        return correlation
    
    @staticmethod
    def create_histogram_with_mean(ax, data, xlabel, ylabel, title, bins=30, color='lightgreen'):
        ax.hist(data, bins=bins, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', 
                  label=f'Promedio: {mean_val:.1f}')
        ax.legend()
        return mean_val
    
    @staticmethod
    def create_category_bar_chart(ax, stats_df, y_col, xlabel, ylabel, title, 
                                 color='skyblue', add_labels=True):
        bars = ax.bar(range(len(stats_df)), stats_df[y_col], alpha=0.7, color=color)
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels(stats_df.index, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if add_labels and 'Count' in stats_df.columns and 'Price_Mean' in stats_df.columns:
            for i, (bar, count, price) in enumerate(zip(bars, stats_df['Count'], stats_df['Price_Mean'])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'n={int(count)}\n${price:,.0f}', ha='center', va='bottom', fontsize=8)
        
        return bars
    
    @staticmethod
    def create_top_vehicles_chart(ax, top_vehicles_df, value_col, title):
        bars = ax.barh(range(len(top_vehicles_df)), top_vehicles_df[value_col], 
                      alpha=0.7, color='green')
        ax.set_yticks(range(len(top_vehicles_df)))
        
        labels = []
        for _, row in top_vehicles_df.iterrows():
            model = row['Model'][:15] + "..." if len(row['Model']) > 15 else row['Model']
            labels.append(f"{row['Brand']} {model}")
        
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Puntuación de Valor (MPG por $10k)')
        ax.set_title(title)
        ax.invert_yaxis()
        
        for bar, value in zip(bars, top_vehicles_df[value_col]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}', ha='left', va='center', fontsize=8)
        
        return bars
    
    @staticmethod
    def save_and_show_plot(filename, show=True):
        plt.tight_layout()
        plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def create_segment_scatter(ax, data, segment_col, x_col, y_col, xlabel, ylabel, title):
        segments = data[segment_col].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        segment_colors = dict(zip(segments, colors))
        
        for segment in segments:
            segment_data = data[data[segment_col] == segment]
            ax.scatter(segment_data[x_col], segment_data[y_col], 
                     label=segment, alpha=0.7, s=50, color=segment_colors[segment])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def create_feature_importance_chart(ax, feature_importance_df, title, top_n=8):
        top_features = feature_importance_df.head(top_n)
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      alpha=0.7, color='lightcoral')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importancia')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    @staticmethod
    def create_ml_prediction_scatter(ax, actual, predicted, title, xlabel="Valores Reales", ylabel="Predicciones"):
        ax.scatter(actual, predicted, alpha=0.6, s=50, color='steelblue')
        
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        r2 = np.corrcoef(actual, predicted)[0, 1] ** 2
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    @staticmethod
    def create_ml_value_comparison(ax, data, actual_col, ml_col, title):
        x_pos = range(len(data))
        width = 0.35
        
        ax.bar([x - width/2 for x in x_pos], data[actual_col], width, 
               label='Valor Actual', alpha=0.7, color='lightblue')
        ax.bar([x + width/2 for x in x_pos], data[ml_col], width,
               label='Valor Predicho ML', alpha=0.7, color='lightcoral')
        
        ax.set_ylabel('Puntuación de Valor')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{row['Brand']} {row['Model']}"[:15] + "..." 
                           for _, row in data.iterrows()], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def create_ml_insights_overview(ax, model_performance, title):
        metrics = []
        values = []
        
        if 'r2_score' in model_performance:
            metrics.append('R² Score')
            values.append(model_performance['r2_score'])
        if 'accuracy' in model_performance:
            metrics.append('Accuracy')
            values.append(model_performance['accuracy'])
        if 'cv_mean' in model_performance:
            metrics.append('CV Mean')
            values.append(model_performance['cv_mean'])
            
        if metrics:
            bars = ax.bar(metrics, values, alpha=0.7, color=['skyblue', 'lightgreen', 'coral'][:len(metrics)])
            ax.set_ylabel('Puntuación')
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
