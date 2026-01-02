"""
Professional visualization suite for crack spread strategy
Publication-quality charts with institutional aesthetics
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class Visualizer:
    """
    Comprehensive visualization toolkit for trading strategy analysis
    
    Features:
    - Price and spread charts
    - Equity curves with drawdown
    - Trade distribution analysis
    - Performance heatmaps
    - Z-score with signal overlays
    - Monte Carlo simulation results
    - Risk metrics visualization
    """
    
    def __init__(self, config):
        self.config = config
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'success': '#06A77D',      # Green
            'danger': '#D81E5B',       # Red
            'warning': '#F77F00',      # Orange
            'neutral': '#264653',      # Dark teal
            'accent': '#E9C46A',       # Gold
        }
    
    def plot_price_series(self, df: pd.DataFrame, 
                         save_path: Optional[str] = None):
        """
        Plot CL and HO price series with volume
        
        Args:
            df: DataFrame with OHLCV data
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Crude Oil Price
        axes[0].plot(df.index, df['CL_Close'], 
                    label='Crude Oil (CL)', 
                    color=self.colors['primary'], 
                    linewidth=1.5)
        axes[0].set_title('Crude Oil Futures (CL=F)', fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Heating Oil Price
        axes[1].plot(df.index, df['HO_Close'], 
                    label='Heating Oil (HO)', 
                    color=self.colors['secondary'], 
                    linewidth=1.5)
        axes[1].set_title('Heating Oil Futures (HO=F)', fontweight='bold')
        axes[1].set_ylabel('Price ($)', fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Volume comparison
        axes[2].bar(df.index, df['CL_Volume'], 
                   label='CL Volume', 
                   color=self.colors['primary'], 
                   alpha=0.6, width=1)
        axes[2].bar(df.index, df['HO_Volume'], 
                   label='HO Volume', 
                   color=self.colors['secondary'], 
                   alpha=0.6, width=1)
        axes[2].set_title('Trading Volume', fontweight='bold')
        axes[2].set_ylabel('Volume', fontweight='bold')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved price series plot to: {save_path}")
        
        plt.show()
    
    def plot_spread_analysis(self, signals: pd.DataFrame, 
                            save_path: Optional[str] = None):
        """
        Comprehensive spread analysis with z-score and signals
        
        Args:
            signals: DataFrame with spread, z-score, and trading signals
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # === Spread with Bollinger Bands ===
        axes[0].plot(signals.index, signals['Spread'], 
                    label='Log Price Spread (CL/HO)', 
                    color=self.colors['neutral'], 
                    linewidth=1.2)
        axes[0].plot(signals.index, signals['Rolling_Mean'], 
                    label='30-Day Mean', 
                    color=self.colors['danger'], 
                    linewidth=2, 
                    linestyle='--')
        
        # Bollinger Bands
        axes[0].fill_between(signals.index,
                            signals['Rolling_Mean'] + signals['Rolling_Std'],
                            signals['Rolling_Mean'] - signals['Rolling_Std'],
                            alpha=0.2, 
                            color=self.colors['accent'], 
                            label='±1 Std Dev')
        axes[0].fill_between(signals.index,
                            signals['Rolling_Mean'] + 2*signals['Rolling_Std'],
                            signals['Rolling_Mean'] - 2*signals['Rolling_Std'],
                            alpha=0.1, 
                            color=self.colors['accent'], 
                            label='±2 Std Dev')
        
        axes[0].set_title('CL-HO Crack Spread with Mean-Reversion Bands', 
                         fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Spread', fontweight='bold')
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        
        # === Z-Score with Thresholds ===
        axes[1].plot(signals.index, signals['Z_Score'], 
                    label='Z-Score', 
                    color=self.colors['primary'], 
                    linewidth=1.2)
        axes[1].axhline(2, color=self.colors['danger'], 
                       linestyle='--', alpha=0.7, linewidth=2, 
                       label='Short Threshold (+2σ)')
        axes[1].axhline(-2, color=self.colors['success'], 
                       linestyle='--', alpha=0.7, linewidth=2, 
                       label='Long Threshold (-2σ)')
        axes[1].axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=1)
        axes[1].fill_between(signals.index, -2, 2, alpha=0.1, color='gray')
        
        # Mark entry signals
        long_entries = signals[signals['Trade_Entry'] & (signals['Position'] > 0)]
        short_entries = signals[signals['Trade_Entry'] & (signals['Position'] < 0)]
        
        axes[1].scatter(long_entries.index, long_entries['Z_Score'], 
                       color=self.colors['success'], marker='^', s=100, 
                       label='Long Entry', zorder=5, edgecolors='black', linewidth=0.5)
        axes[1].scatter(short_entries.index, short_entries['Z_Score'], 
                       color=self.colors['danger'], marker='v', s=100, 
                       label='Short Entry', zorder=5, edgecolors='black', linewidth=0.5)
        
        axes[1].set_title('Z-Score with Trade Signals', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('Z-Score', fontweight='bold')
        axes[1].legend(loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        
        # === Position Tracking ===
        axes[2].fill_between(signals.index, 0, signals['Position'], 
                            where=signals['Position'] > 0,
                            color=self.colors['success'], alpha=0.5, 
                            label='Long Position')
        axes[2].fill_between(signals.index, 0, signals['Position'], 
                            where=signals['Position'] < 0,
                            color=self.colors['danger'], alpha=0.5, 
                            label='Short Position')
        axes[2].axhline(0, color='black', linestyle='-', linewidth=1)
        
        axes[2].set_title('Position Tracking', fontweight='bold', fontsize=14)
        axes[2].set_ylabel('Position', fontweight='bold')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].legend(loc='best', framealpha=0.9)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved spread analysis plot to: {save_path}")
        
        plt.show()
    
    def plot_equity_curve(self, results: pd.DataFrame, 
                         save_path: Optional[str] = None):
        """
        Plot equity curve with drawdown
        
        Args:
            results: Backtest results DataFrame
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # === Equity Curve ===
        axes[0].plot(results.index, results['Portfolio_Value'], 
                    label='Portfolio Value', 
                    color=self.colors['primary'], 
                    linewidth=2)
        axes[0].plot(results.index, results['Cumulative_Max'], 
                    label='Peak Value', 
                    color=self.colors['success'], 
                    linewidth=1.5, 
                    linestyle='--', 
                    alpha=0.7)
        
        # Initial capital line
        axes[0].axhline(self.config.initial_capital, 
                       color='gray', linestyle=':', 
                       label='Initial Capital', linewidth=1.5)
        
        # Fill between equity and peak
        axes[0].fill_between(results.index,
                            results['Portfolio_Value'],
                            results['Cumulative_Max'],
                            where=results['Portfolio_Value'] < results['Cumulative_Max'],
                            color=self.colors['danger'],
                            alpha=0.3,
                            label='Drawdown Period')
        
        axes[0].set_title('Portfolio Equity Curve', fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Portfolio Value ($)', fontweight='bold')
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # === Drawdown ===
        axes[1].fill_between(results.index, 0, results['Drawdown'] * 100,
                            color=self.colors['danger'], alpha=0.6)
        axes[1].set_title('Drawdown', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('Drawdown (%)', fontweight='bold')
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Annotate max drawdown
        max_dd_idx = results['Drawdown'].idxmin()
        max_dd_val = results['Drawdown'].min() * 100
        axes[1].annotate(f'Max DD: {max_dd_val:.2f}%',
                        xy=(max_dd_idx, max_dd_val),
                        xytext=(10, -30),
                        textcoords='offset points',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved equity curve plot to: {save_path}")
        
        plt.show()
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame, 
                               save_path: Optional[str] = None):
        """
        Analyze trade distribution and statistics
        
        Args:
            trades_df: DataFrame with individual trade records
            save_path: Optional path to save figure
        """
        if trades_df.empty:
            print("⚠️  No trades to plot")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # === 1. P&L Distribution ===
        ax1 = fig.add_subplot(gs[0, :2])
        wins = trades_df[trades_df['Win']]['Portfolio_PnL_Pct']
        losses = trades_df[~trades_df['Win']]['Portfolio_PnL_Pct']
        
        ax1.hist([wins, losses], bins=30, label=['Wins', 'Losses'],
                color=[self.colors['success'], self.colors['danger']],
                alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=2)
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.set_xlabel('P&L (%)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === 2. Win Rate Pie Chart ===
        ax2 = fig.add_subplot(gs[0, 2])
        win_rate = (trades_df['Win'].sum() / len(trades_df)) * 100
        sizes = [win_rate, 100 - win_rate]
        colors_pie = [self.colors['success'], self.colors['danger']]
        ax2.pie(sizes, labels=['Wins', 'Losses'], autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        ax2.set_title(f'Win Rate: {win_rate:.1f}%', fontweight='bold')
        
        # === 3. Duration Analysis ===
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(trades_df['Duration_Days'], bins=20, 
                color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax3.axvline(trades_df['Duration_Days'].mean(), 
                   color=self.colors['danger'], linestyle='--', linewidth=2,
                   label=f"Mean: {trades_df['Duration_Days'].mean():.1f} days")
        ax3.set_title('Trade Duration', fontweight='bold')
        ax3.set_xlabel('Days', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # === 4. Cumulative P&L ===
        ax4 = fig.add_subplot(gs[1, 1:])
        cumulative_pnl = trades_df['Portfolio_PnL'].cumsum()
        ax4.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl,
                color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
        ax4.axhline(0, color='black', linestyle='--', linewidth=1)
        ax4.fill_between(range(1, len(cumulative_pnl) + 1), 0, cumulative_pnl,
                        where=cumulative_pnl >= 0, color=self.colors['success'], alpha=0.3)
        ax4.fill_between(range(1, len(cumulative_pnl) + 1), 0, cumulative_pnl,
                        where=cumulative_pnl < 0, color=self.colors['danger'], alpha=0.3)
        ax4.set_title('Cumulative P&L by Trade', fontweight='bold')
        ax4.set_xlabel('Trade Number', fontweight='bold')
        ax4.set_ylabel('Cumulative P&L ($)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # === 5. MAE vs MFE Scatter ===
        ax5 = fig.add_subplot(gs[2, :2])
        colors_scatter = [self.colors['success'] if w else self.colors['danger'] 
                         for w in trades_df['Win']]
        ax5.scatter(trades_df['MAE'], trades_df['MFE'], 
                   c=colors_scatter, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax5.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_title('Maximum Adverse Excursion vs Maximum Favorable Excursion', 
                     fontweight='bold')
        ax5.set_xlabel('MAE (Max Adverse)', fontweight='bold')
        ax5.set_ylabel('MFE (Max Favorable)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # === 6. Summary Statistics Table ===
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_text = f"""
        TRADE STATISTICS
        ─────────────────────
        Total Trades: {len(trades_df)}
        Win Rate: {win_rate:.1f}%
        Avg Win: {wins.mean():.2f}%
        Avg Loss: {losses.mean():.2f}%
        Largest Win: {trades_df['Portfolio_PnL_Pct'].max():.2f}%
        Largest Loss: {trades_df['Portfolio_PnL_Pct'].min():.2f}%
        Avg Duration: {trades_df['Duration_Days'].mean():.1f} days
        Total P&L: ${trades_df['Portfolio_PnL'].sum():,.2f}
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved trade distribution plot to: {save_path}")
        
        plt.show()
    
    def plot_monthly_returns_heatmap(self, results: pd.DataFrame, 
                                    save_path: Optional[str] = None):
        """
        Monthly returns heatmap
        
        Args:
            results: Backtest results DataFrame
            save_path: Optional path to save figure
        """
        # Calculate monthly returns
        monthly_returns = results['Net_Return'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'}, 
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        ax.set_title('Monthly Returns Heatmap (%)', fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('Year', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved monthly returns heatmap to: {save_path}")
        
        plt.show()
    
    def plot_rolling_metrics(self, results: pd.DataFrame, 
                            save_path: Optional[str] = None):
        """
        Plot rolling performance metrics
        
        Args:
            results: Backtest results DataFrame
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Rolling Sharpe Ratio
        axes[0].plot(results.index, results['Rolling_Sharpe'],
                    color=self.colors['primary'], linewidth=1.5)
        axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[0].axhline(1, color=self.colors['success'], linestyle=':', linewidth=1, alpha=0.5)
        
        
        
        
        plt.fill_between(results.index, 0, results['Rolling_Sharpe'],
                        where=results['Rolling_Sharpe'] > 0,
                        color=self.colors['success'], alpha=0.3)
        axes[0].set_title('Rolling 60-Day Sharpe Ratio', fontweight='bold')
        axes[0].set_ylabel('Sharpe Ratio', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Rolling Volatility
        axes[1].plot(results.index, results['Rolling_Volatility'],
                    color=self.colors['warning'], linewidth=1.5)
        axes[1].axhline(results['Rolling_Volatility'].mean(),
                       color='black', linestyle='--', linewidth=1,
                       label=f"Mean: {results['Rolling_Volatility'].mean():.2f}%")
        axes[1].set_title('Rolling 60-Day Volatility', fontweight='bold')
        axes[1].set_ylabel('Volatility (%)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Rolling Win Rate
        axes[2].plot(results.index, results['Rolling_Win_Rate'],
                    color=self.colors['secondary'], linewidth=1.5)
        axes[2].axhline(50, color='black', linestyle='--', linewidth=1)
        axes[2].fill_between(results.index, 50, results['Rolling_Win_Rate'],
                            where=results['Rolling_Win_Rate'] > 50,
                            color=self.colors['success'], alpha=0.3)
        axes[2].fill_between(results.index, 50, results['Rolling_Win_Rate'],
                            where=results['Rolling_Win_Rate'] < 50,
                            color=self.colors['danger'], alpha=0.3)
        axes[2].set_title('Rolling 60-Day Win Rate', fontweight='bold')
        axes[2].set_ylabel('Win Rate (%)', fontweight='bold')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved rolling metrics plot to: {save_path}")
        
        plt.show()
    
    def plot_monte_carlo_results(self, mc_results: pd.DataFrame,
                                 save_path: Optional[str] = None):
        """
        Visualize Monte Carlo simulation results
        
        Args:
            mc_results: DataFrame with simulation results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribution of final returns
        axes[0, 0].hist(mc_results['Total_Return_Pct'], bins=50,
                       color=self.colors['primary'], alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(mc_results['Total_Return_Pct'].mean(),
                          color=self.colors['danger'], linestyle='--',
                          linewidth=2, label=f"Mean: {mc_results['Total_Return_Pct'].mean():.2f}%")
        axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=1)
        axes[0, 0].set_title('Distribution of Returns', fontweight='bold')
        axes[0, 0].set_xlabel('Total Return (%)', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution of final portfolio values
        axes[0, 1].hist(mc_results['Final_Value'], bins=50,
                       color=self.colors['success'], alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(mc_results['Final_Value'].mean(),
                          color=self.colors['danger'], linestyle='--',
                          linewidth=2, label=f"Mean: ${mc_results['Final_Value'].mean():,.0f}")
        axes[0, 1].set_title('Distribution of Final Portfolio Value', fontweight='bold')
        axes[0, 1].set_xlabel('Portfolio Value ($)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Box plot of returns
        axes[1, 0].boxplot([mc_results['Total_Return_Pct']], vert=False,
                          patch_artist=True,
                          boxprops=dict(facecolor=self.colors['accent'], alpha=0.7))
        axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_title('Return Distribution Summary', fontweight='bold')
        axes[1, 0].set_xlabel('Total Return (%)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence intervals
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = [np.percentile(mc_results['Total_Return_Pct'], p) for p in percentiles]
        
        axes[1, 1].barh(percentiles, percentile_values, 
                       color=[self.colors['danger'], self.colors['warning'], 
                              self.colors['primary'], self.colors['accent'], 
                              self.colors['success']],
                       alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].set_title('Percentile Returns', fontweight='bold')
        axes[1, 1].set_xlabel('Return (%)', fontweight='bold')
        axes[1, 1].set_ylabel('Percentile', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved Monte Carlo results plot to: {save_path}")
        
        plt.show()
    
    def plot_regime_analysis(self, signals: pd.DataFrame,
                            save_path: Optional[str] = None):
        """
        Visualize market regime classification
        
        Args:
            signals: DataFrame with regime classifications
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Spread with regime colors
        for regime, color in [('Mean_Reverting', self.colors['success']),
                             ('Trending_Up', self.colors['danger']),
                             ('Trending_Down', self.colors['warning']),
                             ('Volatile_Trending', self.colors['neutral'])]:
            mask = signals['Regime_Composite'] == regime
            if mask.any():
                axes[0].scatter(signals[mask].index, signals[mask]['Spread'],
                              c=color, label=regime, alpha=0.5, s=10)
        
        axes[0].plot(signals.index, signals['Rolling_Mean'],
                    color='black', linewidth=2, label='Mean')
        axes[0].set_title('Market Regime Classification', fontweight='bold')
        axes[0].set_ylabel('Spread', fontweight='bold')
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        
        # Regime transitions
        regime_numeric = pd.Categorical(signals['Regime_Composite']).codes
        axes[1].plot(signals.index, regime_numeric, linewidth=1.5,
                    color=self.colors['primary'])
        axes[1].fill_between(signals.index, 0, regime_numeric,
                            alpha=0.3, color=self.colors['primary'])
        axes[1].set_title('Regime Transitions Over Time', fontweight='bold')
        axes[1].set_ylabel('Regime Code', fontweight='bold')
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved regime analysis plot to: {save_path}")
        
        plt.show()
    
    def create_full_report(self, df: pd.DataFrame, 
                          signals: pd.DataFrame,
                          results: pd.DataFrame,
                          trades_df: pd.DataFrame):
        """
        Generate complete visual report
        
        Args:
            df: Raw market data
            signals: Strategy signals
            results: Backtest results
            trades_df: Individual trades
        """
        print("\n" + "=" * 60)
        print("📊 GENERATING COMPREHENSIVE VISUAL REPORT")
        print("=" * 60)
        
        output_dir = self.config.results_dir
        
        # 1. Price series
        print("\n1️⃣  Generating price series plots...")
        self.plot_price_series(df, save_path=f"{output_dir}/01_price_series.{self.config.plot_format}")
        
        # 2. Spread analysis
        print("2️⃣  Generating spread analysis plots...")
        self.plot_spread_analysis(signals, save_path=f"{output_dir}/02_spread_analysis.{self.config.plot_format}")
        
        # 3. Equity curve
        print("3️⃣  Generating equity curve...")
        self.plot_equity_curve(results, save_path=f"{output_dir}/03_equity_curve.{self.config.plot_format}")
        
        # 4. Trade distribution
        if not trades_df.empty:
            print("4️⃣  Generating trade distribution analysis...")
            self.plot_trade_distribution(trades_df, save_path=f"{output_dir}/04_trade_distribution.{self.config.plot_format}")
        
        # 5. Monthly returns heatmap
        print("5️⃣  Generating monthly returns heatmap...")
        self.plot_monthly_returns_heatmap(results, save_path=f"{output_dir}/05_monthly_heatmap.{self.config.plot_format}")
        
        # 6. Rolling metrics
        print("6️⃣  Generating rolling metrics...")
        self.plot_rolling_metrics(results, save_path=f"{output_dir}/06_rolling_metrics.{self.config.plot_format}")
        
        # 7. Regime analysis
        if 'Regime_Composite' in signals.columns:
            print("7️⃣  Generating regime analysis...")
            self.plot_regime_analysis(signals, save_path=f"{output_dir}/07_regime_analysis.{self.config.plot_format}")
        
        print("\n✅ Complete visual report generated!")
        print(f"   All plots saved to: {output_dir}/")
        print("=" * 60)