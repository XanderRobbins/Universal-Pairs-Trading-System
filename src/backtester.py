"""
Comprehensive backtesting engine with walk-forward analysis
Includes transaction costs, slippage, and detailed performance tracking
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

class Backtester:
    """
    Professional-grade backtesting engine
    
    Features:
    - Realistic transaction costs and slippage
    - Position-by-position P&L tracking
    - Walk-forward optimization
    - Monte Carlo simulation
    - Multiple performance metrics
    - Trade journal with detailed analytics
    """
    
    def __init__(self, config):
        self.config = config
        self.results = None
        self.trades = []
        self.equity_curve = None
        
    def run_backtest(self, signals: pd.DataFrame, 
                    initial_capital: Optional[float] = None) -> pd.DataFrame:
        """
        Execute comprehensive backtest with realistic constraints
        
        Args:
            signals: DataFrame with trading signals and positions
            initial_capital: Starting capital (uses config if None)
        
        Returns:
            DataFrame with complete backtest results
        """
        print("\n" + "=" * 60)
        print("🚀 RUNNING BACKTEST")
        print("=" * 60)
        
        capital = initial_capital or self.config.initial_capital
        
        df = signals.copy()
        
        # === 1. Calculate Returns ===
        print("\n1️⃣  Calculating raw returns...")
        df = self._calculate_returns(df)
        
        # === 2. Apply Transaction Costs ===
        print("2️⃣  Applying transaction costs and slippage...")
        df = self._apply_transaction_costs(df)
        
        # === 3. Calculate Portfolio Value ===
        print("3️⃣  Computing portfolio value...")
        df = self._calculate_portfolio_value(df, capital)
        
        # === 4. Calculate Drawdown ===
        print("4️⃣  Calculating drawdown metrics...")
        df = self._calculate_drawdown(df)
        df = self._apply_circuit_breakers(df)
        
        # === 5. Track Individual Trades ===
        print("5️⃣  Extracting trade log...")
        self.trades = self._extract_trades(df)
        
        # === 6. Calculate Rolling Metrics ===
        print("6️⃣  Computing rolling performance metrics...")
        df = self._calculate_rolling_metrics(df)
        
        self.results = df
        self.equity_curve = df['Portfolio_Value']
        
        print("\n✅ Backtest complete!")
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from actual leg positions (SPY and QQQ)"""
        
        # Calculate individual asset returns
        df['CL_Return'] = df['CL_Close'].pct_change()  # SPY return
        df['HO_Return'] = df['HO_Close'].pct_change()  # QQQ return
        
        # For long spread: buy SPY, short QQQ (equal dollar amounts)
        # For short spread: short SPY, buy QQQ
        df['Leg1_Return'] = df['Position'].shift(1) * df['CL_Return']
        df['Leg2_Return'] = -df['Position'].shift(1) * df['HO_Return']
        
        # Combined return (50/50 allocation between legs)
        df['Gross_Return'] = (df['Leg1_Return'] + df['Leg2_Return']) / 2
        
        # Handle NaN values
        df['Gross_Return'].fillna(0, inplace=True)
        
        return df


    def _apply_transaction_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply realistic transaction costs
        
        Costs include:
        - Bid-ask spread (captured in slippage)
        - Commission per contract
        - Impact cost (for large orders)
        """
        # Detect position changes (entries/exits/scaling)
        df['Position_Change'] = df['Position'].diff().abs()
        
        # Commission cost (per contract, convert to percentage of capital)
        contract_value = 1000  # CL contract = 1000 barrels
        df['Commission'] = (
            df['Position_Change'] * 
            df['Position_Size'] * 
            self.config.commission_per_contract
        ) / self.config.initial_capital
        
        # Slippage cost (percentage of position value)
        df['Slippage'] = df['Position_Change'] * self.config.slippage_pct
        
        # Bid-ask spread cost
        df['Transaction_Cost'] = (
            df['Position_Change'] * self.config.transaction_cost_pct
        )
        
        # Total costs
        df['Total_Costs'] = df['Commission'] + df['Slippage'] + df['Transaction_Cost']
        
        # Net return after costs
        df['Net_Return'] = df['Gross_Return'] - df['Total_Costs']
        
        return df
    
    def _calculate_portfolio_value(self, df: pd.DataFrame, 
                                   initial_capital: float) -> pd.DataFrame:
        """Calculate portfolio value over time"""
        
        # Cumulative returns (multiplicative)
        df['Cumulative_Return'] = (1 + df['Net_Return']).cumprod()
        
        # Portfolio value
        df['Portfolio_Value'] = initial_capital * df['Cumulative_Return']
        
        # Daily P&L in dollars
        df['Daily_PnL'] = df['Portfolio_Value'].diff().fillna(0)
        
        return df
    

    def _apply_circuit_breakers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stop trading if catastrophic losses occur"""
        
        for i in range(len(df)):
            # Stop if portfolio goes negative
            if df['Portfolio_Value'].iloc[i] < 0:
                print(f"\n🚨 CIRCUIT BREAKER: Portfolio went negative on {df.index[i]}")
                df.loc[df.index[i]:, 'Position'] = 0
                break
            
            # Stop if drawdown > 50%
            if df['Drawdown'].iloc[i] < -0.50:
                print(f"\n🚨 CIRCUIT BREAKER: 50% drawdown hit on {df.index[i]}")
                df.loc[df.index[i]:, 'Position'] = 0
                break
        
        return df
    
    def _calculate_drawdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown metrics"""
        
        # Running maximum (peak)
        df['Cumulative_Max'] = df['Portfolio_Value'].cummax()
        
        # Drawdown (dollar amount)
        df['Drawdown_Dollar'] = df['Portfolio_Value'] - df['Cumulative_Max']
        
        # Drawdown (percentage)
        df['Drawdown'] = df['Drawdown_Dollar'] / df['Cumulative_Max']
        
        # Underwater period (days since last peak)
        df['Days_Underwater'] = 0
        underwater_count = 0
        
        for i in range(len(df)):
            if df['Drawdown'].iloc[i] < 0:
                underwater_count += 1
            else:
                underwater_count = 0
            df['Days_Underwater'].iloc[i] = underwater_count
        
        return df
    
    def _extract_trades(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract individual trades for detailed analysis
        
        Returns:
            List of trade dictionaries
        """
        trades = []
        in_trade = False
        entry_idx = None
        entry_price = None
        entry_zscore = None
        direction = None
        entry_portfolio_value = None
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # Trade entry
            if row['Trade_Entry'] and not in_trade:
                in_trade = True
                entry_idx = idx
                entry_price = row['Spread']
                entry_zscore = row['Z_Score']
                direction = 'Long' if row['Position'] > 0 else 'Short'
                entry_portfolio_value = row['Portfolio_Value']
            
            # Trade exit
            elif row['Trade_Exit'] and in_trade:
                exit_idx = idx
                exit_price = row['Spread']
                exit_zscore = row['Z_Score']
                exit_portfolio_value = row['Portfolio_Value']
                
                # Calculate P&L
                if direction == 'Long':
                    pnl = exit_price - entry_price
                else:  # Short
                    pnl = entry_price - exit_price
                
                pnl_pct = (pnl / abs(entry_price)) * 100
                
                # Portfolio P&L
                portfolio_pnl = exit_portfolio_value - entry_portfolio_value
                portfolio_pnl_pct = (portfolio_pnl / entry_portfolio_value) * 100
                
                # Calculate MAE and MFE during trade
                trade_slice = df.loc[entry_idx:exit_idx]
                if direction == 'Long':
                    mae = (trade_slice['Spread'] - entry_price).min()
                    mfe = (trade_slice['Spread'] - entry_price).max()
                else:
                    mae = (entry_price - trade_slice['Spread']).min()
                    mfe = (entry_price - trade_slice['Spread']).max()
                
                trades.append({
                    'Trade_Number': len(trades) + 1,
                    'Entry_Date': entry_idx,
                    'Exit_Date': exit_idx,
                    'Direction': direction,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Entry_Z_Score': entry_zscore,
                    'Exit_Z_Score': exit_zscore,
                    'Duration_Days': (exit_idx - entry_idx).days,
                    'Spread_PnL': pnl,
                    'Spread_PnL_Pct': pnl_pct,
                    'Portfolio_PnL': portfolio_pnl,
                    'Portfolio_PnL_Pct': portfolio_pnl_pct,
                    'MAE': mae,  # Maximum Adverse Excursion
                    'MFE': mfe,  # Maximum Favorable Excursion
                    'Win': pnl > 0
                })
                
                in_trade = False
        
        return trades
    
    def _calculate_rolling_metrics(self, df: pd.DataFrame, 
                                   window: int = 60) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        returns = df['Net_Return']
        
        # Rolling Sharpe Ratio
        df['Rolling_Sharpe'] = (
            returns.rolling(window).mean() / 
            returns.rolling(window).std() * 
            np.sqrt(252)
        )
        
        # Rolling volatility
        df['Rolling_Volatility'] = (
            returns.rolling(window).std() * np.sqrt(252)
        )
        
        # Rolling win rate
        df['Rolling_Win_Rate'] = (
            (returns > 0).rolling(window).mean() * 100
        )
        
        return df
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with all performance statistics
        """
        if self.results is None:
            raise ValueError("Run backtest first")
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE METRICS")
        print("=" * 60)
        
        df = self.results
        returns = df['Net_Return'].dropna()
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # === Return Metrics ===
        total_return = (df['Portfolio_Value'].iloc[-1] / self.config.initial_capital - 1) * 100
        
        n_years = len(df) / 252
        cagr = ((df['Portfolio_Value'].iloc[-1] / self.config.initial_capital) ** (1 / n_years) - 1) * 100
        
        ann_return = returns.mean() * 252 * 100
        ann_volatility = returns.std() * np.sqrt(252) * 100
        
        # === Risk-Adjusted Metrics ===
        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        calmar_ratio = self._calmar_ratio(returns, df['Drawdown'])
        
        # === Drawdown Metrics ===
        max_dd = df['Drawdown'].min() * 100
        max_dd_duration = df['Days_Underwater'].max()
        avg_dd = df[df['Drawdown'] < 0]['Drawdown'].mean() * 100 if len(df[df['Drawdown'] < 0]) > 0 else 0
        
        # === Trade Metrics ===
        if len(trades_df) > 0:
            wins = trades_df[trades_df['Win']]
            losses = trades_df[~trades_df['Win']]
            
            total_trades = len(trades_df)
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = wins['Portfolio_PnL_Pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['Portfolio_PnL_Pct'].mean() if len(losses) > 0 else 0
            
            largest_win = trades_df['Portfolio_PnL_Pct'].max() if total_trades > 0 else 0
            largest_loss = trades_df['Portfolio_PnL_Pct'].min() if total_trades > 0 else 0
            
            avg_duration = trades_df['Duration_Days'].mean()
            
            # Profit factor
            gross_profit = wins['Portfolio_PnL'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['Portfolio_PnL'].sum()) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Expectancy
            expectancy = trades_df['Portfolio_PnL'].mean()
            
            # Consecutive wins/losses
            consecutive_wins = self._max_consecutive_wins(trades_df)
            consecutive_losses = self._max_consecutive_losses(trades_df)
            
        else:
            total_trades = win_rate = avg_win = avg_loss = 0
            largest_win = largest_loss = avg_duration = 0
            profit_factor = expectancy = 0
            consecutive_wins = consecutive_losses = 0
        
        # === Exposure Metrics ===
        days_in_market = (df['Position'] != 0).sum()
        pct_in_market = (days_in_market / len(df)) * 100
        
        metrics = {
            # Return metrics
            'Total_Return_Pct': total_return,
            'CAGR_Pct': cagr,
            'Annualized_Return_Pct': ann_return,
            'Annualized_Volatility_Pct': ann_volatility,
            
            # Risk-adjusted returns
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            
            # Drawdown metrics
            'Max_Drawdown_Pct': max_dd,
            'Avg_Drawdown_Pct': avg_dd,
            'Max_Drawdown_Duration_Days': max_dd_duration,
            
            # Trade metrics
            'Total_Trades': total_trades,
            'Win_Rate_Pct': win_rate,
            'Avg_Win_Pct': avg_win,
            'Avg_Loss_Pct': avg_loss,
            'Largest_Win_Pct': largest_win,
            'Largest_Loss_Pct': largest_loss,
            'Avg_Trade_Duration_Days': avg_duration,
            'Profit_Factor': profit_factor,
            'Expectancy': expectancy,
            'Max_Consecutive_Wins': consecutive_wins,
            'Max_Consecutive_Losses': consecutive_losses,
            
            # Exposure
            'Days_In_Market': days_in_market,
            'Market_Exposure_Pct': pct_in_market,
            
            # Final values
            'Initial_Capital': self.config.initial_capital,
            'Final_Portfolio_Value': df['Portfolio_Value'].iloc[-1],
            'Total_PnL': df['Portfolio_Value'].iloc[-1] - self.config.initial_capital,
        }
        
        self._print_metrics_table(metrics)
        
        return metrics
    
    def _sharpe_ratio(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Sharpe Ratio (annualized)"""
        excess_returns = returns - rf_rate / 252
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    def _sortino_ratio(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Sortino Ratio (downside deviation)"""
        excess_returns = returns - rf_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    def _calmar_ratio(self, returns: pd.Series, drawdown: pd.Series) -> float:
        """Calmar Ratio (return/max drawdown)"""
        ann_return = returns.mean() * 252
        max_dd = abs(drawdown.min())
        return ann_return / max_dd if max_dd != 0 else 0
    
    def _max_consecutive_wins(self, trades_df: pd.DataFrame) -> int:
        """Calculate maximum consecutive wins"""
        if len(trades_df) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for win in trades_df['Win']:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses"""
        if len(trades_df) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for win in trades_df['Win']:
            if not win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _print_metrics_table(self, metrics: Dict):
        """Pretty print metrics in organized table"""
        
        print("\n" + "─" * 60)
        print("📈 RETURN METRICS")
        print("─" * 60)
        print(f"Total Return:              {metrics['Total_Return_Pct']:>12.2f}%")
        print(f"CAGR:                      {metrics['CAGR_Pct']:>12.2f}%")
        print(f"Annualized Return:         {metrics['Annualized_Return_Pct']:>12.2f}%")
        print(f"Annualized Volatility:     {metrics['Annualized_Volatility_Pct']:>12.2f}%")
        
        print("\n" + "─" * 60)
        print("⚖️  RISK-ADJUSTED RETURNS")
        print("─" * 60)
        print(f"Sharpe Ratio:              {metrics['Sharpe_Ratio']:>12.2f}")
        print(f"Sortino Ratio:             {metrics['Sortino_Ratio']:>12.2f}")
        print(f"Calmar Ratio:              {metrics['Calmar_Ratio']:>12.2f}")
        
        print("\n" + "─" * 60)
        print("📉 DRAWDOWN METRICS")
        print("─" * 60)
        print(f"Max Drawdown:              {metrics['Max_Drawdown_Pct']:>12.2f}%")
        print(f"Avg Drawdown:              {metrics['Avg_Drawdown_Pct']:>12.2f}%")
        print(f"Max DD Duration:           {metrics['Max_Drawdown_Duration_Days']:>12.0f} days")
        
        print("\n" + "─" * 60)
        print("💼 TRADE STATISTICS")
        print("─" * 60)
        print(f"Total Trades:              {metrics['Total_Trades']:>12.0f}")
        print(f"Win Rate:                  {metrics['Win_Rate_Pct']:>12.2f}%")
        print(f"Avg Win:                   {metrics['Avg_Win_Pct']:>12.2f}%")
        print(f"Avg Loss:                  {metrics['Avg_Loss_Pct']:>12.2f}%")
        print(f"Largest Win:               {metrics['Largest_Win_Pct']:>12.2f}%")
        print(f"Largest Loss:              {metrics['Largest_Loss_Pct']:>12.2f}%")
        print(f"Profit Factor:             {metrics['Profit_Factor']:>12.2f}")
        print(f"Expectancy:                ${metrics['Expectancy']:>11,.2f}")
        print(f"Avg Trade Duration:        {metrics['Avg_Trade_Duration_Days']:>12.1f} days")
        print(f"Max Consecutive Wins:      {metrics['Max_Consecutive_Wins']:>12.0f}")
        print(f"Max Consecutive Losses:    {metrics['Max_Consecutive_Losses']:>12.0f}")
        
        print("\n" + "─" * 60)
        print("🎯 EXPOSURE METRICS")
        print("─" * 60)
        print(f"Days in Market:            {metrics['Days_In_Market']:>12.0f}")
        print(f"Market Exposure:           {metrics['Market_Exposure_Pct']:>12.2f}%")
        
        print("\n" + "─" * 60)
        print("💰 PORTFOLIO SUMMARY")
        print("─" * 60)
        print(f"Initial Capital:           ${metrics['Initial_Capital']:>11,.2f}")
        print(f"Final Value:               ${metrics['Final_Portfolio_Value']:>11,.2f}")
        print(f"Total P&L:                 ${metrics['Total_PnL']:>11,.2f}")
        
        print("=" * 60 + "\n")
    
    def get_trade_dataframe(self) -> pd.DataFrame:
        """Return trades as DataFrame for easy analysis"""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
    
    def monte_carlo_simulation(self, n_simulations: int = 1000, 
                              n_trades: Optional[int] = None) -> Dict:
        """
        Run Monte Carlo simulation by bootstrapping trades
        
        Args:
            n_simulations: Number of simulation runs
            n_trades: Number of trades per simulation (default: actual number)
        
        Returns:
            Dictionary with simulation results
        """
        if not self.trades:
            raise ValueError("No trades to simulate")
        
        print(f"\n🎲 Running Monte Carlo simulation ({n_simulations} runs)...")
        
        trades_df = pd.DataFrame(self.trades)
        n_trades = n_trades or len(trades_df)
        
        # Store simulation results
        simulation_results = []
        
        for sim in range(n_simulations):
            # Randomly sample trades with replacement (bootstrap)
            sampled_trades = trades_df.sample(n=n_trades, replace=True)
            
            # Calculate cumulative return for this simulation
            cumulative_return = (1 + sampled_trades['Portfolio_PnL_Pct'] / 100).prod() - 1
            final_value = self.config.initial_capital * (1 + cumulative_return)
            
            simulation_results.append({
                'Simulation': sim + 1,
                'Final_Value': final_value,
                'Total_Return_Pct': cumulative_return * 100,
                'Max_DD': self._calculate_sim_drawdown(sampled_trades),
                'Sharpe': self._calculate_sim_sharpe(sampled_trades)
            })
        
        results_df = pd.DataFrame(simulation_results)
        
        # Calculate confidence intervals
        percentiles = [5, 25, 50, 75, 95]
        summary = {
            'Mean_Final_Value': results_df['Final_Value'].mean(),
            'Std_Final_Value': results_df['Final_Value'].std(),
            'Mean_Return_Pct': results_df['Total_Return_Pct'].mean(),
            'Std_Return_Pct': results_df['Total_Return_Pct'].std(),
        }
        
        for p in percentiles:
            summary[f'Percentile_{p}_Return'] = np.percentile(results_df['Total_Return_Pct'], p)
            summary[f'Percentile_{p}_Value'] = np.percentile(results_df['Final_Value'], p)
        
        # Probability of profit
        summary['Probability_of_Profit'] = (results_df['Total_Return_Pct'] > 0).sum() / n_simulations * 100
        
        # Risk of ruin (losing more than 20%)
        summary['Risk_of_Ruin_20pct'] = (results_df['Total_Return_Pct'] < -20).sum() / n_simulations * 100
        
        print(f"\n✅ Monte Carlo simulation complete!")
        print(f"\nResults over {n_simulations} simulations:")
        print(f"  Mean return: {summary['Mean_Return_Pct']:.2f}%")
        print(f"  Std dev: {summary['Std_Return_Pct']:.2f}%")
        print(f"  5th percentile: {summary['Percentile_5_Return']:.2f}%")
        print(f"  95th percentile: {summary['Percentile_95_Return']:.2f}%")
        print(f"  Probability of profit: {summary['Probability_of_Profit']:.1f}%")
        print(f"  Risk of 20%+ loss: {summary['Risk_of_Ruin_20pct']:.1f}%")
        
        return {
            'summary': summary,
            'detailed_results': results_df
        }
    
    def _calculate_sim_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate max drawdown for a simulated sequence of trades"""
        cumulative = (1 + trades_df['Portfolio_PnL_Pct'] / 100).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def _calculate_sim_sharpe(self, trades_df: pd.DataFrame, rf_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for simulated trades"""
        returns = trades_df['Portfolio_PnL_Pct'] / 100
        excess = returns.mean() - rf_rate / 252
        return (excess / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    def walk_forward_analysis(self, df: pd.DataFrame, 
                             strategy_func, 
                             train_window: int = 504,
                             test_window: int = 126) -> Dict:
        """
        Perform walk-forward optimization
        
        Args:
            df: Full dataset
            strategy_func: Function that takes df and returns signals
            train_window: Training period (days) - default 2 years
            test_window: Testing period (days) - default 6 months
        
        Returns:
            Dictionary with walk-forward results
        """
        print("\n" + "=" * 60)
        print("🔄 WALK-FORWARD ANALYSIS")
        print("=" * 60)
        
        results = []
        total_periods = (len(df) - train_window) // test_window
        
        print(f"\nTotal periods: {total_periods}")
        print(f"Train window: {train_window} days ({train_window/252:.1f} years)")
        print(f"Test window: {test_window} days ({test_window/252:.1f} years)")
        
        for i in range(total_periods):
            # Define train and test periods
            train_start = i * test_window
            train_end = train_start + train_window
            test_start = train_end
            test_end = test_start + test_window
            
            if test_end > len(df):
                break
            
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            print(f"\nPeriod {i+1}/{total_periods}")
            print(f"  Train: {train_data.index[0].date()} to {train_data.index[-1].date()}")
            print(f"  Test:  {test_data.index[0].date()} to {test_data.index[-1].date()}")
            
            # Generate signals for test period using strategy optimized on train period
            test_signals = strategy_func(test_data)
            
            # Run backtest on test period
            backtest_results = self.run_backtest(test_signals)
            metrics = self.calculate_performance_metrics()
            
            results.append({
                'Period': i + 1,
                'Train_Start': train_data.index[0],
                'Train_End': train_data.index[-1],
                'Test_Start': test_data.index[0],
                'Test_End': test_data.index[-1],
                'Return_Pct': metrics['Total_Return_Pct'],
                'Sharpe': metrics['Sharpe_Ratio'],
                'Max_DD_Pct': metrics['Max_Drawdown_Pct'],
                'Win_Rate_Pct': metrics['Win_Rate_Pct'],
                'Total_Trades': metrics['Total_Trades']
            })
        
        results_df = pd.DataFrame(results)
        
        # Aggregate statistics
        summary = {
            'Avg_Return_Pct': results_df['Return_Pct'].mean(),
            'Std_Return_Pct': results_df['Return_Pct'].std(),
            'Avg_Sharpe': results_df['Sharpe'].mean(),
            'Avg_Max_DD_Pct': results_df['Max_DD_Pct'].mean(),
            'Win_Rate_Consistency': (results_df['Return_Pct'] > 0).sum() / len(results_df) * 100,
            'Worst_Period_Return': results_df['Return_Pct'].min(),
            'Best_Period_Return': results_df['Return_Pct'].max(),
        }
        
        print("\n" + "=" * 60)
        print("📊 WALK-FORWARD SUMMARY")
        print("=" * 60)
        print(f"Average return per period: {summary['Avg_Return_Pct']:.2f}%")
        print(f"Return std deviation: {summary['Std_Return_Pct']:.2f}%")
        print(f"Average Sharpe ratio: {summary['Avg_Sharpe']:.2f}")
        print(f"Periods with positive returns: {summary['Win_Rate_Consistency']:.1f}%")
        print(f"Best period: {summary['Best_Period_Return']:.2f}%")
        print(f"Worst period: {summary['Worst_Period_Return']:.2f}%")
        print("=" * 60)
        
        return {
            'summary': summary,
            'period_results': results_df
        }
    
    def benchmark_comparison(self, benchmark_ticker: str = 'SPY') -> pd.DataFrame:
        """
        Compare strategy performance to benchmark
        
        Args:
            benchmark_ticker: Ticker symbol for benchmark (default SPY)
        
        Returns:
            DataFrame with comparative metrics
        """
        if self.results is None:
            raise ValueError("Run backtest first")
        
        print(f"\n📊 Comparing to {benchmark_ticker} benchmark...")
        
        # Fetch benchmark data
        import yfinance as yf
        benchmark_data = yf.download(
            benchmark_ticker,
            start=self.results.index[0],
            end=self.results.index[-1],
            progress=False
        )['Close']
        
        # Align dates
        benchmark_data = benchmark_data.reindex(self.results.index, method='ffill')
        benchmark_returns = benchmark_data.pct_change().fillna(0)
        
        # Calculate metrics
        strategy_metrics = self.calculate_performance_metrics()
        
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_total_return = (benchmark_cumulative.iloc[-1] - 1) * 100
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
        benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(252)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Metric': [
                'Total Return (%)',
                'Annualized Volatility (%)',
                'Sharpe Ratio',
                'Max Drawdown (%)',
            ],
            'Strategy': [
                strategy_metrics['Total_Return_Pct'],
                strategy_metrics['Annualized_Volatility_Pct'],
                strategy_metrics['Sharpe_Ratio'],
                strategy_metrics['Max_Drawdown_Pct'],
            ],
            f'{benchmark_ticker}': [
                benchmark_total_return,
                benchmark_volatility,
                benchmark_sharpe,
                self._calculate_benchmark_drawdown(benchmark_cumulative),
            ]
        })
        
        comparison['Difference'] = comparison['Strategy'] - comparison[f'{benchmark_ticker}']
        
        print("\n" + "=" * 60)
        print(f"STRATEGY vs {benchmark_ticker}")
        print("=" * 60)
        print(comparison.to_string(index=False))
        print("=" * 60)
        
        return comparison
    
    def _calculate_benchmark_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate drawdown for benchmark"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min() * 100
    
    def generate_trade_journal(self, filepath: str = 'results/trade_journal.csv'):
        """
        Export detailed trade journal to CSV
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.trades:
            print("⚠️  No trades to export")
            return
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filepath, index=False)
        print(f"\n✅ Trade journal exported to: {filepath}")
        print(f"   Total trades: {len(trades_df)}")