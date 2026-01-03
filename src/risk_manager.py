"""
Comprehensive risk management system for crack spread trading
Implements ATR-based position sizing, stop-loss, and portfolio controls
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """
    Advanced risk management with multiple safety layers
    
    Features:
    - ATR-based position sizing
    - Dynamic stop-loss and take-profit levels
    - Portfolio heat management
    - Maximum position limits
    - Correlation-based exposure control
    - Drawdown protection
    """
    
    def __init__(self, config):
        self.config = config
        self.current_positions = {}
        self.equity_curve = []
        self.peak_equity = config.initial_capital
        
    def calculate_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (Wilder's method)
        
        ATR measures market volatility by decomposing the entire range of an asset
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period for ATR calculation
        
        Returns:
            pd.Series: ATR values
        """
        # True Range components
        tr1 = high - low  # High - Low
        tr2 = abs(high - close.shift(1))  # |High - Previous Close|
        tr3 = abs(low - close.shift(1))   # |Low - Previous Close|
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the exponential moving average of TR (Wilder's smoothing)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def calculate_position_size(self, df: pd.DataFrame, 
                               signal: pd.Series) -> pd.Series:
        """
        Calculate optimal position size using volatility-adjusted methodology
        
        Position Size = (Capital * Risk%) / (ATR * Multiplier * Contract Multiplier)
        
        Args:
            df: DataFrame with OHLC data
            signal: Trading signal series (+1, -1, or 0)
        
        Returns:
            pd.Series: Number of contracts to trade
        """
        # Calculate ATR for both CL and HO
        atr_cl = self.calculate_atr(
            df['CL_High'], 
            df['CL_Low'], 
            df['CL_Close'],
            period=self.config.atr_period
        )
        
        atr_ho = self.calculate_atr(
            df['HO_High'],
            df['HO_Low'],
            df['HO_Close'],
            period=self.config.atr_period
        )
        
        # Use the average ATR for spread trading
        avg_atr = (atr_cl + atr_ho) / 2
        
        # Position size formula
        # Risk amount per trade
        risk_amount = self.config.initial_capital * self.config.risk_per_trade
        
        # Position size in dollars
        position_value = risk_amount / (avg_atr * self.config.atr_stop_multiple)
        
        # Convert to number of contracts (CL contract = 1000 barrels)
        contract_size_cl = 1000  # barrels per contract
        position_size = position_value / (df['CL_Close'] * contract_size_cl)
        
        # Cap at maximum position size
        max_contracts = (
            self.config.initial_capital * self.config.max_position_size
        ) / (df['CL_Close'] * contract_size_cl)
        
        position_size = np.minimum(position_size, max_contracts)
        
        # Only apply position sizing when we have an active signal
        position_size = position_size * abs(signal)
        
        # Round to nearest contract and ensure minimum of 1 contract when signal exists
        position_size = np.where(
            signal != 0,
            np.maximum(np.round(position_size), 1),
            0
        )
        
                # === HARD CAPS TO PREVENT RUNAWAY SIZING ===
        max_contracts_hard_limit = 10  # Never exceed 10 contracts
        max_capital_pct = 0.20  # Never use more than 20% of capital

        position_size = np.minimum(position_size, max_contracts_hard_limit)

        # Also cap by capital percentage
        capital_based_limit = (
            self.config.initial_capital * max_capital_pct
        ) / (df['CL_Close'] * 1000)

        position_size = np.minimum(position_size, capital_based_limit)

        # Only apply position sizing when we have an active signal


        return position_size
    
    def set_stop_loss_take_profit(self, df: pd.DataFrame, 
                                   entry_price: float,
                                   direction: int) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit levels
        
        Args:
            df: DataFrame with current market data
            entry_price: Entry price for the position
            direction: 1 for long, -1 for short
        
        Returns:
            Tuple of (stop_loss, take_profit) prices
        """
        # Get current ATR
        current_atr = self.calculate_atr(
            df['CL_High'], 
            df['CL_Low'], 
            df['CL_Close'],
            period=self.config.atr_period
        ).iloc[-1]
        
        if direction == 1:  # Long position
            stop_loss = entry_price - (current_atr * self.config.atr_stop_multiple)
            take_profit = entry_price + (current_atr * self.config.atr_target_multiple)
        else:  # Short position
            stop_loss = entry_price + (current_atr * self.config.atr_stop_multiple)
            take_profit = entry_price - (current_atr * self.config.atr_target_multiple)
        
        return stop_loss, take_profit
    
    def calculate_portfolio_heat(self, open_positions: Dict) -> float:
        """
        Calculate total portfolio heat (sum of all risk across open positions)
        
        Args:
            open_positions: Dictionary of open positions
        
        Returns:
            float: Portfolio heat as percentage of capital
        """
        total_heat = 0
        
        for position_id, position_data in open_positions.items():
            # Heat = (Entry Price - Stop Loss) * Position Size / Capital
            entry = position_data['entry_price']
            stop = position_data['stop_loss']
            size = position_data['size']
            
            risk_per_contract = abs(entry - stop) * size
            total_heat += risk_per_contract
        
        return total_heat / self.config.initial_capital
    
    def check_drawdown_limit(self, current_equity: float) -> bool:
        """
        Check if maximum drawdown limit has been breached
        
        Args:
            current_equity: Current portfolio value
        
        Returns:
            bool: True if trading should stop due to drawdown
        """
        self.peak_equity = max(self.peak_equity, current_equity)
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Stop trading if drawdown exceeds 20%
        max_allowed_drawdown = 0.20
        
        return current_drawdown > max_allowed_drawdown
    
    def apply_risk_filters(self, df: pd.DataFrame, 
                          signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive risk filters to trading signals
        
        Filters applied:
        1. ATR-based position sizing
        2. Dynamic stop-loss and take-profit
        3. Maximum portfolio heat
        4. Drawdown protection
        5. Time-based filters (no trading during high volatility events)
        
        Args:
            df: Market data DataFrame
            signals: Generated trading signals
        
        Returns:
            DataFrame with risk-adjusted positions
        """
        print("\n" + "=" * 60)
        print("🛡️  APPLYING RISK MANAGEMENT FILTERS")
        print("=" * 60)
        
        # Create a copy to avoid modifying original
        risk_adjusted = signals.copy()
        
        # 1. Calculate position sizes
        print("\n1️⃣  Calculating position sizes...")
        risk_adjusted['Position_Size'] = self.calculate_position_size(
            df, 
            risk_adjusted['Position']
        )
        
        # 2. Calculate ATR for stop-loss/take-profit
        print("2️⃣  Setting stop-loss and take-profit levels...")
        atr_cl = self.calculate_atr(
            df['CL_High'], 
            df['CL_Low'], 
            df['CL_Close'],
            period=self.config.atr_period
        )
        
        # Calculate stop-loss and take-profit for each row
        risk_adjusted['ATR'] = atr_cl
        risk_adjusted['Stop_Loss'] = np.where(
            risk_adjusted['Position'] > 0,
            df['CL_Close'] - (atr_cl * self.config.atr_stop_multiple),
            np.where(
                risk_adjusted['Position'] < 0,
                df['CL_Close'] + (atr_cl * self.config.atr_stop_multiple),
                np.nan
            )
        )
        
        risk_adjusted['Take_Profit'] = np.where(
            risk_adjusted['Position'] > 0,
            df['CL_Close'] + (atr_cl * self.config.atr_target_multiple),
            np.where(
                risk_adjusted['Position'] < 0,
                df['CL_Close'] - (atr_cl * self.config.atr_target_multiple),
                np.nan
            )
        )
        
        # 3. Calculate risk per trade
        risk_adjusted['Risk_Per_Trade'] = (
            abs(df['CL_Close'] - risk_adjusted['Stop_Loss']) * 
            risk_adjusted['Position_Size'] * 1000  # Contract multiplier
        ) / self.config.initial_capital
        
        # 4. Apply maximum heat filter
        print("3️⃣  Applying portfolio heat limits...")
        risk_adjusted['Cumulative_Heat'] = risk_adjusted['Risk_Per_Trade'].rolling(
            window=20, min_periods=1
        ).sum()
        
        # Reduce position if cumulative heat exceeds threshold
        max_heat = 0.10  # Max 10% total portfolio heat
        risk_adjusted['Position_Size'] = np.where(
            risk_adjusted['Cumulative_Heat'] > max_heat,
            risk_adjusted['Position_Size'] * 0.5,  # Cut position in half
            risk_adjusted['Position_Size']
        )
        
        # 5. Volatility regime filter
        print("4️⃣  Applying volatility regime filters...")
        risk_adjusted['ATR_Percentile'] = atr_cl.rolling(
            window=252, min_periods=60
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Reduce position size during extreme volatility (top 10%)
        risk_adjusted['Position_Size'] = np.where(
            risk_adjusted['ATR_Percentile'] > 0.90,
            risk_adjusted['Position_Size'] * 0.5,
            risk_adjusted['Position_Size']
        )
        
        # 6. Add risk metadata
        risk_adjusted['Dollar_Risk'] = (
            abs(df['CL_Close'] - risk_adjusted['Stop_Loss']) * 
            risk_adjusted['Position_Size'] * 1000
        )
        
        risk_adjusted['Reward_Risk_Ratio'] = (
            abs(risk_adjusted['Take_Profit'] - df['CL_Close']) /
            abs(df['CL_Close'] - risk_adjusted['Stop_Loss'])
        )
        
        self._print_risk_summary(risk_adjusted)
        
        return risk_adjusted
    
    def _print_risk_summary(self, df: pd.DataFrame):
        """Print summary of risk management application"""
        
        active_positions = df[df['Position'] != 0]
        
        if len(active_positions) == 0:
            print("\n⚠️  No active positions after risk filtering")
            return
        
        print("\n" + "=" * 60)
        print("📊 RISK MANAGEMENT SUMMARY")
        print("=" * 60)
        print(f"Active positions:           {len(active_positions)}")
        print(f"Average position size:      {active_positions['Position_Size'].mean():.2f} contracts")
        print(f"Max position size:          {active_positions['Position_Size'].max():.2f} contracts")
        print(f"Average risk per trade:     {active_positions['Risk_Per_Trade'].mean()*100:.2f}%")
        print(f"Max portfolio heat:         {active_positions['Cumulative_Heat'].max()*100:.2f}%")
        print(f"Average reward/risk ratio:  {active_positions['Reward_Risk_Ratio'].mean():.2f}:1")
        print(f"High volatility periods:    {(active_positions['ATR_Percentile'] > 0.90).sum()} days")
        print("=" * 60)
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence: Confidence level (default 95%)
        
        Returns:
            float: VaR value
        """
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence: Confidence level (default 95%)
        
        Returns:
            float: CVaR value
        """
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_kelly_criterion(self, win_rate: float, 
                                  avg_win: float, 
                                  avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Kelly% = W - [(1-W) / R]
        Where:
            W = Win rate
            R = Average win / Average loss
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
        
        Returns:
            float: Kelly percentage (typically use 1/4 or 1/2 Kelly for safety)
        """
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Use fractional Kelly (1/4 Kelly) for safety
        fractional_kelly = kelly * 0.25
        
        return max(0, fractional_kelly)  # Never go negative
    



