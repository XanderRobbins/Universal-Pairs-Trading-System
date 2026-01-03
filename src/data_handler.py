"""
Data acquisition and preprocessing for CL-HO spread strategy
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional, Dict
from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    """Comprehensive data handling with validation and quality checks"""
    
    def __init__(self, config):
        self.config = config
        self.cl_data = None
        self.ho_data = None
        self.df = None
        self.validation_results = {}
        
    def fetch_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Fetch historical data with error handling and validation
        
        Returns:
            pd.DataFrame: Cleaned and aligned price data
        """
        if verbose:
            print("=" * 60)
            print("📥 FETCHING MARKET DATA")
            print("=" * 60)
        
        try:
            # Download data
            if verbose:
                print(f"Downloading {self.config.cl_ticker} data...")
            cl_raw = yf.download(
                self.config.cl_ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                auto_adjust=True  # Adjust for splits/dividends
            )
            
            if verbose:
                print(f"Downloading {self.config.ho_ticker} data...")
            ho_raw = yf.download(
                self.config.ho_ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                auto_adjust=True
            )
            
            # Clean data
            cl_clean = self._clean_dataframe(cl_raw, 'CL')
            ho_clean = self._clean_dataframe(ho_raw, 'HO')
            
            # Merge and align
            df = pd.DataFrame({
                'CL_Close': cl_clean['Close'],
                'CL_High': cl_clean['High'],
                'CL_Low': cl_clean['Low'],
                'CL_Volume': cl_clean['Volume'],
                'HO_Close': ho_clean['Close'],
                'HO_High': ho_clean['High'],
                'HO_Low': ho_clean['Low'],
                'HO_Volume': ho_clean['Volume']
            }).dropna()
            
            # Data quality checks
            df = self._quality_filter(df)
            
            # Store clean data
            self.df = df
            self.cl_data = cl_clean
            self.ho_data = ho_clean
            
            if verbose:
                print(f"\n✅ Successfully loaded {len(df)} trading days")
                print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
                print(f"   CL price range: ${df['CL_Close'].min():.2f} - ${df['CL_Close'].max():.2f}")
                print(f"   HO price range: ${df['HO_Close'].min():.2f} - ${df['HO_Close'].max():.2f}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Clean multi-index columns and handle missing data"""
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.droplevel(1)
        
        # Replace zeros with NaN
        df = df.replace(0, np.nan)
        
        # Forward fill small gaps (FIXED METHOD)
        df = df.ffill(limit=3)
        
        # Drop remaining NaNs
        df = df.dropna()
        
        return df
    
    def _quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters"""
        original_len = len(df)
        
        # Remove prices below threshold
        df = df[
            (df['CL_Close'] > self.config.min_price_threshold) &
            (df['HO_Close'] > self.config.min_price_threshold)
        ]
        
        # Remove extreme outliers (z-score method)
        for col in ['CL_Close', 'HO_Close']:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < self.config.outlier_std_threshold]
        
        # Remove days with zero volume
        df = df[(df['CL_Volume'] > 0) & (df['HO_Volume'] > 0)]
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"   Filtered out {removed} low-quality data points ({removed/original_len*100:.2f}%)")
        
        return df
    
    def compute_crack_spread(self, method: str = 'log') -> pd.Series:
        """
        Compute crack spread (CL-HO relationship)
        
        Args:
            method: 'log' for log spread, 'simple' for simple spread,
                   'ratio' for price ratio
        
        Returns:
            pd.Series: Computed spread
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        if method == 'log':
            spread = np.log(self.df['CL_Close']) - np.log(self.df['HO_Close'])
        elif method == 'simple':
            spread = self.df['CL_Close'] - self.df['HO_Close']
        elif method == 'ratio':
            spread = self.df['CL_Close'] / self.df['HO_Close']
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return spread
    
    def test_cointegration(self, verbose: bool = True) -> Dict[str, float]:
        """
        Comprehensive cointegration analysis
        
        Returns:
            Dict with test results
        """
        if verbose:
            print("\n" + "=" * 60)
            print("🔬 STATISTICAL VALIDATION")
            print("=" * 60)
        
        cl = self.df['CL_Close'].values
        ho = self.df['HO_Close'].values
        
        # Cointegration test
        _, coint_pval, _ = coint(cl, ho)
        is_cointegrated = coint_pval < self.config.cointegration_pvalue
        
        if verbose:
            print(f"\n1. Cointegration Test (Engle-Granger)")
            print(f"   P-value: {coint_pval:.6f}")
            print(f"   Result: {'✅ COINTEGRATED' if is_cointegrated else '❌ NOT COINTEGRATED'}")
            print(f"   Interpretation: {'Strong mean-reversion expected' if is_cointegrated else 'Weak relationship'}")
        
        results = {
            'coint_pvalue': coint_pval,
            'is_cointegrated': is_cointegrated
        }
        
        self.validation_results.update(results)
        return results
    
    def test_stationarity(self, spread: pd.Series, verbose: bool = True) -> Dict[str, float]:
        """Augmented Dickey-Fuller test for spread stationarity"""
        result = adfuller(spread.dropna(), autolag='AIC')
        adf_stat = result[0]
        pvalue = result[1]
        critical_values = result[4]
        is_stationary = pvalue < 0.05
        
        if verbose:
            print(f"\n2. Stationarity Test (Augmented Dickey-Fuller)")
            print(f"   ADF Statistic: {adf_stat:.4f}")
            print(f"   P-value: {pvalue:.6f}")
            print(f"   Critical values: 1%={critical_values['1%']:.3f}, "
                  f"5%={critical_values['5%']:.3f}, 10%={critical_values['10%']:.3f}")
            print(f"   Result: {'✅ STATIONARY' if is_stationary else '❌ NON-STATIONARY'}")
        
        results = {
            'adf_statistic': adf_stat,
            'adf_pvalue': pvalue,
            'is_stationary': is_stationary
        }
        
        self.validation_results.update(results)
        return results
    
    def calculate_half_life(self, spread: pd.Series, verbose: bool = True) -> float:
        """
        Calculate mean-reversion half-life using Ornstein-Uhlenbeck process
        
        Returns:
            float: Half-life in days
        """
        # Prepare data for regression
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align
        spread_lag, spread_diff = spread_lag.align(spread_diff, join='inner')
        
        # OLS regression: Δy_t = α + β*y_{t-1} + ε_t
        # Half-life = -log(2) / β
        beta = np.polyfit(spread_lag, spread_diff, 1)[0]
        half_life = -np.log(2) / beta if beta < 0 else np.inf
        
        if verbose:
            print(f"\n3. Mean-Reversion Speed (Half-Life)")
            print(f"   Half-life: {half_life:.2f} days")
            if half_life < 30:
                print(f"   Assessment: ✅ FAST mean-reversion (excellent for trading)")
            elif half_life < 60:
                print(f"   Assessment: ✅ MODERATE mean-reversion (good for trading)")
            else:
                print(f"   Assessment: ⚠️  SLOW mean-reversion (requires patience)")
        
        self.validation_results['half_life'] = half_life
        return half_life
    
    def calculate_hedge_ratio(self, method: str = 'ols') -> float:
        """
        Calculate optimal hedge ratio between CL and HO
        
        Args:
            method: 'ols' for ordinary least squares, 'tls' for total least squares
        
        Returns:
            float: Hedge ratio (beta coefficient)
        """
        cl = self.df['CL_Close'].values
        ho = self.df['HO_Close'].values
        
        if method == 'ols':
            # OLS regression: CL = α + β*HO + ε
            beta = np.polyfit(ho, cl, 1)[0]
        elif method == 'tls':
            # Total least squares (accounts for noise in both variables)
            from scipy.linalg import svd
            X = np.vstack([ho, cl]).T
            X_centered = X - X.mean(axis=0)
            U, s, Vt = svd(X_centered)
            beta = Vt[0, 1] / Vt[0, 0]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"\n4. Hedge Ratio")
        print(f"   Beta (CL/HO): {beta:.4f}")
        print(f"   Interpretation: {beta:.4f} barrels of HO hedge 1 barrel of CL")
        
        self.validation_results['hedge_ratio'] = beta
        return beta
    
    def generate_summary_report(self) -> str:
        """Generate summary of data validation"""
        if not self.validation_results:
            return "No validation performed yet."
        
        report = "\n" + "=" * 60
        report += "\n📊 DATA VALIDATION SUMMARY"
        report += "\n" + "=" * 60
        
        # Cointegration
        if self.validation_results.get('is_cointegrated'):
            report += "\n✅ Pair is cointegrated - suitable for mean-reversion"
        else:
            report += "\n❌ Pair is NOT cointegrated - high risk strategy"
        
        # Stationarity
        if self.validation_results.get('is_stationary'):
            report += "\n✅ Spread is stationary - predictable behavior"
        else:
            report += "\n❌ Spread is non-stationary - may trend"
        
        # Half-life
        hl = self.validation_results.get('half_life', float('inf'))
        if hl < 60:
            report += f"\n✅ Half-life of {hl:.1f} days - good for short-term trading"
        else:
            report += f"\n⚠️  Half-life of {hl:.1f} days - slow mean-reversion"
        
        # Overall assessment
        report += "\n" + "-" * 60
        all_good = (
            self.validation_results.get('is_cointegrated', False) and
            self.validation_results.get('is_stationary', False) and
            hl < 60
        )
        
        if all_good:
            report += "\n🎯 OVERALL: EXCELLENT pair for mean-reversion strategy"
        else:
            report += "\n⚠️  OVERALL: Proceed with caution - mixed signals"
        
        report += "\n" + "=" * 60 + "\n"
        return report
    

    def calculate_rolling_cointegration(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate cointegration on a rolling basis
        
        Args:
            window: Rolling window in days (252 = 1 year, 126 = 6 months)
        
        Returns:
            DataFrame with rolling cointegration p-values and status
        """
        print(f"\n🔄 Calculating rolling cointegration (window={window} days)...")
        
        cl = self.df['CL_Close'].values
        ho = self.df['HO_Close'].values  # Will work for NG too
        
        rolling_results = []
        
        # Calculate cointegration for each rolling window
        for i in range(window, len(cl)):
            # Extract window of data
            cl_window = cl[i-window:i]
            ho_window = ho[i-window:i]
            
            # Run cointegration test
            try:
                _, pvalue, _ = coint(cl_window, ho_window)
                rolling_results.append({
                    'Date': self.df.index[i],
                    'Coint_PValue': pvalue,
                    'Is_Cointegrated': pvalue < 0.05
                })
            except Exception as e:
                # Handle calculation errors gracefully
                rolling_results.append({
                    'Date': self.df.index[i],
                    'Coint_PValue': np.nan,
                    'Is_Cointegrated': False
                })
        
        results_df = pd.DataFrame(rolling_results).set_index('Date')
        
        # Print summary statistics
        valid_results = results_df['Coint_PValue'].dropna()
        pct_cointegrated = (results_df['Is_Cointegrated'].sum() / len(results_df)) * 100
        
        print(f"✅ Analysis complete:")
        print(f"   Cointegrated: {pct_cointegrated:.1f}% of the time")
        print(f"   Mean p-value: {valid_results.mean():.4f}")
        print(f"   Min p-value: {valid_results.min():.4f}")
        print(f"   Max p-value: {valid_results.max():.4f}")
        
        return results_df