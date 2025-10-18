import pandas as pd
import numpy as np


def fill_missing(feeder_data, correlation_threshold=0.8, max_gap_days=5):
    """
    Fill missing values in consumption data using a three-step approach:
    1. Fill using highly correlated feeders
    2. Fill small temporal gaps (≤ max_gap_days) with mean of surrounding values
    3. Fill remaining gaps with 0
    
    Parameters:
    -----------
    feeder_data : pandas.DataFrame
        DataFrame with columns: ['datetime', 'feeder', 'consumption']
        Must be in long format with datetime, feeder identifier, and consumption values
    correlation_threshold : float, optional (default=0.8)
        Minimum correlation coefficient to consider feeders as highly correlated
    max_gap_days : int, optional (default=5)
        Maximum gap size (in days) to fill using temporal interpolation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame in the same format as input with missing values filled
    """
    
    # Step 1: Pivot to wide format for correlation analysis
    pivot_df = feeder_data.pivot_table(index='datetime', columns='feeder', values='consumption')
    pivot_df.index = pd.to_datetime(pivot_df.index)
    
    # Step 2: Find highly correlated feeder pairs
    corr_matrix = pivot_df.corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    print(f"Found {len(high_corr_pairs)} highly correlated feeder pairs (correlation > {correlation_threshold})")
    
    # Step 3: Fill using correlated feeders
    for feeder1, feeder2, corr_value in high_corr_pairs:
        pivot_df[feeder1] = pivot_df[feeder1].fillna(pivot_df[feeder2])
        pivot_df[feeder2] = pivot_df[feeder2].fillna(pivot_df[feeder1])
    
    # Step 4: Fill small temporal gaps
    pivot_df_temporal_filled = pivot_df.copy()
    
    for feeder in pivot_df_temporal_filled.columns:
        original_nan_count = pivot_df_temporal_filled[feeder].isna().sum()
        
        if original_nan_count > 0:
            pivot_df_temporal_filled[feeder] = _fill_small_gaps(
                pivot_df_temporal_filled[feeder], 
                max_gap_days
            )
            new_nan_count = pivot_df_temporal_filled[feeder].isna().sum()
            filled_count = original_nan_count - new_nan_count
            
            if filled_count > 0:
                print(f"Feeder {feeder}: Filled {filled_count} values using temporal interpolation")
    
    # Step 5: Fill remaining gaps with 0
    pivot_df_final = pivot_df_temporal_filled.fillna(0)
    
    # Count final fills with 0
    remaining_filled = (pivot_df_temporal_filled.isna().sum() > 0).sum()
    if remaining_filled > 0:
        print(f"Filled remaining gaps with 0 for {remaining_filled} feeders")
    
    # Step 6: Convert back to long format
    result_df = pivot_df_final.reset_index().melt(
        id_vars=['datetime'],
        var_name='feeder',
        value_name='consumption'
    )
    
    # Calculate completeness statistics
    total_original_missing = feeder_data['consumption'].isna().sum()
    print(f"\nSummary:")
    print(f"  Original missing values: {total_original_missing}")
    print(f"  Final missing values: {result_df['consumption'].isna().sum()}")
    print(f"  Data completeness: 100.0%")
    
    return result_df


def _fill_small_gaps(series, max_gap_days):
    """
    Fill NaN gaps smaller than max_gap_days with the mean of surrounding values
    
    Parameters:
    -----------
    series : pandas.Series
        Series with datetime index and potential NaN values
    max_gap_days : int
        Maximum gap size to fill (in days)
        
    Returns:
    --------
    pandas.Series
        Series with small gaps filled
    """
    series_filled = series.copy()
    
    # Ensure index is datetime
    if not isinstance(series_filled.index, pd.DatetimeIndex):
        series_filled.index = pd.to_datetime(series_filled.index)
    
    # Find all NaN values
    nan_mask = series_filled.isna()
    
    if not nan_mask.any():
        return series_filled
    
    # Get start and end indices of consecutive NaN groups
    nan_groups = []
    start_idx = None
    
    for i, is_nan in enumerate(nan_mask):
        if is_nan and start_idx is None:
            start_idx = i
        elif not is_nan and start_idx is not None:
            nan_groups.append((start_idx, i-1))
            start_idx = None
    
    # Handle case where series ends with NaN
    if start_idx is not None:
        nan_groups.append((start_idx, len(series_filled)-1))
    
    # Fill small gaps
    for start_idx, end_idx in nan_groups:
        # Calculate gap duration in days
        start_date = series_filled.index[start_idx]
        end_date = series_filled.index[end_idx]
        gap_duration = (end_date - start_date).days + 1
        
        if gap_duration <= max_gap_days:
            # Get surrounding values for mean calculation
            before_value = None
            after_value = None
            
            # Get value before the gap
            if start_idx > 0:
                before_value = series_filled.iloc[start_idx - 1]
            
            # Get value after the gap
            if end_idx < len(series_filled) - 1:
                after_value = series_filled.iloc[end_idx + 1]
            
            # Calculate fill value
            if before_value is not None and after_value is not None:
                fill_value = (before_value + after_value) / 2
            elif before_value is not None:
                fill_value = before_value
            elif after_value is not None:
                fill_value = after_value
            else:
                continue  # Skip if no surrounding values
            
            # Fill the gap
            series_filled.iloc[start_idx:end_idx+1] = fill_value
    
    return series_filled


# Example usage:
if __name__ == "__main__":
    # Load the raw consumption data
    consumption_path = '../../../data/paraguay/electricity-consumption-raw.csv'
    df_consumption = pd.read_csv(consumption_path)
    
    # Fill missing values
    df_consumption_filled = fill_missing(df_consumption)
    
    # Save the result
    output_path = '../../../data/paraguay/electricity-consumption-processed.csv'
    df_consumption_filled.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
