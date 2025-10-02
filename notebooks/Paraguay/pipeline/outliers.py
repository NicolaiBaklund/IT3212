import pandas as pd
import numpy as np


def cap_outliers(feeder_data, 
                 stage1_window_days=30, stage1_threshold=2.70,
                 stage2_window_days=14, stage2_threshold=2.70,
                 stage3_window_days=7, stage3_threshold=2.70):
    """
    Cap outliers in consumption data using a three-stage rolling Z-score approach.
    Each stage identifies outliers and caps them at the 75th percentile, then the
    next stage operates on the already-capped data.
    
    Parameters:
    -----------
    feeder_data : pandas.DataFrame
        DataFrame with columns: ['datetime', 'feeder', 'consumption']
        Must be in long format with datetime, feeder identifier, and consumption values
    stage1_window_days : int, optional (default=30)
        Rolling window size in days for first stage (broader context)
    stage1_threshold : float, optional (default=2.60)
        Z-score threshold for first stage outlier detection
    stage2_window_days : int, optional (default=14)
        Rolling window size in days for second stage (medium context)
    stage2_threshold : float, optional (default=3.0)
        Z-score threshold for second stage outlier detection
    stage3_window_days : int, optional (default=7)
        Rolling window size in days for third stage (narrower context)
    stage3_threshold : float, optional (default=3.5)
        Z-score threshold for third stage outlier detection
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame in the same format as input with outliers capped
    """
    
    # Ensure datetime is in correct format
    feeder_data = feeder_data.copy()
    feeder_data['datetime'] = pd.to_datetime(feeder_data['datetime'])
    
    # Get unique feeders
    unique_feeders = feeder_data['feeder'].unique()
    
    # Store results for each feeder
    processed_feeders = []
    
    print(f"Processing {len(unique_feeders)} feeders with 3-stage outlier capping...")
    print("=" * 60)
    
    for feeder in unique_feeders:
        # Extract feeder data
        df_feeder = feeder_data[feeder_data['feeder'] == feeder].copy()
        df_feeder = df_feeder.sort_values('datetime')
        
        # Stage 1: First iteration with broad window
        window_hours_1 = stage1_window_days * 24
        zscore_mask_1 = _rolling_zscore_outliers(
            df_feeder, 
            window_hours_1, 
            threshold=stage1_threshold
        )
        
        percentile_75_1 = df_feeder['consumption'].rolling(window_hours_1, center=True, min_periods=1).quantile(0.75)
        df_feeder['consumption_capped_1'] = df_feeder['consumption'].copy()
        df_feeder.loc[zscore_mask_1, 'consumption_capped_1'] = percentile_75_1[zscore_mask_1]
        
        # Stage 2: Second iteration on already capped data
        window_hours_2 = stage2_window_days * 24
        df_temp = df_feeder.copy()
        df_temp['consumption'] = df_feeder['consumption_capped_1']
        zscore_mask_2 = _rolling_zscore_outliers(
            df_temp, 
            window_hours_2, 
            threshold=stage2_threshold
        )
        
        percentile_75_2 = df_feeder['consumption_capped_1'].rolling(window_hours_2, center=True, min_periods=1).quantile(0.75)
        df_feeder['consumption_capped_2'] = df_feeder['consumption_capped_1'].copy()
        df_feeder.loc[zscore_mask_2, 'consumption_capped_2'] = percentile_75_2[zscore_mask_2]
        
        # Stage 3: Third iteration on second capped data
        window_hours_3 = stage3_window_days * 24
        df_temp2 = df_feeder.copy()
        df_temp2['consumption'] = df_feeder['consumption_capped_2']
        zscore_mask_3 = _rolling_zscore_outliers(
            df_temp2, 
            window_hours_3, 
            threshold=stage3_threshold
        )
        
        percentile_75_3 = df_feeder['consumption_capped_2'].rolling(window_hours_3, center=True, min_periods=1).quantile(0.75)
        df_feeder['consumption_capped_final'] = df_feeder['consumption_capped_2'].copy()
        df_feeder.loc[zscore_mask_3, 'consumption_capped_final'] = percentile_75_3[zscore_mask_3]
        
        # Keep only necessary columns
        df_feeder['consumption'] = df_feeder['consumption_capped_final']
        df_feeder = df_feeder[['datetime', 'feeder', 'consumption']]
        
        # Report statistics
        n_outliers_1 = zscore_mask_1.sum()
        n_outliers_2 = zscore_mask_2.sum()
        n_outliers_3 = zscore_mask_3.sum()
        total_outliers = n_outliers_1 + n_outliers_2 + n_outliers_3
        
        if total_outliers > 0:
            print(f"{feeder}: Stage1={n_outliers_1}, Stage2={n_outliers_2}, "
                  f"Stage3={n_outliers_3}, Total={total_outliers}")
        
        processed_feeders.append(df_feeder)
    
    # Combine all feeders
    result_df = pd.concat(processed_feeders, ignore_index=True)
    result_df = result_df.sort_values(['feeder', 'datetime'])
    
    print("=" * 60)
    print("3-stage outlier capping completed!")
    
    return result_df


def _rolling_zscore_outliers(df, window_hours, threshold=3):
    """
    Detect outliers using rolling Z-score method.
    
    This function calculates Z-scores using a rolling window approach, where the mean 
    and standard deviation are computed over a sliding window centered at each point.
    Points with absolute Z-scores exceeding the threshold are flagged as outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time series data with a 'consumption' column
    window_hours : int
        Size of the rolling window in hours
    threshold : float, default=3
        Z-score threshold for outlier detection
        
    Returns:
    --------
    pandas.Series
        Boolean series where True indicates an outlier, False indicates normal data
    """
    df = df.copy()
    
    # Calculate rolling mean and standard deviation
    roll_mean = df['consumption'].rolling(window_hours, center=True, min_periods=1).mean()
    roll_std = df['consumption'].rolling(window_hours, center=True, min_periods=1).std()
    
    # Calculate Z-score: (value - local_mean) / local_std
    # Add small constant to std to prevent division by zero
    z = (df['consumption'] - roll_mean) / (roll_std + 1e-9)
    
    # Return boolean mask where absolute Z-score exceeds threshold
    return np.abs(z) > threshold


# Example usage:
if __name__ == "__main__":
    # Load the processed consumption data (after filling missing values)
    consumption_path = '../../../data/paraguay/electricity-consumption-processed.csv'
    df_consumption = pd.read_csv(consumption_path)
    
    # Cap outliers
    df_consumption_capped = cap_outliers(df_consumption)
    
    # Save the result
    output_path = '../../../data/paraguay/electricity-consumption-capped.csv'
    df_consumption_capped.to_csv(output_path, index=False)
    print(f"\nCapped data saved to: {output_path}")
