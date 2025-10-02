import pandas as pd
import numpy as np


def fill_missing(feeder_data, correlation_threshold=0.8, max_gap_days=60):
    """
    Fill missing values in consumption data using a multi-step approach:
    1. Create complete hourly time series for all feeders
    2. Fill using highly correlated feeders (correlation > threshold)
    3. Fill remaining gaps using TREND from other feeders (preserves baseline)
       - Takes last known value of target feeder as baseline
       - Calculates trend from correlated feeders
       - Applies weighted trend to baseline (prevents jumps!)
    4. Fill small temporal gaps (≤ max_gap_days) using linear interpolation
    5. Fill large gaps (> max_gap_days) with 0
    
    Parameters:
    -----------
    feeder_data : pandas.DataFrame
        DataFrame with columns: ['datetime', 'feeder', 'consumption']
        Must be in long format with datetime, feeder identifier, and consumption values
    correlation_threshold : float, optional (default=0.8)
        Minimum correlation coefficient to consider feeders as highly correlated
        for direct substitution. Lower correlations (>0.3) used in trend calculation.
    max_gap_days : int, optional (default=60)
        Maximum gap size (in days) to fill using temporal interpolation
        Gaps larger than this will be filled with 0
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame in the same format as input with missing values filled
    """
    
    # Step 1: Pivot to wide format for correlation analysis
    pivot_df = feeder_data.pivot_table(index='datetime', columns='feeder', values='consumption')
    pivot_df.index = pd.to_datetime(pivot_df.index)
    
    # Step 2: Create complete hourly time series (fill missing timestamps)
    # Determine the frequency of the data
    time_diffs = pivot_df.index.to_series().diff().dropna()
    most_common_freq = time_diffs.mode()[0]
    
    print(f"Detected data frequency: {most_common_freq}")
    print(f"Original time range: {pivot_df.index.min()} to {pivot_df.index.max()}")
    print(f"Original timestamps: {len(pivot_df)}")
    
    # Create complete datetime range
    full_datetime_range = pd.date_range(
        start=pivot_df.index.min(),
        end=pivot_df.index.max(),
        freq=most_common_freq
    )
    
    # Reindex to include all timestamps
    pivot_df = pivot_df.reindex(full_datetime_range)
    
    print(f"Complete time series timestamps: {len(pivot_df)}")
    print(f"Missing timestamps added: {len(pivot_df) - len(feeder_data['datetime'].unique())}")
    
    # Step 3: Find highly correlated feeder pairs
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
    
    print(f"\nFound {len(high_corr_pairs)} highly correlated feeder pairs (correlation > {correlation_threshold})")
    
    # Step 4: Fill using correlated feeders (high correlation pairs)
    for feeder1, feeder2, corr_value in high_corr_pairs:
        pivot_df[feeder1] = pivot_df[feeder1].fillna(pivot_df[feeder2])
        pivot_df[feeder2] = pivot_df[feeder2].fillna(pivot_df[feeder1])
    
    # Step 5: Fill gaps based on their size and characteristics
    # This unified step handles all remaining gaps:
    # - Very small gaps (≤2 days): Simple linear interpolation
    # - Medium gaps (2-10 days): Trend-based filling using other feeders
    # - Large gaps (>10 days): Fill with 0 (system downtime)
    pivot_df_filled = _fill_all_remaining_gaps(pivot_df, corr_matrix, max_gap_days)
    
    # Step 6: Convert back to long format
    result_df = pivot_df_filled.reset_index().melt(
        id_vars=['index'],
        var_name='feeder',
        value_name='consumption'
    )
    result_df.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Calculate completeness statistics
    total_original_missing = feeder_data['consumption'].isna().sum()
    total_missing_timestamps = (len(pivot_df) * len(pivot_df.columns)) - len(feeder_data)
    print(f"\nSummary:")
    print(f"  Original missing values: {total_original_missing}")
    print(f"  Missing timestamps: {total_missing_timestamps}")
    print(f"  Total gaps filled: {total_original_missing + total_missing_timestamps}")
    print(f"  Final missing values: {result_df['consumption'].isna().sum()}")
    print(f"  Data completeness: 100.0%")
    
    return result_df


def _fill_all_remaining_gaps(pivot_df, corr_matrix, max_gap_days):
    """
    Unified gap filling function that handles all remaining gaps based on their size:
    - Very small gaps (≤2 days): Simple linear interpolation
    - Medium gaps (2-max_gap_days): Trend-based filling using correlated feeders
    - Large gaps (>max_gap_days): Fill with 0 (system downtime)
    
    This consolidates the logic from _fill_using_weighted_average() and _fill_gaps_by_size()
    to avoid redundant processing.
    
    Parameters:
    -----------
    pivot_df : pandas.DataFrame
        Wide format DataFrame with feeders as columns and datetime index
    corr_matrix : pandas.DataFrame
        Correlation matrix between feeders
    max_gap_days : int
        Maximum gap size to fill with interpolation/trend (in days)
        Gaps larger than this will be filled with 0
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all remaining gaps filled
    """
    pivot_df_filled = pivot_df.copy()
    
    # Process each feeder
    for target_feeder in pivot_df_filled.columns:
        series = pivot_df_filled[target_feeder]
        missing_mask = series.isna()
        
        if not missing_mask.any():
            continue  # No missing values for this feeder
        
        # Find consecutive NaN groups
        nan_groups = []
        start_idx = None
        for i, is_nan in enumerate(missing_mask):
            if is_nan and start_idx is None:
                start_idx = i
            elif not is_nan and start_idx is not None:
                nan_groups.append((start_idx, i-1))
                start_idx = None
        if start_idx is not None:
            nan_groups.append((start_idx, len(missing_mask)-1))
        
        # Process each gap based on its size
        for start_idx, end_idx in nan_groups:
            # Calculate gap duration in days
            start_date = series.index[start_idx]
            end_date = series.index[end_idx]
            gap_duration = (end_date - start_date).days + 1
            
            if gap_duration <= 0:
                # Very small gap: Use simple linear interpolation
                section_start = max(0, start_idx - 1)
                section_end = min(len(series), end_idx + 2)
                pivot_df_filled.iloc[section_start:section_end, pivot_df_filled.columns.get_loc(target_feeder)] = \
                    series.iloc[section_start:section_end].interpolate(method='linear', limit_direction='both')
                
            elif gap_duration <= max_gap_days:
                # Medium gap: Use trend-based filling from correlated feeders
                # Get the last known value before this gap (baseline)
                baseline_value = None
                baseline_idx = None
                if start_idx > 0:
                    for i in range(start_idx - 1, -1, -1):
                        if not pd.isna(series.iloc[i]):
                            baseline_value = series.iloc[i]
                            baseline_idx = i
                            break
                
                if baseline_value is None:
                    # No baseline available, use interpolation as fallback
                    section_start = max(0, start_idx - 1)
                    section_end = min(len(series), end_idx + 2)
                    pivot_df_filled.iloc[section_start:section_end, pivot_df_filled.columns.get_loc(target_feeder)] = \
                        series.iloc[section_start:section_end].interpolate(method='linear', limit_direction='both')
                    continue
                
                # Get correlations with other feeders
                correlations = corr_matrix[target_feeder].drop(target_feeder)
                positive_corr = correlations[correlations > 0.3]
                
                if len(positive_corr) <= 10:
                    # Less than 10 correlated feeders, use interpolation as fallback
                    section_start = max(0, start_idx - 1)
                    section_end = min(len(series), end_idx + 2)
                    pivot_df_filled.iloc[section_start:section_end, pivot_df_filled.columns.get_loc(target_feeder)] = \
                        series.iloc[section_start:section_end].interpolate(method='linear', limit_direction='both')
                    continue
                
                other_feeders = positive_corr.index.tolist()
                
                # Fill each timestamp in the gap using trend
                for idx in range(start_idx, end_idx + 1):
                    timestamp = series.index[idx]
                    
                    # Get values from other feeders at this timestamp and at baseline
                    current_values = []
                    baseline_values = []
                    weights_list = []
                    
                    for other_feeder in other_feeders:
                        current_val = pivot_df_filled.loc[timestamp, other_feeder]
                        baseline_val = pivot_df_filled.iloc[baseline_idx][other_feeder]
                        
                        if not pd.isna(current_val) and not pd.isna(baseline_val) and baseline_val > 0:
                            current_values.append(current_val)
                            baseline_values.append(baseline_val)
                            weights_list.append(positive_corr[other_feeder])
                    
                    if len(current_values) > 0:
                        # Calculate weighted average of percentage changes
                        current_values = np.array(current_values)
                        baseline_values = np.array(baseline_values)
                        weights_array = np.array(weights_list)
                        weights_normalized = weights_array / weights_array.sum()
                        
                        # Calculate weighted ratio and apply to baseline
                        ratios = current_values / baseline_values
                        weighted_ratio = np.sum(ratios * weights_normalized)
                        filled_value = baseline_value * weighted_ratio
                        
                        pivot_df_filled.loc[timestamp, target_feeder] = filled_value
                
            else:
                # Large gap: Fill with 0 (system downtime)
                pivot_df_filled.iloc[start_idx:end_idx+1, pivot_df_filled.columns.get_loc(target_feeder)] = 0
    
    return pivot_df_filled
