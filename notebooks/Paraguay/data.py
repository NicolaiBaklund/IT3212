import pandas as pd

class data:
    """
    Base data class containing file paths and common data functionality
    """


    consumption_path = '../../data/paraguay/electricity-consumption-raw.csv'
    weather_path = '../../data/paraguay/meteorological-raw.csv'
    location_path = '../../data/paraguay/substations-geographical-location.csv'
    
    def __init__(self, name=None):
        self.name = name
        self.data = None
    
    def load_data(self):
        """Load data from file - to be implemented by subclasses"""
        pass
    
    def get_data(self):
        """Return loaded data"""
        return self.data


class weather:
    """
    Should contain weather 
    """

class substations(data):
    """
    Substations class that contains all substation objects found in consumption_path.
    Automatically loads all unique substations from the consumption data.
    """
    
    def __init__(self):
        super().__init__()
        self.substation_objects = []
        self.substation_ids = []
        
        # Automatically load all substations
        self.load_all_substations()
    
    def load_all_substations(self):
        """Load all unique substations from the consumption data"""
        try:
            # Load the consumption dataset
            df = pd.read_csv(self.consumption_path)
            
            # Get unique substation IDs
            unique_substation_ids = df['substation'].unique()
            self.substation_ids = sorted(unique_substation_ids)
            
            # Create substation objects for each unique substation
            for substation_id in self.substation_ids:
                new_substation = substation(substation_id)
                self.substation_objects.append(new_substation)
            
            print(f"Loaded {len(self.substation_objects)} substations: {self.substation_ids}")
            
        except Exception as e:
            print(f"Error loading substations: {e}")
    
    def get_substations(self):
        """Return list of all substation objects"""
        return self.substation_objects
    
    def get_substation_ids(self):
        """Return list of all substation IDs"""
        return self.substation_ids
    
    def get_substation_count(self):
        """Return number of substations"""
        return len(self.substation_objects)
    
    def get_substation_by_id(self, substation_id):
        """
        Get a specific substation object by its ID
        
        Parameters:
        -----------
        substation_id : str or int
            The ID of the substation to retrieve
            
        Returns:
        --------
        substation object or None
            The substation object if found, None otherwise
        """
        for substation_obj in self.substation_objects:
            if substation_obj.get_substation_id() == substation_id:
                return substation_obj
        
        print(f"Substation with ID {substation_id} not found")
        return None
    
    def get_total_feeder_count(self):
        """Return total number of feeders across all substations"""
        total_feeders = sum(substation_obj.get_feeder_count() for substation_obj in self.substation_objects)
        return total_feeders
    
    def get_substations_summary(self):
        """
        Get a summary of all substations including their feeder counts
        
        Returns:
        --------
        list of dict
            List containing summary information for each substation
        """
        summary = []
        for substation_obj in self.substation_objects:
            summary.append({
                'substation_id': substation_obj.get_substation_id(),
                'substation_name': substation_obj.get_substation_name(),
                'feeder_count': substation_obj.get_feeder_count(),
                'feeder_ids': substation_obj.get_feeder_ids(),
                'coordinates': substation_obj.get_coordinates()
            })
        return summary
    
    def print_summary(self):
        """Print a formatted summary of all substations"""
        print("="*60)
        print("SUBSTATIONS SUMMARY")
        print("="*60)
        print(f"Total substations: {self.get_substation_count()}")
        print(f"Total feeders: {self.get_total_feeder_count()}")
        print("-"*60)
        
        for substation_obj in self.substation_objects:
            coords = substation_obj.get_coordinates()
            coord_str = f"({coords[0]:.4f}, {coords[1]:.4f})" if coords[0] is not None else "N/A"
            
            print(f"Substation {substation_obj.get_substation_id()}: {substation_obj.get_substation_name()}")
            print(f"  Location: {coord_str}")
            print(f"  Feeders: {substation_obj.get_feeder_count()} ({substation_obj.get_feeder_ids()})")
            print()


class substation(data):
    """
    Substation class that inherits from data.
    Should contain different feeder objects
    """
    
    def __init__(self, substation_id):
        super().__init__()
        self.substation_id = substation_id
        self.latitude = None
        self.longitude = None
        self.name = None
        self.feeders = []
        
        # Automatically load location data and feeders
        self.load_location_data()
        self.load_feeders()
        
    def load_location_data(self):
        """Load location data (latitude, longitude, name) for this substation"""
        try:
            # Load the location dataset
            df = pd.read_csv(self.location_path)
            
            # Find the row for this substation
            substation_row = df[df['Code'] == self.substation_id]
            
            if not substation_row.empty:
                self.latitude = substation_row.iloc[0]['Latitude']
                self.longitude = substation_row.iloc[0]['Longitude']
                self.name = substation_row.iloc[0]['Substation Name']
                print(f"Loaded location data for substation {self.substation_id}: {self.name}")
            else:
                print(f"Warning: No location data found for substation {self.substation_id}")
                
        except Exception as e:
            print(f"Error loading location data for substation {self.substation_id}: {e}")
    
    def load_feeders(self):
        """Load all feeders that belong to this substation"""
        try:
            # Load the entire consumption dataset
            df = pd.read_csv(self.consumption_path)
            
            # Get unique feeders for this substation
            substation_feeders = df[df['substation'] == self.substation_id]['feeder'].unique()
            
            # Create feeder objects for each unique feeder
            for feeder_id in substation_feeders:
                new_feeder = feeder(feeder_id)
                self.feeders.append(new_feeder)
            
            print(f"Loaded {len(self.feeders)} feeders for substation {self.substation_id}: {sorted(substation_feeders)}")
            
        except Exception as e:
            print(f"Error loading feeders for substation {self.substation_id}: {e}")
        
    def set_coordinates(self, latitude, longitude):
        """Set the geographical coordinates of the substation"""
        self.latitude = latitude
        self.longitude = longitude
    
    def get_coordinates(self):
        """Return the coordinates as a tuple (latitude, longitude)"""
        return (self.latitude, self.longitude)
        
    def add_feeder(self, feeder):
        """Add a feeder to this substation"""
        self.feeders.append(feeder)
        
    def get_feeders(self):
        """Return list of feeders in this substation"""
        return self.feeders
    
    def get_feeder_count(self):
        """Return number of feeders in this substation"""
        return len(self.feeders)
    
    def get_feeder_ids(self):
        """Return list of feeder IDs in this substation"""
        return [feeder.get_feeder_id() for feeder in self.feeders]
    
    def get_substation_id(self):
        """Return substation ID"""
        return self.substation_id
    
    def get_substation_name(self):
        """Return substation name"""
        return self.name


class feeder(data):
    """
    Feeder class that inherits from data.
    Represents an individual electrical feeder within a substation
    """
    
    def __init__(self, feeder_id, name=None):
        super().__init__(name)
        self.feeder_id = feeder_id
        # Extract substation ID as the first letter of feeder ID
        self.substation_id = str(feeder_id)[0] if feeder_id else None
        self.consumption_data = self.load_consumption_data()
        
    def load_consumption_data(self):
        """Load consumption data for this feeder"""
        try:
            # Load the entire consumption dataset
            df = pd.read_csv(self.consumption_path)
            
            # Filter data for this specific feeder
            feeder_data = df[df['feeder'] == self.feeder_id].copy()
            
            # Convert datetime column to datetime type
            feeder_data['datetime'] = pd.to_datetime(feeder_data['datetime'])
            
            # Select only datetime and consumption columns
            result = feeder_data[['datetime', 'consumption']].copy()
            
            # Sort by datetime
            result = result.sort_values('datetime').reset_index(drop=True)
            
            return result
            
        except Exception as e:
            print(f"Error loading consumption data for feeder {self.feeder_id}: {e}")
            return None
    
    def get_feeder_id(self):
        """Return feeder ID"""
        return self.feeder_id
    
    def get_substation_id(self):
        """Return the ID of the substation this feeder belongs to"""
        return self.substation_id
    
    def get_consumption_data(self):
        """Return consumption data for this feeder"""
        return self.consumption_data
    
    def get_consumption_dataframe(self):
        """
        Return consumption data as a DataFrame with datetime and consumption columns
        """
        if self.consumption_data is not None:
            return self.consumption_data.copy()
        else:
            return None
        
    def plot(self, start_date=None, end_date=None, title=None):
        """
        Plot consumption data for this feeder within the specified date range
        
        Parameters:
        -----------
        start_date : str or pd.Timestamp, optional
            Start date for plotting (e.g., '2017-01-01' or '2017-01-01 12:00:00')
            If None, uses the earliest available date
        end_date : str or pd.Timestamp, optional
            End date for plotting (e.g., '2020-12-31' or '2020-12-31 23:59:59')
            If None, uses the latest available date
        title : str, optional
            Custom title for the plot. If None, uses default title with feeder ID
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive plotly figure object
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Error: plotly is required for plotting. Please install it with: pip install plotly")
            return None
            
        if self.consumption_data is None or self.consumption_data.empty:
            print(f"No consumption data available for feeder {self.feeder_id}")
            return None
            
        # Work with a copy of the data
        plot_data = self.consumption_data.copy()
        
        # Convert string dates to pandas timestamps if provided
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            
        # Filter data by date range
        if start_date is not None:
            plot_data = plot_data[plot_data['datetime'] >= start_date]
        if end_date is not None:
            plot_data = plot_data[plot_data['datetime'] <= end_date]
            
        if plot_data.empty:
            print(f"No data available for feeder {self.feeder_id} in the specified date range")
            return None
            
        # Create the plot
        fig = go.Figure()
        
        # Add consumption line
        fig.add_trace(go.Scatter(
            x=plot_data['datetime'],
            y=plot_data['consumption'],
            mode='lines',
            name=f'Feeder {self.feeder_id}',
            line=dict(width=1.5),
            hovertemplate='<b>Date:</b> %{x}<br><b>Consumption:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Set title
        if title is None:
            date_range_str = ""
            if start_date is not None or end_date is not None:
                start_str = start_date.strftime('%Y-%m-%d') if start_date is not None else "Start"
                end_str = end_date.strftime('%Y-%m-%d') if end_date is not None else "End"
                date_range_str = f" ({start_str} to {end_str})"
            title = f"Electricity Consumption - Feeder {self.feeder_id}{date_range_str}"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Date",
            yaxis_title="Consumption",
            hovermode='x unified',
            showlegend=True,
            template="plotly_white",
            height=500
        )
        
        # Show the plot
        fig.show()
        
        # Print some basic statistics about the plotted data
        non_null_data = plot_data['consumption'].dropna()
        if not non_null_data.empty:
            print(f"\nPlotted data statistics for Feeder {self.feeder_id}:")
            print(f"  Date range: {plot_data['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {plot_data['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
            print(f"  Total records: {len(plot_data):,}")
            print(f"  Non-null records: {len(non_null_data):,}")
            print(f"  Missing records: {len(plot_data) - len(non_null_data):,}")
            print(f"  Mean consumption: {non_null_data.mean():.2f}")
            print(f"  Min consumption: {non_null_data.min():.2f}")
            print(f"  Max consumption: {non_null_data.max():.2f}")
        
        return fig
    
    
    def get_missing_analysis(self):
        """
        Comprehensive Missing Data Analysis for this feeder
        Returns detailed statistics about missing data patterns
        """
        import numpy as np
        from datetime import timedelta
        
        if self.consumption_data is None:
            print(f"No data available for feeder {self.feeder_id}")
            return None
            
        print("="*80)
        print(f"COMPREHENSIVE MISSING DATA ANALYSIS - FEEDER {self.feeder_id}")
        print("="*80)

        # Define the expected time range (2017-2020, hourly data)
        start_date = pd.Timestamp('2017-01-01 00:00:00')
        end_date = pd.Timestamp('2020-12-31 23:00:00')
        expected_datetime_range = pd.date_range(start=start_date, end=end_date, freq='H')
        expected_total_records = len(expected_datetime_range)

        print(f"\n📅 EXPECTED DATA PERIOD:")
        print(f"   Start: {start_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   End: {end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Expected records: {expected_total_records:,} (hourly data)")

        # Get feeder data
        feeder_data = self.consumption_data.copy()
        
        # 1. OVERALL STATISTICS
        actual_records = len(feeder_data)
        available_records = feeder_data['consumption'].notna().sum()
        missing_records = expected_total_records - available_records
        missing_percentage = (missing_records / expected_total_records) * 100
        null_records = feeder_data['consumption'].isna().sum()

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Feeder ID: {self.feeder_id}")
        print(f"   Expected total records: {expected_total_records:,}")
        print(f"   Actual records in dataset: {actual_records:,}")
        print(f"   Available records (non-null): {available_records:,}")
        print(f"   Null records in dataset: {null_records:,}")
        print(f"   Missing records (not in dataset): {expected_total_records - actual_records:,}")
        print(f"   Total missing/null records: {missing_records:,}")
        print(f"   Missing percentage: {missing_percentage:.2f}%")
        print(f"   Data completeness: {100 - missing_percentage:.2f}%")

        # 2. TIME-BASED ANALYSIS
        print(f"\n⏰ TIME-BASED MISSING DATA ANALYSIS:")

        yearly_analysis = []
        for year in range(2017, 2021):
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31 23:59:59')
            
            # Expected records for this year
            year_expected_hours = len(pd.date_range(start=year_start, end=year_end, freq='H'))
            
            # Actual records for this year
            year_data = feeder_data[
                (feeder_data['datetime'] >= year_start) & 
                (feeder_data['datetime'] <= year_end)
            ]
            year_actual_total = len(year_data)
            year_available_total = year_data['consumption'].notna().sum()
            year_missing_total = year_expected_hours - year_available_total
            year_missing_pct = (year_missing_total / year_expected_hours) * 100
            
            yearly_analysis.append({
                'year': year,
                'expected_total': year_expected_hours,
                'actual_total': year_actual_total,
                'available_total': year_available_total,
                'missing_total': year_missing_total,
                'missing_percentage': year_missing_pct
            })

        print(f"   Missing data by year:")
        print(f"   {'Year':<6} {'Missing %':<10} {'Missing/Expected':<17} {'Available':<10}")
        print(f"   {'-'*6:<6} {'-'*10:<10} {'-'*17:<17} {'-'*10:<10}")
        for analysis in yearly_analysis:
            print(f"   {analysis['year']:<6} {analysis['missing_percentage']:<10.1f} {analysis['missing_total']:>7,}/{analysis['expected_total']:<8,} {analysis['available_total']:>9,}")

        # 3. MONTHLY ANALYSIS
        print(f"\n📅 MONTHLY MISSING DATA ANALYSIS:")
        monthly_stats = []
        for year in range(2017, 2021):
            for month in range(1, 13):
                month_start = pd.Timestamp(f'{year}-{month:02d}-01')
                if month == 12:
                    month_end = pd.Timestamp(f'{year+1}-01-01') - pd.Timedelta(hours=1)
                else:
                    month_end = pd.Timestamp(f'{year}-{month+1:02d}-01') - pd.Timedelta(hours=1)
                
                # Expected records for this month
                month_expected_hours = len(pd.date_range(start=month_start, end=month_end, freq='H'))
                
                # Actual records for this month
                month_data = feeder_data[
                    (feeder_data['datetime'] >= month_start) & 
                    (feeder_data['datetime'] <= month_end)
                ]
                month_available = month_data['consumption'].notna().sum()
                month_missing = month_expected_hours - month_available
                month_missing_pct = (month_missing / month_expected_hours) * 100 if month_expected_hours > 0 else 0
                
                monthly_stats.append({
                    'year_month': f"{year}-{month:02d}",
                    'expected': month_expected_hours,
                    'available': month_available,
                    'missing': month_missing,
                    'missing_pct': month_missing_pct
                })

        # Show months with highest missing data
        monthly_df = pd.DataFrame(monthly_stats)
        worst_months = monthly_df.nlargest(10, 'missing_pct')
        
        print(f"   Top 10 months with most missing data:")
        print(f"   {'Month':<8} {'Missing %':<10} {'Missing/Expected':<17} {'Available':<10}")
        print(f"   {'-'*8:<8} {'-'*10:<10} {'-'*17:<17} {'-'*10:<10}")
        for _, row in worst_months.iterrows():
            print(f"   {row['year_month']:<8} {row['missing_pct']:<10.1f} {row['missing']:>7,}/{row['expected']:<8,} {row['available']:>9,}")

        # 4. BIGGEST TIME GAPS ANALYSIS
        print(f"\n🕳️  BIGGEST TIME GAPS (Consecutive missing periods):")
        
        # Sort data by datetime
        sorted_data = feeder_data.sort_values('datetime').copy()
        sorted_data['is_missing'] = sorted_data['consumption'].isna()
        
        # Find consecutive missing periods
        sorted_data['gap_group'] = (sorted_data['is_missing'] != sorted_data['is_missing'].shift()).cumsum()
        missing_groups = sorted_data[sorted_data['is_missing']].groupby('gap_group')
        
        gaps = []
        for group_id, group in missing_groups:
            if len(group) > 1:  # Only consider gaps longer than 1 record
                start_time = group['datetime'].min()
                end_time = group['datetime'].max()
                # Calculate actual time difference
                time_diff = end_time - start_time
                duration_hours = time_diff.total_seconds() / 3600
                duration_days = duration_hours / 24
                
                gaps.append({
                    'start': start_time,
                    'end': end_time,
                    'duration_hours': duration_hours,
                    'duration_days': duration_days,
                    'records_missing': len(group)
                })
        
        # Sort gaps by duration
        gaps = sorted(gaps, key=lambda x: x['duration_hours'], reverse=True)[:10]
        
        if gaps:
            print(f"   {'Rank':<4} {'Start Date':<12} {'End Date':<12} {'Duration':<15} {'Records':<8}")
            print(f"   {'-'*4:<4} {'-'*12:<12} {'-'*12:<12} {'-'*15:<15} {'-'*8:<8}")
            
            for i, gap in enumerate(gaps, 1):
                start_str = gap['start'].strftime('%Y-%m-%d')
                end_str = gap['end'].strftime('%Y-%m-%d')
                if gap['duration_days'] >= 1:
                    duration_str = f"{gap['duration_days']:.1f} days"
                else:
                    duration_str = f"{gap['duration_hours']:.0f} hours"
                print(f"   {i:<4} {start_str:<12} {end_str:<12} {duration_str:<15} {gap['records_missing']:<8}")
        else:
            print("   No significant gaps found (all missing data are single records)")

        # 5. DATA QUALITY SUMMARY
        print(f"\n📈 DATA QUALITY SUMMARY:")
        
        # Categorize data quality
        if missing_percentage == 0:
            quality_level = "EXCELLENT"
            quality_color = "🟢"
        elif missing_percentage <= 5:
            quality_level = "GOOD"
            quality_color = "🟡"
        elif missing_percentage <= 20:
            quality_level = "FAIR"
            quality_color = "🟠"
        elif missing_percentage <= 50:
            quality_level = "POOR"
            quality_color = "🔴"
        else:
            quality_level = "VERY POOR"
            quality_color = "🔴"
        
        print(f"   Data Quality: {quality_color} {quality_level}")
        print(f"   Completeness: {100 - missing_percentage:.2f}%")
        print(f"   Recommendation: ", end="")
        
        if missing_percentage == 0:
            print("Data is complete and ready for analysis")
        elif missing_percentage <= 5:
            print("Minor missing data - suitable for most analyses")
        elif missing_percentage <= 20:
            print("Moderate missing data - consider interpolation for time series analysis")
        elif missing_percentage <= 50:
            print("Significant missing data - use with caution, consider data imputation")
        else:
            print("Extensive missing data - not recommended for reliable analysis")

        # 6. STATISTICAL SUMMARY OF AVAILABLE DATA
        if available_records > 0:
            print(f"\n📊 STATISTICAL SUMMARY OF AVAILABLE DATA:")
            consumption_stats = feeder_data['consumption'].describe()
            print(f"   Count: {consumption_stats['count']:,.0f}")
            print(f"   Mean: {consumption_stats['mean']:.2f}")
            print(f"   Std: {consumption_stats['std']:.2f}")
            print(f"   Min: {consumption_stats['min']:.2f}")
            print(f"   25%: {consumption_stats['25%']:.2f}")
            print(f"   50%: {consumption_stats['50%']:.2f}")
            print(f"   75%: {consumption_stats['75%']:.2f}")
            print(f"   Max: {consumption_stats['max']:.2f}")

        print(f"\n" + "="*80)
        
        # Return summary statistics for programmatic use
        return {
            'feeder_id': self.feeder_id,
            'expected_records': expected_total_records,
            'actual_records': actual_records,
            'available_records': available_records,
            'missing_records': missing_records,
            'missing_percentage': missing_percentage,
            'data_quality': quality_level,
            'yearly_analysis': yearly_analysis,
            'largest_gaps': gaps[:5] if gaps else [],
            'consumption_stats': feeder_data['consumption'].describe().to_dict() if available_records > 0 else None
        }
    
    def apply_iqr(self, window_hours=24*30, outlier_action='cap', iqr_multiplier=1.5, min_periods=None):
        """
        Apply Interquartile Range (IQR) outlier detection and handling with rolling window approach.
        
        This method uses a rolling window to calculate dynamic IQR boundaries, making it more 
        suitable for time series data with seasonal patterns and trends.
        
        Parameters:
        -----------
        window_hours : int, default 24*7 (1 week)
            Size of the rolling window in hours for calculating IQR statistics.
            - Smaller windows (24h): More sensitive to short-term patterns, better for detecting 
              sudden anomalies but may flag normal variations as outliers
            - Larger windows (168h/1week): Better for seasonal patterns, more stable boundaries
            - Very large windows (720h/1month): Most stable, good for long-term trends
            
        outlier_action : str, default 'cap'
            Action to take when outliers are detected:
            - 'cap': Replace outliers with the boundary values (Q1-1.5*IQR or Q3+1.5*IQR)
            - 'remove': Remove outlier records entirely (sets to NaN)
            - 'transform': Apply log transformation to reduce impact of extreme values
            - 'flag': Only flag outliers without modification (adds 'is_outlier' column)
            
        iqr_multiplier : float, default 1.5
            Multiplier for IQR to define outlier boundaries.
            - 1.5: Standard outlier detection (Tukey's method)
            - 3.0: More conservative, only extreme outliers
            - 1.0: More aggressive outlier detection
            
        min_periods : int, optional
            Minimum number of observations required to calculate rolling statistics.
            If None, uses window_hours // 2
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with processed consumption data including outlier information
            
        Outlier Handling Justifications:
        --------------------------------
        
        1. CAPPING (Recommended for most cases):
           - Preserves data completeness for time series analysis
           - Maintains temporal relationships
           - Reduces impact of measurement errors and anomalies
           - Suitable for forecasting and trend analysis
           
        2. REMOVAL:
           - Use when outliers represent clear data quality issues
           - Not recommended for sparse data or when maintaining data continuity is important
           - Can create gaps that affect time series modeling
           
        3. TRANSFORMATION:
           - Useful when data has natural heavy-tailed distribution
           - Preserves relative relationships while reducing extreme values
           - Good for data with exponential growth patterns
           
        4. FLAGGING ONLY:
           - Use for exploratory analysis to understand outlier patterns
           - When domain expertise is needed to decide on outlier handling
           - For data quality assessment
        
        Rolling Window Approach Rationale:
        ---------------------------------
        - Electricity consumption has daily, weekly, and seasonal patterns
        - Fixed global thresholds may incorrectly flag normal seasonal peaks/lows
        - Rolling windows adapt to local data characteristics
        - Better handles non-stationary time series data
        """
        import numpy as np
        import pandas as pd
        
        if self.consumption_data is None or self.consumption_data.empty:
            print(f"No consumption data available for feeder {self.feeder_id}")
            return None
            
        # Work with a copy of the data
        df = self.consumption_data.copy()
        
        # Ensure data is sorted by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Set minimum periods if not specified
        if min_periods is None:
            min_periods = max(window_hours // 2, 24)  # At least 24 hours
            
        print(f"="*80)
        print(f"IQR OUTLIER DETECTION AND HANDLING - FEEDER {self.feeder_id}")
        print(f"="*80)
        print(f"Window size: {window_hours} hours ({window_hours/24:.1f} days)")
        print(f"IQR multiplier: {iqr_multiplier}")
        print(f"Outlier action: {outlier_action.upper()}")
        print(f"Minimum periods: {min_periods}")
        
        # Calculate rolling statistics
        print(f"\nCalculating rolling IQR statistics...")
        
        # Rolling quantiles
        rolling_q1 = df['consumption'].rolling(
            window=window_hours, 
            min_periods=min_periods,
            center=True
        ).quantile(0.25)
        
        rolling_q3 = df['consumption'].rolling(
            window=window_hours, 
            min_periods=min_periods,
            center=True
        ).quantile(0.75)
        
        # Calculate IQR and boundaries
        rolling_iqr = rolling_q3 - rolling_q1
        lower_bound = rolling_q1 - iqr_multiplier * rolling_iqr
        upper_bound = rolling_q3 + iqr_multiplier * rolling_iqr
        
        # Identify outliers
        df['is_outlier'] = (
            (df['consumption'] < lower_bound) | 
            (df['consumption'] > upper_bound)
        )
        
        # Count outliers
        total_records = len(df[df['consumption'].notna()])
        outlier_count = df['is_outlier'].sum()
        outlier_percentage = (outlier_count / total_records * 100) if total_records > 0 else 0
        
        print(f"Total non-null records: {total_records:,}")
        print(f"Outliers detected: {outlier_count:,} ({outlier_percentage:.2f}%)")
        
        # Store original values for comparison
        df['consumption_original'] = df['consumption'].copy()
        df['lower_bound'] = lower_bound
        df['upper_bound'] = upper_bound
        
        # Apply outlier handling
        if outlier_action == 'cap':
            print(f"\nCapping outliers to boundary values...")
            
            # Cap lower outliers
            lower_outliers = df['consumption'] < lower_bound
            df.loc[lower_outliers, 'consumption'] = lower_bound[lower_outliers]
            
            # Cap upper outliers  
            upper_outliers = df['consumption'] > upper_bound
            df.loc[upper_outliers, 'consumption'] = upper_bound[upper_outliers]
            
            capped_count = lower_outliers.sum() + upper_outliers.sum()
            print(f"Capped {capped_count:,} outlier values")
            
        elif outlier_action == 'remove':
            print(f"\nRemoving outlier records...")
            original_count = df['consumption'].notna().sum()
            df.loc[df['is_outlier'], 'consumption'] = np.nan
            removed_count = original_count - df['consumption'].notna().sum()
            print(f"Removed {removed_count:,} outlier values (set to NaN)")
            
        elif outlier_action == 'transform':
            print(f"\nApplying log transformation to reduce outlier impact...")
            
            # Only transform positive values
            positive_mask = df['consumption'] > 0
            
            if positive_mask.sum() > 0:
                # Add small constant to handle zero values
                df.loc[positive_mask, 'consumption'] = np.log1p(df.loc[positive_mask, 'consumption'])
                print(f"Applied log1p transformation to {positive_mask.sum():,} positive values")
            else:
                print("Warning: No positive values found for log transformation")
                
        elif outlier_action == 'flag':
            print(f"\nOutliers flagged only - no data modification applied")
            
        else:
            raise ValueError(f"Invalid outlier_action: {outlier_action}. Must be 'cap', 'remove', 'transform', or 'flag'")
        
        # Calculate statistics before and after processing
        print(f"\n" + "="*50)
        print(f"BEFORE AND AFTER COMPARISON")
        print(f"="*50)
        
        # Original data stats
        orig_stats = df['consumption_original'].describe()
        print(f"\nOriginal data statistics:")
        print(f"  Count: {orig_stats['count']:,.0f}")
        print(f"  Mean: {orig_stats['mean']:.2f}")
        print(f"  Std: {orig_stats['std']:.2f}")
        print(f"  Min: {orig_stats['min']:.2f}")
        print(f"  Max: {orig_stats['max']:.2f}")
        
        # Processed data stats
        processed_stats = df['consumption'].describe()
        print(f"\nProcessed data statistics:")
        print(f"  Count: {processed_stats['count']:,.0f}")
        print(f"  Mean: {processed_stats['mean']:.2f}")
        print(f"  Std: {processed_stats['std']:.2f}")
        print(f"  Min: {processed_stats['min']:.2f}")
        print(f"  Max: {processed_stats['max']:.2f}")
        
        # Show example outliers if any were found
        if outlier_count > 0 and outlier_action != 'flag':
            print(f"\nExample outlier adjustments (first 5):")
            outlier_examples = df[df['is_outlier']].head()
            
            for idx, row in outlier_examples.iterrows():
                direction = "LOW" if row['consumption_original'] < row['lower_bound'] else "HIGH"
                print(f"  {row['datetime'].strftime('%Y-%m-%d %H:%M')} | "
                      f"Original: {row['consumption_original']:.2f} → "
                      f"Processed: {row['consumption']:.2f} | "
                      f"Bounds: [{row['lower_bound']:.2f}, {row['upper_bound']:.2f}] | "
                      f"{direction} outlier")
        
        # Update the feeder's consumption data
        self.consumption_data = df[['datetime', 'consumption']].copy()
        
        print(f"\n✅ Outlier processing completed for feeder {self.feeder_id}")
        print(f"Updated consumption data stored in feeder object")
        print(f"="*80)
        
        # Return detailed results for analysis
        return {
            'feeder_id': self.feeder_id,
            'window_hours': window_hours,
            'iqr_multiplier': iqr_multiplier,
            'outlier_action': outlier_action,
            'total_records': total_records,
            'outliers_detected': outlier_count,
            'outlier_percentage': outlier_percentage,
            'original_stats': orig_stats.to_dict(),
            'processed_stats': processed_stats.to_dict(),
            'processed_data': df,  # Full data with all analysis columns
            'bounds_data': df[['datetime', 'lower_bound', 'upper_bound']].copy()
        }

