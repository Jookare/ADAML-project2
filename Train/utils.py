import pandas as pd

def pretreat_data(df_train, df_test):
    """
    Pre-treat training and testing data by removing outliers.
    """
    print("Pre-treating data...")
    
    # Compute reference rolling statistics from the training set
    reference_stats = compute_reference_stats(df_train)
    
    # Remove outliers from training and test sets using these stats
    df_train_cleaned = remove_outliers(df_train, reference_stats)
    df_test_cleaned = remove_outliers(df_test, reference_stats)
    
    df_train_norm, train_stats = normalize_data(df_train_cleaned)
    df_test_norm, _ = normalize_data(df_test_cleaned, train_stats)
    
    
    return df_train_norm, df_test_norm, reference_stats


def compute_reference_stats(df_train):
    """
    Compute rolling mean and standard deviation over the reference period.
    """
    
    reference_stats = {}
    reference_period = df_train.iloc[: 2*365]  # First two years
    
    for col in df_train.columns[1:]:
        roll = reference_period[col].rolling(window=30, min_periods=1, center=True)
        roll_mean = roll.mean()
        roll_std = roll.std()

        mean = pd.concat([roll_mean, roll_mean, roll_mean.iloc[:118]], ignore_index=True) # Repeat for almost the whole period
        std = pd.concat([roll_std, roll_std, roll_std.iloc[:118]], ignore_index=True)

        reference_stats[col] = {
            'mean': mean,
            'std': std
        }
        
    return reference_stats


def remove_outliers(df, reference_stats, is_test=False):
    """
    Replace outliers with rolling mean based on reference statistics.
    """
    
    for col in df.columns[1:]:
        whole_period = df[col].copy()
        ref_mean = reference_stats[col]['mean']
        ref_std = reference_stats[col]['std']
     
        # Calculate z-score and remove outliers
        z_score = (whole_period - ref_mean) / ref_std
        out_bool = z_score.abs() > 4
        whole_period[out_bool] = ref_mean[out_bool]
        
        df[col] = whole_period

    return df


def normalize_data(df, stats = None):
    """
    Normalizes the data to zero mean and unit variance
    """
    
    if stats is None:
        stats = {}  # Initialize if no stats provided

    normalized_df = df.copy()  # Avoid modifying the original DataFrame
    for col in df.columns[1:]:
        data = df[col]
        
        if col in stats:  # Use provided stats if available
            mean, std = stats[col]['mean'], stats[col]['std']
            
        else:  # Compute stats if not provided
            mean, std = data.mean(), data.std()
            stats[col] = {'mean': mean, 'std': std}

        # Apply normalization
        normalized_df[col] = (data - mean) / std

    return normalized_df, stats


def rolling_train_valid_split(df, months=2, window_size=1, horizon=1):
    """
    Creates rolling train-validation splits for time-series data.
    """
    # Ensure the dataframe is sorted by date
    df = df.sort_values('date')
    
    # Get start and end dates
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    
    # Loop through each year in the data
    for year in range(start_date.year, end_date.year):
        # Define training range: from January 1st of the current year to December 31st of the same year
        train_start_date = pd.Timestamp(f"{year}-01-01")
        train_end_date = pd.Timestamp(f"{year}-12-31")
        
        # Select training data for the current year
        train = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)]
        
        # Define validation range
        valid_start_date = train_end_date - pd.Timedelta(days=window_size+horizon)  # Last month of the year
        valid_end_date = train_end_date + pd.DateOffset(months=months)
        
        # Select validation data
        valid = df[(df['date'] >= valid_start_date) & (df['date'] <= valid_end_date)]
        
        # Stop if there isn't enough validation data
        if valid.empty or (len(valid) < (10 + window_size + horizon)):
            valid = None
        
        yield train, valid
        
        
