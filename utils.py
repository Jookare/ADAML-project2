def remove_outliers(df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3 - q1
        fence_low  = q1 -2 *iqr
        fence_high = q3 +2 *iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        
        return df_out, fence_low, fence_high
