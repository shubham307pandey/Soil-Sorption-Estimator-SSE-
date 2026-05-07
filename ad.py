import pandas as pd
import numpy as np


def std_ad(df: pd.DataFrame):
    """
    Standard AD calculation using mean and std of training data.
    """

    df = df.copy()

    mean = df.mean()
    #std = df.std().replace(0, np.nan) 
    std = df.std()

    std_df = np.abs(df - mean) / std

    sk_score = std_df.mean(axis=1) + (1.28 * std_df.std(axis=1))
    max_std = std_df.max(axis=1)
    min_std = std_df.min(axis=1)

    ad_status = np.full(len(df), "Inside AD", dtype=object)

    # Conditions
    cond1 = (max_std > 3) & (min_std > 3)
    cond2 = ((max_std > 3) & (min_std < 3)) & (sk_score > 3)
    cond3 = ((max_std < 3) & (min_std > 3)) & (sk_score > 3)

    ad_status[cond1 | cond2 | cond3] = "Outside AD"

    df["AD Status"] = ad_status

    return df

