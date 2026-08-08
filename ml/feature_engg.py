from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
def normalization(df):
    num_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore",
                "CodingTestScore", "Internships"]
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df["PlacementStatus"])
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    columns = train_df.columns.tolist()

    # Convert scaled arrays back to DataFrames
    train_df = pd.DataFrame(train_scaled, columns=columns, index=train_df.index)
    test_df = pd.DataFrame(test_scaled, columns=columns, index=test_df.index)
    return train_df, test_df
