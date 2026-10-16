
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


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


def data_cleaning(df):
    # missing value imputation for workshops using mean
    df["Workshops"] = df["Workshops"].fillna(df["Workshops"].mean())
    df["AptitudeTestScore"] = df["AptitudeTestScore"].fillna(df["AptitudeTestScore"].median())

    imputer = KNNImputer(n_neighbors=3)

    numerical_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore",
                      "CodingTestScore", "Internships"]

    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])



def simple_linear_regression(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    x=df[['SGPA_Sem1']]
    y=df['PlacementStatus']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    linear_regression = LinearRegression()
    linear_regression.fit(train_x, train_y)
    y_pred = linear_regression.predict(test_x)
    msel = mean_squared_error(test_y, y_pred)
    rsql = r2_score(test_y, y_pred)
    print("MSEL",msel)
    print("RSquare Score",rsql)


df = pd.read_csv(r"C:\Users\gsree\Downloads\Telegram Desktop\placement_predict_50k Dataset.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())

#data_cleaning(df)

print(df.isnull().sum())
simple_linear_regression(df)

def gradient_descent(x, y, learning_rate=0.01, iterations=100):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    m = len(y)

    theta0 = 0.0 #bias
    theta1 = 0.0  #variance

    mse_history = []

    for i in range(iterations):
        y_pred = theta0 + theta1 * x

        error = y_pred - y

        mse = np.mean(error ** 2)
        mse_history.append(mse)

        d_theta0 = (2 / m) * np.sum(error)
        d_theta1 = (2 / m) * np.sum(error * x)

        theta0 = theta0 - learning_rate * d_theta0
        theta1 = theta1 - learning_rate * d_theta1

    return theta0, theta1, mse_history

def simple_linear_regression_GD(df):
    from sklearn.metrics import mean_squared_error, r2_score
    x=df[['SGPA_Sem3']]
    y=df['PlacementStatus']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
   # linear_regression = LinearRegression()
    #linear_regression.fit(train_x, train_y)
    theta0, theta1, mse_history = gradient_descent(
        train_x, train_y, learning_rate=0.01,iterations=100)
    y_pred = theta0 + theta1 * test_x
    msel = mean_squared_error(test_y, y_pred)
    rsql = r2_score(test_y, y_pred)
    print("MSEL",msel)
    print("RSquare Score",rsql)


simple_linear_regression_GD(df)

def simple_logistic_regression(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    x = df[['CGPA']]
    y = df['PlacementStatus']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42 ,stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_x, train_y)
    theta0 = model.intercept_[0]
    print("Theta0:", theta0)
    print("Accuracy:", accuracy_score(test_y, model.predict(test_x)))

simple_logistic_regression(df)
def decision_tree(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    x = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return dt
decision_tree(df)

def  decision_tree_pruned(df):
    from sklearn.metrics import accuracy_score,classification_report
    x = df[["CGPA","Internships","AptitudeTestScore"]]
    y = df["PlacementStatus"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(criterion="gini", max_depth=4,
                                   min_samples_split=10,min_samples_leaf=5,
                                   random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Pruned Decision Tree Accuracy:",   accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model
decision_tree_pruned(df)
def random_forest_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=5,
        random_state=42
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    return model
def compare(sample_df, m1, m2, m3):
    X_sample = sample_df[["CGPA", "Internships", "AptitudeTestScore"]]
    y_actual = sample_df["PlacementStatus"]

    pred_dt = m1.predict(X_sample)
    pred_pruned = m2.predict(X_sample)
    pred_rf = m3.predict(X_sample)

    results = pd.DataFrame({
        "Student": sample_df.index,
        "Actual": y_actual.values,
        "Decision Tree": pred_dt,
        "Pruned Decision Tree": pred_pruned,
        "Random Forest": pred_rf
    })
    print("\nPrediction Comparison for 5 Random Students")
    print("="* 80)
    print(results.to_string(index=False))

m1 = decision_tree(df)
m2 = decision_tree_pruned(df)
m3 = random_forest_model(df)
sample_df = df.sample(n=10, random_state=42)
compare(sample_df, m1, m2, m3)
