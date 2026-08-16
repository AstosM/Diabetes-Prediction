from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# DATASET PATH

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "diabetes.csv")

# FEATURES AND TARGET

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=2
)

# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL

model = SVC(kernel="linear")

model.fit(X_train, y_train)

# MODEL ACCURACY

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)