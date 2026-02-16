import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================
# 1. Load Data
# =============================

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# =============================
# 2. Merge Datasets
# =============================

data = deliveries.merge(matches[['id','winner']],
                        left_on='match_id',
                        right_on='id')

# =============================
# 3. Create Target Column
# =============================

data['win'] = (data['batting_team'] == data['winner']).astype(int)

# =============================
# 4. Keep Only 2nd Innings
# =============================

second_innings = data[data['inning'] == 2].copy()

# =============================
# 5. Calculate Target Runs
# =============================

total_runs = second_innings.groupby('match_id')['total_runs'].sum().reset_index()
total_runs.rename(columns={'total_runs':'target_runs'}, inplace=True)

second_innings = second_innings.merge(total_runs, on='match_id')

# =============================
# 6. Feature Engineering
# =============================

# Runs left
second_innings['runs_left'] = second_innings['target_runs'] - second_innings.groupby('match_id')['total_runs'].cumsum()

# Balls left
second_innings['balls_left'] = 120 - ((second_innings['over'] - 1) * 6 + second_innings['ball'])

# Wickets left
second_innings['wickets_left'] = 10 - second_innings.groupby('match_id')['is_wicket'].cumsum()

# =============================
# 7. Prepare Model Data
# =============================

model_data = second_innings[['runs_left','balls_left','wickets_left','win']]

# Remove invalid rows
model_data = model_data[(model_data['runs_left'] >= 0) & (model_data['balls_left'] > 0)]

X = model_data[['runs_left','balls_left','wickets_left']]
y = model_data['win']

# =============================
# 8. Scale Features
# =============================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =============================
# 9. Train Model
# =============================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy


git init


