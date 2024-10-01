import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize individual models
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
gb_model = GradientBoostingClassifier(n_estimators=100)

# Create a voting classifier with the individual models
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('ada', ada_model),
        ('gb', gb_model)
    ],
    voting='hard'  # 'hard' for majority voting, 'soft' for averaging probabilities
)

# Train the voting classifier
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Voting Classifier Accuracy: {accuracy:.2f}')
