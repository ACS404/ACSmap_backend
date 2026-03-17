## titanic.py — Titanic ML Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns

 
class TitanicModel:
    """A class used to represent the Titanic Model for passenger survival prediction."""
 
    # Singleton instance — trained once, reused for all predictions
    _instance = None
 
    def __init__(self):
        self.model = None       # Logistic Regression model
        self.dt = None          # Decision Tree model (for feature importance)
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'
        self.titanic_data = sns.load_dataset('titanic')
        self.encoder = OneHotEncoder(handle_unknown='ignore')
 
    def _clean(self):
        """Drop unused columns, encode binary fields, and one-hot encode 'embarked'."""
        td = self.titanic_data
 
        # Drop columns not used in prediction
        td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'],
                axis=1, inplace=True)
 
        # Binary encode sex and alone
        td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
        td['alone'] = td['alone'].apply(lambda x: 1 if x == True else 0)
 
        # Drop rows missing 'embarked' before encoding
        td.dropna(subset=['embarked'], inplace=True)
 
        # One-hot encode 'embarked'
        onehot = self.encoder.fit_transform(td[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols, index=td.index)
        self.titanic_data = pd.concat([td, onehot_df], axis=1)
        self.titanic_data.drop(['embarked'], axis=1, inplace=True)
 
        # Extend features list with one-hot columns
        self.features.extend(cols)
 
        # Final drop of any remaining NaN rows
        self.titanic_data.dropna(inplace=True)
 
    def _train(self):
        """Train logistic regression (primary) and decision tree (feature importance)."""
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]
 
        # Logistic Regression — primary prediction model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
 
        # Decision Tree — used only to report feature importances
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
 
    @classmethod
    def get_instance(cls):
        """
        Returns the singleton TitanicModel instance.
        Cleans and trains the model on first call; subsequent calls return cached instance.
 
        Returns:
            TitanicModel: trained singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance
 
    def predict(self, passenger):
        """
        Predict the survival probability of a passenger.
 
        Args:
            passenger (dict): passenger attributes with keys:
                - pclass (int): 1, 2, or 3
                - sex (str): 'male' or 'female'
                - age (float): passenger age in years
                - sibsp (int): siblings/spouses aboard
                - parch (int): parents/children aboard
                - fare (float): ticket fare (0–512)
                - embarked (str): 'C', 'Q', or 'S'
                - alone (bool): True if travelling alone
 
        Returns:
            dict: {'die': float, 'survive': float} — probabilities summing to 1
        """
        passenger_df = pd.DataFrame(passenger, index=[0])
 
        # Binary encode
        passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger_df['alone'] = passenger_df['alone'].apply(lambda x: 1 if x == True else 0)
 
        # One-hot encode embarked
        onehot = self.encoder.transform(passenger_df[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols, index=passenger_df.index)
        passenger_df = pd.concat([passenger_df, onehot_df], axis=1)
 
        # Drop columns not used by the model
        drop_cols = [c for c in ['embarked', 'name'] if c in passenger_df.columns]
        passenger_df.drop(drop_cols, axis=1, inplace=True)
 
        # Predict
        die, survive = np.squeeze(self.model.predict_proba(passenger_df[self.features]))
        return {'die': round(float(die), 4), 'survive': round(float(survive), 4)}
 
    def feature_weights(self):
        """
        Returns the relative importance of each feature used in prediction.
 
        Returns:
            dict: {feature_name (str): importance (float)}
        """
        importances = self.dt.feature_importances_
        return {feature: round(float(imp), 4)
                for feature, imp in zip(self.features, importances)}
 
 
def initTitanic():
    """Load and train the TitanicModel singleton. Call at app startup."""
    TitanicModel.get_instance()
 
 
def testTitanic():
    """Quick smoke-test for the TitanicModel."""
    passenger = {
        'name': 'Test Passenger',
        'pclass': 2,
        'sex': 'male',
        'age': 30,
        'sibsp': 0,
        'parch': 0,
        'fare': 16.00,
        'embarked': 'S',
        'alone': True
    }
    model = TitanicModel.get_instance()
    prob = model.predict(passenger)
    print(f"Survival probability: {prob['survive']:.2%}")
    print(f"Feature weights: {model.feature_weights()}")
 
 
if __name__ == "__main__":
    testTitanic()
 