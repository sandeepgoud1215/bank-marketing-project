from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def save_model(model, filename='model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def save_object(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

