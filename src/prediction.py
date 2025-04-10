import pickle

def load_model(filename='model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, X):
    return model.predict(X)
