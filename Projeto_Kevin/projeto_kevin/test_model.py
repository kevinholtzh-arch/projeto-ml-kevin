import joblib

def test_model():
    model = joblib.load("modelo_Kevin_v3.pkl")
    
    sample = [[1]*30]
    prediction = model.predict(sample)
    
    assert prediction[0] in [0, 1]