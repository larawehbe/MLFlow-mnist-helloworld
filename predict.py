import mlflow

model_path = r'./mlruns/0/afb9bdd769d24d76a28766746f6773dc/artifacts/model'
m = mlflow.keras.load_model(model_path)
print(type(m))