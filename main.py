from utils import build_and_train, preprocessing
import mlflow
import click



@click.command()
@click.option("--epochs", help="Epochs", default=6, type=int)

def main(epochs):
    uri = 'sqlite:///test.db'
    mlflow.set_tracking_uri(uri)
    mlflow.autolog()
    with mlflow.start_run():
        ds_train,ds_test = preprocessing()
        build_and_train(ds_train,ds_test,epochs)


if __name__ == '__main__':
    main()
   