from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import haversine_vectorized, compute_rmse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from memoized_property import memoized_property
from xgboost import XGBRegressor



import joblib

import mlflow
from mlflow.tracking import MlflowClient



class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train = None
        self.y_train = None
        self.experiment_name = "[BR] [RJ] [wecalderonc] Linear 1.0.0"


    def set_pipeline(self, estimator, estimator_name):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            (estimator_name, estimator)
            #('linear_model', LinearRegression())
        ])

        self.pipeline = pipeline
        return self.pipeline

    def run(self):
        '''returns a trained pipelined model'''
        # Hold out
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        #self.mlflow_log_param(param_name, param_value)
        self.mlflow_log_metric("rmse", rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    # train
    estimators = {
      "linear_model": LinearRegression(),
      "xgboost": XGBRegressor(max_depth=10, n_estimators=100, learning_rate=0.1)
    }

    # xgb_reg.fit(X_train, y_train,
    #             # evaluate loss at each iteration
    #             # eval_set=[(X_train, y_train), (X_val, y_val)],
    #             # stop iterating when eval loss increases 5 times in a row
    #             # early_stopping_rounds=5
    #             )

    # y_pred = xgb_reg.predict(X_val)


    for name, model in estimators.items():
      trainer = Trainer(X, y)

      trainer.set_pipeline(model, name)
      trainer.run()
      # evaluate
      rmse = trainer.evaluate(trainer.X_test, trainer.y_test)
      experiment_id = trainer.mlflow_experiment_id
      #save model
      trainer.save_model()

      print(
          f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
      print(rmse)



