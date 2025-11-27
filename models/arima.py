import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from model_base import ModelBase, TrainData


class ModelARIMA(ModelBase):
    def __init__(self):
        pass

    def train(self, train_data: TrainData):
        train = train_data.train_y
        test = train_data.test_y

        print(f"ARIMA: Train size = {len(train)}, Test size = {len(test)}")

        aic, best_order, best_seasonal = self._search_best_AIC(
            data=train,
            p_values=[0, 1],
            d_values=[0, 1],
            q_values=[0, 1],
            P_values=[0, 1],
            D_values=[0, 1],
            Q_values=[0, 1],
            period=7,
        )

        print(f"ARIMA: Best params found: {best_order} x {best_seasonal} AIC={aic}")

        self.best_order = best_order
        self.best_seasonal = best_seasonal

        model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal)
        print(f"ARIMA: Start fit")
        results = model.fit()
        print(f"ARIMA: Fit finished")

        self.model = model
        self.results = results


    def predict(self, steps: int = 30):
        if self.results is None:
            raise RuntimeError("Model is not trained! Call train() first.")

        print(f"ARIMA: Forecasting next {steps} steps...")

        forecast_obj = self.results.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean

        return forecast
    
    def _get_AIC_for_params(
            self, data: list,
            p: int, d: int, q: int,
            P: int, D: int, Q: int,
            period: int):
        """
        Returns AIC of model SARIMA(p,d,q)(P,D,Q)[period].
        """
    
        try:
            model = sm.tsa.statespace.SARIMAX(
                data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            result = model.fit(disp=False)
            return result.aic

        except Exception:
            return float("inf")

    def _search_best_AIC(
            self, data: list,
            p_values, d_values, q_values,
            P_values, D_values, Q_values,
            period: int):
        """
        Returns:
            best_aic, (p,d,q), (P,D,Q,period)
        """

        smallest_aic = float("inf")
        best_order = None
        best_seasonal = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                aic = self._get_AIC_for_params(
                                    data,
                                    p, d, q,
                                    P, D, Q,
                                    period
                                )

                                if aic < smallest_aic:
                                    smallest_aic = aic
                                    best_order = (p, d, q)
                                    best_seasonal = (P, D, Q, period)

        return smallest_aic, best_order, best_seasonal
