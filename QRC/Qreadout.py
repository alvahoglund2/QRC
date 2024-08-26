from reservoirpy.nodes import Ridge


class Qreadout:
    def __init__(self, regularization_term=1e-6):
        """
        Initializes a Ridge regression object for readout of the reservoir.
        The regularization_term is used to control the amount of regularization applied to the model.
        """
        self.ridge = Ridge(ridge=regularization_term)

    def get_predictions(self, I_train, target_train, I_test, warmup=100):
        """
        Fits the ridge regression model to the training data and predicts the target values for the test data.
        The warmup parameter is used to discard the initial outputs from the reservoir, dependent on the reservoir's initial state
        """
        self.ridge = self.ridge.fit(
            I_train, target_train.reshape(target_train.shape[0], 1), warmup=warmup
        )
        target_pred = self.ridge.run(I_test)
        return target_pred

    def get_readout_weights(self):
        """
        Returns the readout weights of the ridge regression model.
        """
        return self.ridge.W
