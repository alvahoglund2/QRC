from reservoirpy.nodes import Ridge
import numpy as np


class Qreadout:
    def __init__(self, regularization_term: float = 1e-6) -> None:
        """
        Initializes a Ridge regression object for readout of the reservoir.
        The regularization_term is used to control the amount of regularization applied to the model.
        """
        self.ridge = Ridge(ridge=regularization_term)

    def get_predictions(
        self,
        I_train: np.ndarray,
        target_train: np.ndarray,
        I_test: np.ndarray,
        warmup: int = 100,
    ) -> np.ndarray:
        """
        Fits the ridge regression model to the training data and predicts the target values for the test data.
        The warmup parameter is used to discard the initial outputs from the reservoir, dependent on the reservoir's initial state
        """
        self.ridge = self.ridge.fit(
            I_train, target_train.reshape(target_train.shape[0], 1), warmup=warmup
        )
        target_pred = self.ridge.run(I_test)
        return target_pred

    def get_readout_weights(self) -> np.ndarray:
        """
        Returns the readout weights of the ridge regression model.
        """
        return self.ridge.W
