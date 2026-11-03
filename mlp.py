"""
Rede neural MLP implementada do zero (forward + backprop + gradient descent).
Usa apenas NumPy para operações matriciais — sem frameworks de ML.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivada da sigmoide em função da saída já activada (a = sigmoid(z))."""
    return a * (1.0 - a)


class MLP:
    """Perceptron multicamada com camadas totalmente conectadas e sigmoide."""

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        """
        layer_sizes: [n_entrada, n_oculta1, ..., n_saida]
        Para classificação binária, n_saida = 1.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes precisa de pelo menos entrada e saída")
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Inicialização Xavier / Glorot
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

        self._activations: list[np.ndarray] = []
        self._pre_activations: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n_amostras, n_entrada)"""
        self._activations = [x]
        self._pre_activations = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            self._pre_activations.append(z)
            if i == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = sigmoid(z)
            self._activations.append(a)
        return a

    def backward(
        self,
        y_true: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """
        Retropropagação com entropia cruzada binária.
        y_true: (n_amostras, 1) valores 0 ou 1.
        Retorna loss médio ponderado.
        """
        n = y_true.shape[0]
        if sample_weight is None:
            sample_weight = np.ones((n, 1))
        y_pred = self._activations[-1]
        eps = 1e-9
        loss = -np.mean(
            sample_weight
            * (
                y_true * np.log(y_pred + eps)
                + (1 - y_true) * np.log(1 - y_pred + eps)
            )
        )

        # Gradiente na saída (BCE + sigmoide)
        delta = (y_pred - y_true) * sample_weight / n

        grad_w: list[np.ndarray] = []
        grad_b: list[np.ndarray] = []

        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = self._activations[i]
            grad_w.insert(0, a_prev.T @ delta)
            grad_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i > 0:
                delta = (delta @ self.weights[i].T) * sigmoid_derivative(
                    self._activations[i]
                )

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

        return float(loss)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        class_weight: tuple[float, float] | None = None,
        verbose: bool = False,
    ) -> list[float]:
        """Treina a rede; retorna histórico de loss por época."""
        y = y.reshape(-1, 1)
        if class_weight is not None:
            w0, w1 = class_weight
            sw = np.where(y > 0.5, w1, w0)
            sw = sw.astype(float)
        else:
            sw = None

        history: list[float] = []
        for epoch in range(epochs):
            self.forward(x)
            if sw is not None:
                loss = self.backward(y, sample_weight=sw.reshape(-1, 1))
            else:
                loss = self.backward(y)
            history.append(loss)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  época {epoch + 1}/{epochs} — loss: {loss:.4f}")
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(x)
        return (proba >= threshold).astype(int).ravel()
