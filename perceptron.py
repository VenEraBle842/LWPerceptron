import numpy as np

class Perceptron:
    def __init__(self, n_features: int, loss: str = 'bce', l2_lambda: float = 0.0, init_type: str = 'small'):
        self.n_features = n_features
        self.loss_type = loss
        self.l2_lambda = l2_lambda
        self.history = {'train_loss': [], 'val_loss': []}
        
        # Инициализация весов
        if init_type == 'zeros':
            self.w = np.zeros(n_features)
        elif init_type == 'small':
            self.w = np.random.randn(n_features) * 0.5
        elif init_type == 'large':
            self.w = np.random.randn(n_features) * 10.0
        else:
            raise ValueError("Неизвестный тип инициализации")
            
        self.b = 0.0
        
        # Переменные для Momentum SGD
        self.v_w = np.zeros(n_features)
        self.v_b = 0.0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        # np.clip предотвращает переполнение экспоненты (OverflowWarning)
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def compute_loss(self, X: np.ndarray, y_true: np.ndarray) -> float:
        z = np.dot(X, self.w) + self.b
        
        if self.loss_type == 'bce':
            y_pred = self.sigmoid(z)
            eps = 1e-15 # Защита от логарифма нуля
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            
        elif self.loss_type == 'hinge':
            # Переводим метки из {0, 1} в {-1, 1} для Hinge loss
            y_true_hinge = np.where(y_true <= 0, -1, 1)
            loss = np.mean(np.maximum(0, 1 - y_true_hinge * z))
        else:
            raise ValueError(f"Неизвестная функция потерь: {self.loss_type}")
            
        # L2-регуляризация (штраф за большие веса)
        l2_reg = (self.l2_lambda / 2) * np.sum(self.w ** 2)
        return loss + l2_reg

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray, 
            epochs: int, lr: float, batch_size: int, momentum: float = 0.0):
        
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Перемешивание перед каждой эпохой (важно для SGD!)
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                z = np.dot(X_batch, self.w) + self.b
                
                # Вычисление градиентов в зависимости от функции потерь
                if self.loss_type == 'bce':
                    y_pred = self.sigmoid(z)
                    dz = y_pred - y_batch
                elif self.loss_type == 'hinge':
                    y_true_hinge = np.where(y_batch <= 0, -1, 1)
                    # Градиент Hinge loss не равен 0 только если отступ < 1
                    mask = (y_true_hinge * z) < 1
                    dz = -y_true_hinge * mask
                else:
                    raise ValueError(f"Неизвестная функция потерь: {self.loss_type}")
                
                # Матричные операции для вычисления dL/dw и dL/db
                dw = np.dot(X_batch.T, dz) / len(X_batch) + self.l2_lambda * self.w
                db = np.mean(dz)
                
                # Обновление весов с учетом Momentum
                self.v_w = momentum * self.v_w + lr * dw
                self.v_b = momentum * self.v_b + lr * db
                
                self.w -= self.v_w
                self.b -= self.v_b
                
            # Запись истории loss
            self.history['train_loss'].append(self.compute_loss(X_train, y_train))
            self.history['val_loss'].append(self.compute_loss(X_val, y_val))

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.loss_type == 'hinge':
            z = np.dot(X, self.w) + self.b
            return np.where(z > 0, 1, 0)
        return (self.forward(X) >= threshold).astype(int)
