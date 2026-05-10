import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from perceptron import Perceptron

def plot_decision_boundary(X, y, model, ax, title=""):
    """
    Визуализация точек данных и разделяющей границы перцептрона.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    Z = model.predict(grid).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    ax.set_title(title)


def generate_gaussian_clouds(n_samples=500, noise=0.05, centers=((2, 2), (-2, -2)), cov=((1.0, 0.0), (0.0, 1.0))):
    """
    Генерация гауссовых облаков вокруг заданных центров.
    По умолчанию создает линейно разделимые классы при малом шуме (единичная матрица ковариации).
    """
    n_class = n_samples // 2
    X1 = np.random.multivariate_normal(mean=centers[0], cov=cov, size=n_class)
    X2 = np.random.multivariate_normal(mean=centers[1], cov=cov, size=n_class)
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_class), np.ones(n_class)])
    
    if noise > 0:
        flip_mask = np.random.rand(n_samples) < noise
        y = np.where(flip_mask, 1 - y, y)
    return X, y


def generate_xor(n_samples=500, noise=0.05):
    """
    Генерация нелинейно разделимых данных (XOR), используя гауссовы облака.
    """
    half_samples = n_samples // 2
    
    # Верхняя половина плоскости: центры (2, 2) (класс 0) и (-2, 2) (класс 1)
    X1, y1 = generate_gaussian_clouds(half_samples, noise=0.0, centers=((2, 2), (-2, 2)))
    
    # Нижняя половина плоскости: центры (-2, -2) (класс 0) и (2, -2) (класс 1)
    X2, y2 = generate_gaussian_clouds(n_samples - half_samples, noise=0.0, centers=((-2, -2), (2, -2)))
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    # Добавляем общий шум ко всему объединенному датасету
    if noise > 0:
        flip_mask = np.random.rand(n_samples) < noise
        y = np.where(flip_mask, 1 - y, y)
        
    # Перемешиваем данные
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def custom_cv(X, y, k_folds=5):
    """
    Кастомная реализация K-Fold кросс-валидации для подбора гиперпараметров.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    learning_rates = [0.01, 0.1, 0.5]
    batch_sizes = [16, 32]
    
    best_acc = 0
    best_params = {}
    
    print(f"\nЗапуск {k_folds}-Fold Cross Validation...")
    for lr in learning_rates:
        for bs in batch_sizes:
            fold_accs = []
            for train_idx, val_idx in kf.split(X):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]
                
                model = Perceptron(n_features=X.shape[1])
                model.fit(X_tr, y_tr, X_va, y_va, epochs=30, lr=lr, batch_size=bs)
                fold_accs.append(accuracy_score(y_va, model.predict(X_va)))
                
            avg_acc = np.mean(fold_accs)
            print(f"LR: {lr}, Batch: {bs} -> Средняя точность CV: {avg_acc:.4f}")
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = {'lr': lr, 'batch_size': bs}
                
    return best_params
