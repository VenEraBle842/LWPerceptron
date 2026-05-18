import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from perceptron import Perceptron
from utils import plot_decision_boundary, generate_gaussian_clouds, generate_xor, generate_circle, custom_cv

def main():
    os.makedirs('results', exist_ok=True)
    print("=== ЛАБОРАТОРНАЯ РАБОТА: ОДНОСЛОЙНЫЙ ПЕРЦЕПТРОН ===\n")

    print("[Шаг 1] Подготовка данных...")
    X, y = make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_informative=2, 
        random_state=42, n_clusters_per_class=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[Шаг 2-3] Обучение базовой модели...")
    model_base = Perceptron(n_features=2, loss='bce', init_type='small')
    model_base.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    print(f"Train Accuracy: {accuracy_score(y_train, model_base.predict(X_train)):.4f}")
    print(f"Test Accuracy:  {accuracy_score(y_test, model_base.predict(X_test)):.4f}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(model_base.history['train_loss'], label='Train Loss')
    ax1.plot(model_base.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Binary Cross-Entropy')
    ax1.set_title('Кривая обучения (Loss)')
    ax1.legend()

    plot_decision_boundary(X_train, y_train, model_base, ax2, "Разделяющая граница (Train Data)")
    plt.tight_layout()
    plt.savefig('results/basic_training.png')
    print("Сохранен график: results/basic_training.png\n")
    plt.close()

    print("[Шаг 4] Проведение экспериментов...")
    experiments_records = []

    # Learning Rate
    for lr in [0.001, 0.01, 0.5, 1.0]:
        p = Perceptron(n_features=2)
        p.fit(X_train, y_train, X_test, y_test, epochs=50, lr=lr, batch_size=32)
        acc = accuracy_score(y_test, p.predict(X_test))
        experiments_records.append({'Type': 'LR', 'Value': lr, 'Test Acc': acc, 'Final Val Loss': p.history['val_loss'][-1]})

    # Batch Size
    for bs in [1, 16, 64, 256]:
        p = Perceptron(n_features=2)
        p.fit(X_train, y_train, X_test, y_test, epochs=50, lr=0.1, batch_size=bs)
        acc = accuracy_score(y_test, p.predict(X_test))
        experiments_records.append({'Type': 'Batch Size', 'Value': bs, 'Test Acc': acc, 'Final Val Loss': p.history['val_loss'][-1]})

    # Initialization
    init_histories = {}
    for init in ['zeros', 'small', 'large']:
        p = Perceptron(n_features=2, init_type=init)
        p.fit(X_train, y_train, X_test, y_test, epochs=50, lr=0.1, batch_size=32)
        acc = accuracy_score(y_test, p.predict(X_test))
        experiments_records.append({'Type': 'Init', 'Value': init, 'Test Acc': acc, 'Final Val Loss': p.history['val_loss'][-1]})
        init_histories[init] = p.history['val_loss']

    df_results = pd.DataFrame(experiments_records)
    print("Результаты экспериментов:")
    print(df_results.to_string(), "\n")

    df_results.to_csv('results/experiments_results.csv', index=False)
    print("Таблица сохранена в файл: results/experiments_results.csv\n")

    plt.figure(figsize=(8, 4))
    for init, hist in init_histories.items():
        plt.plot(hist, label=f'Init: {init}')
    plt.xlabel('Эпоха')
    plt.ylabel('Validation Loss')
    plt.title('Влияние инициализации весов на сходимость')
    plt.legend()
    plt.savefig('results/init_experiment.png')
    print("Сохранен график: results/init_experiment.png\n")
    plt.close()

    print("=== ДОПОЛНИТЕЛЬНЫЕ ЗАДАНИЯ ===\n")

    # 1. Gaussian Clouds и XOR & Circle Datasets
    print("[Доп 1.1] Обучение на Гауссовых облаках...")
    X_gauss, y_gauss = generate_gaussian_clouds(500, noise=0.05)
    p_gauss = Perceptron(n_features=2)
    p_gauss.fit(X_gauss, y_gauss, X_gauss, y_gauss, epochs=100, lr=0.1, batch_size=32)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_decision_boundary(X_gauss, y_gauss, p_gauss, ax, f"Обучение на Гауссовых облаках (Acc: {accuracy_score(y_gauss, p_gauss.predict(X_gauss)):.2f})")
    plt.savefig('results/gauss_experiment.png')
    print("Сохранен график: results/gauss_experiment.png\n")
    plt.close()

    print("[Доп 1.2] Обучение на XOR данных...")
    X_xor, y_xor = generate_xor(500, noise=0.0)
    p_xor = Perceptron(n_features=2)
    p_xor.fit(X_xor, y_xor, X_xor, y_xor, epochs=100, lr=0.1, batch_size=32)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_decision_boundary(X_xor, y_xor, p_xor, ax, f"Обучение на XOR (Acc: {accuracy_score(y_xor, p_xor.predict(X_xor)):.2f})")
    plt.savefig('results/xor_experiment.png')
    print("Сохранен график: results/xor_experiment.png\n")
    plt.close()

    print("[Доп 1.3] Обучение на данных в виде окружностей...")
    X_circle, y_circle = generate_circle(n_samples=500, noise=0.05)
    p_circle = Perceptron(n_features=2)
    p_circle.fit(X_circle, y_circle, X_circle, y_circle, epochs=100, lr=0.1, batch_size=32)

    fig, ax = plt.subplots(figsize=(6, 5))
    acc_circle = accuracy_score(y_circle, p_circle.predict(X_circle))
    plot_decision_boundary(X_circle, y_circle, p_circle, ax, f"Обучение на окружности (Acc: {acc_circle:.2f})")
    plt.savefig('results/circle_experiment.png')
    print("Сохранен график: results/circle_experiment.png\n")
    plt.close()

    # 2. Hinge Loss и L2
    print("[Доп 2] Сравнение функций потерь и регуляризации...")
    p_hinge = Perceptron(n_features=2, loss='hinge')
    p_hinge.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.05, batch_size=32)
    acc_bce = accuracy_score(y_test, model_base.predict(X_test))
    acc_hinge = accuracy_score(y_test, p_hinge.predict(X_test))
    print(f"Accuracy (BCE):   {acc_bce:.4f}")
    print(f"Accuracy (Hinge): {acc_hinge:.4f}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(model_base.history['val_loss'], label='BCE Val Loss', color='blue')
    ax1.set_title('Сходимость BCE Loss')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(p_hinge.history['val_loss'], label='Hinge Val Loss', color='purple')
    ax2.set_title('Сходимость Hinge Loss')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/loss_comparison.png')
    print("Сохранен график: results/loss_comparison.png\n")
    plt.close()

    print("Исследование влияния коэффициента L2-регуляризации...")
    l2_lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    weight_norms = []
    test_accs = []
    
    for lam in l2_lambdas:
        p_reg = Perceptron(n_features=2, loss='bce', l2_lambda=lam)
        p_reg.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        weight_norms.append(np.linalg.norm(p_reg.w))
        test_accs.append(accuracy_score(y_test, p_reg.predict(X_test)))
        if lam in [0.0, 1.0]:
            print(f"Lambda {lam}: Норма весов = {weight_norms[-1]:.4f}, Test Acc = {test_accs[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x_labels = [str(l) for l in l2_lambdas]
    
    ax1.plot(x_labels, weight_norms, marker='o', color='green')
    ax1.set_title('Влияние L2 на норму весов')
    ax1.set_xlabel('Коэффициент регуляризации (lambda)')
    ax1.set_ylabel('L2 Норма вектора весов')
    ax1.grid(True)
    
    ax2.plot(x_labels, test_accs, marker='s', color='red')
    ax2.set_title('Влияние L2 на качество обобщения')
    ax2.set_xlabel('Коэффициент регуляризации (lambda)')
    ax2.set_ylabel('Test Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/l2_regularization_effect.png')
    print("\nСохранен график: results/l2_regularization_effect.png\n")
    plt.close()

    # 3. Метрики и ошибки
    print("[Доп 3] Расчет метрик и визуализация ошибок...")
    preds = model_base.predict(X_test)
    probs = model_base.forward(X_test)

    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1-score:  {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    errors_mask = (preds != y_test)
    ax1.scatter(X_test[~errors_mask, 0], X_test[~errors_mask, 1], c=y_test[~errors_mask], cmap='coolwarm', alpha=0.5, label='Верно')
    ax1.scatter(X_test[errors_mask, 0], X_test[errors_mask, 1], color='red', marker='X', s=150, label='Ошибки модели')
    ax1.set_title('Визуализация ошибок на Test')
    ax1.legend()

    fpr, tpr, _ = roc_curve(y_test, probs)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, probs):.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('results/metrics_and_errors.png')
    print("Сохранен график: results/metrics_and_errors.png\n")
    plt.close()

    # 4. Momentum SGD
    print("[Доп 4] Тестирование Momentum SGD...")
    momentums = [0.0, 0.5, 0.9, 0.99]
    plt.figure(figsize=(8, 5))
    for beta in momentums:
        p = Perceptron(n_features=2)
        p.fit(X_train, y_train, X_test, y_test, epochs=30, lr=0.05, batch_size=16, momentum=beta)
        plt.plot(p.history['val_loss'], label=f'Beta = {beta}')

    plt.title("Влияние параметра Momentum на сходимость")
    plt.xlabel("Эпоха")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig('results/momentum_experiment.png')
    print("Сохранен график: results/momentum_experiment.png\n")
    plt.close()

    # 5. Кросс-валидация
    print("[Доп 5] Кросс-валидация...")
    best_p = custom_cv(X_train, y_train)
    print(f"\nЛучшие гиперпараметры CV: {best_p}")

    print("\n=== РАБОТА УСПЕШНО ЗАВЕРШЕНА ===")

if __name__ == "__main__":
    main()
