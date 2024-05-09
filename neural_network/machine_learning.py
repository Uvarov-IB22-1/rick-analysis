import random
import numpy as np
from config import THREAT, RISK, NEURONS, ALPHA, EPOCHS, BATCH
from config import rectified_linear_activation, softmax, prediction, softmax_batched, cross_entropy, \
    cross_entropy_batched, one_hot_encoding, one_hot_encoding_batched, rectified_linear_activation_derivative, calculate_accuracy, TEST_DATASET


WEIGHTS_1 = np.random.rand(THREAT, NEURONS)
BIAS_1 = np.random.rand(1, NEURONS)
WEIGHTS_2 = np.random.rand(NEURONS, RISK)
BIAS_2 = np.random.rand(1, RISK)


WEIGHTS_1 = (WEIGHTS_1 - 0.5) * 2 * np.sqrt(1 / THREAT)
BIAS_1 = (BIAS_1 - 0.5) * 2 * np.sqrt(1 / THREAT)
WEIGHTS_2 = (WEIGHTS_2 - 0.5) * 2 * np.sqrt(1 / NEURONS)
BIAS_2 = (BIAS_2 - 0.5) * 2 * np.sqrt(1 / NEURONS)


for _ in range(EPOCHS):

    random.shuffle(TEST_DATASET)
    lenght = len(TEST_DATASET)

    for iteration in range(lenght // BATCH):

        x_batched, target_batched = zip(*TEST_DATASET[BATCH * iteration: BATCH * (iteration + 1)])
        x = np.concatenate(x_batched, axis=0)
        target = np.array(target_batched)

        # Прямое распространение
        t_1 = x @ WEIGHTS_1 + BIAS_1
        HIDDEN_1 = rectified_linear_activation(t_1)
        t_2 = HIDDEN_1 @ WEIGHTS_2 + BIAS_2
        output = softmax_batched(t_2)
        ERROR = np.sum(cross_entropy_batched(output, target))

        # Обратное распространение
        y_one_hot_encoding = one_hot_encoding_batched(target, RISK)
        derivative_E_dt_2 = output - y_one_hot_encoding
        derivative_E_dWEIGHTS_2 = HIDDEN_1.T @ derivative_E_dt_2
        derivative_E_dBIAS_2 = np.sum(derivative_E_dt_2, axis=0, keepdims=True)
        derivative_E_dHIDDEN_1 = derivative_E_dt_2 @ WEIGHTS_2.T
        derivative_E_dt1 = derivative_E_dHIDDEN_1 * rectified_linear_activation_derivative(t_1)
        derivative_E_dWEIGHTS_1 = x.T @ derivative_E_dt1
        derivative_E_dBIAS1 = np.sum(derivative_E_dt1, axis=0, keepdims=True)

        # Обновление весов
        WEIGHTS_1 = WEIGHTS_1 - ALPHA * derivative_E_dWEIGHTS_1
        BIAS_1 = BIAS_1 - ALPHA * derivative_E_dBIAS1
        WEIGHTS_2 = WEIGHTS_2 - ALPHA * derivative_E_dWEIGHTS_2
        BIAS_2 = BIAS_2 - ALPHA * derivative_E_dBIAS_2


f1 = open('W1', 'w')
for iteration in WEIGHTS_1.tolist():
    f1.write(' '.join(map(str, iteration)) + '\n')
f1.close()


f2 = open('W2', 'w')
for iteration in WEIGHTS_2.tolist():
    f2.write(' '.join(map(str, iteration)) + '\n')
f2.close()


f3 = open('b1', 'w')
for iteration in BIAS_1.tolist():
    f3.write(' '.join(map(str, iteration)) + '\n')
f3.close()


f4 = open('b2', 'w')
for iteration in BIAS_2.tolist():
    f4.write(' '.join(map(str, iteration)) + '\n')
f4.close()


print("Точность предсказаний:", calculate_accuracy(TEST_DATASET, WEIGHTS_1, WEIGHTS_2, BIAS_1, BIAS_2))
