import random
import numpy as np
import matplotlib.pyplot as plt
from config import THREAT, RISK, NEURONS, ALPHA, EPOCHS, BATCH_SIZE
from config import relu, softmax, predict, softmax_batch, sparse_cross_entropy, \
    sparse_cross_entropy_batch, to_full, to_full_batch, relu_deriv, calc_accuracy, dataset


W1 = np.random.rand(THREAT, NEURONS)  # вес
b1 = np.random.rand(1, NEURONS)  # смещение
W2 = np.random.rand(NEURONS, RISK)
b2 = np.random.rand(1, RISK)


W1 = (W1 - 0.5) * 2 * np.sqrt(1 / THREAT)
b1 = (b1 - 0.5) * 2 * np.sqrt(1 / THREAT)
W2 = (W2 - 0.5) * 2 * np.sqrt(1 / NEURONS)
b2 = (b2 - 0.5) * 2 * np.sqrt(1 / NEURONS)


loss_arr = []
for ep in range(EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):

        batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Прямое распространение
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y))

        # Обратное распространение
        y_full = to_full_batch(y, RISK)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Обновление весов
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)


f1 = open('W1', 'w')
for i in W1.tolist():
    f1.write(' '.join(map(str,i))+'\n')
f1.close()


f2 = open('W2', 'w')
for i in W2.tolist():
    f2.write(' '.join(map(str, i))+'\n')
f2.close()


f3 = open('b1', 'w')
for i in b1.tolist():
    f3.write(' '.join(map(str, i))+'\n')
f3.close()


f4 = open('b2', 'w')
for i in b2.tolist():
    f4.write(' '.join(map(str, i))+'\n')
f4.close()


print("Accuracy:", calc_accuracy(dataset, W1, W2, b1, b2))


# plt.plot(loss_arr)
# plt.show()
