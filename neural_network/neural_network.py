import numpy as np
from config import THREAT, RISK, NEURONS, W1, b1, W2, b2
from config import relu, softmax, predict


u = np.array([90, 80, 100, 70])
prediction = predict(u, W1, W2, b1, b2)


risk_classification = ['Низшая', 'Средняя', 'Высшая']
print('Группа риска:', risk_classification[np.argmax(prediction)])
