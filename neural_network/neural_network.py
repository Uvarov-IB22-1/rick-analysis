import numpy as np
from config import THREAT, RISK, NEURONS, WEIGHTS_1, BIAS_1, WEIGHTS_2, BIAS_2
from config import rectified_linear_activation, softmax, prediction


u = np.array([90, 80, 100, 70])
prediction = prediction(u, WEIGHTS_1, WEIGHTS_2, BIAS_1, BIAS_2)


risk_classification = ['Низшая', 'Средняя', 'Высшая']
print('Группа риска:', risk_classification[np.argmax(prediction)])
