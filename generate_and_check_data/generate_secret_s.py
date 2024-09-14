import csv
import numpy as np

with open('../s5/s.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(10):
        writer.writerow(np.random.randint(-2, 3, size=1024).astype(object))
