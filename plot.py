import matplotlib.pyplot as plt
import csv

with open('entropy.data', newline='') as fp:
    data = list(csv.reader(fp))

entropy   = [float(x[0]) for x in data]
avg_len   = [float(x[1]) for x in data]
avg_len_m = [float(x[2]) for x in data]
avg_len_x = [float(x[3]) for x in data]
avg_len_n = [float(x[4]) for x in data]
avg_len_r = [float(x[5]) for x in data]

plt.style.use('seaborn-ticks')
plt.plot([0, 15], [0, 15], color="k")
plt.scatter(entropy, avg_len_x, color="c", s=1)
plt.scatter(entropy, avg_len_n, color="y", s=1)
plt.scatter(entropy, avg_len_r, color="b", s=1)
plt.scatter(entropy, avg_len_m, color="g", s=1)
plt.scatter(entropy, avg_len,   color="r", s=1)
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel("entropy")
plt.ylabel("average length")
plt.show()
