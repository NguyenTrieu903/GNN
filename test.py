import matplotlib.pyplot as plt
feat_counts = [15, 20, 10, 25, 18, 22, 14, 28, 12,
               16, 21, 19, 17, 23, 20, 30, 24, 13, 11, 27]


plt.hist(feat_counts, bins=20)
plt.title("Number of features per graph distribution")
plt.xlabel("Number of features")
plt.ylabel("Number of graphs")
plt.show()
