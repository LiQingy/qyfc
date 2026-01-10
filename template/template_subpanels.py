import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 5))

gs = GridSpec(
    nrows=2, ncols=2,
    height_ratios=[2, 1],  # 上:下 = 3:1
    hspace=0.0
)

ax10 = fig.add_subplot(gs[0])
ax11 = fig.add_subplot(gs[2], sharex=ax10)
ax20 = fig.add_subplot(gs[1])
ax21 = fig.add_subplot(gs[3], sharex=ax20)

ax10.plot(0, 0)
ax11.plot(0, 0)

ax20.plot(0, 0)
ax21.plot(0, 0)

ax2.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
ax1.set_ylabel(r"$P(k)$")
ax2.set_ylabel("res")

plt.tight_layout()
plt.show()
