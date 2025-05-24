#state 0 : xt = 2 + e_t

#state 1 : xt = 1 + e_t

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import norm

def generateData(n=1000):
    
    observedstates = np.zeros(n+1)
    hiddenstates = np.zeros(n+1, dtype=int)
    
    hiddenmarkov = np.array([[0.75, 0.25],[0.3, 0.7]])

    observedstates[0] = np.random.normal(1, .25)

    for i in range(1,n+1):
        curr_state = hiddenstates[i-1]
        hiddenstates[i] = np.random.choice([0, 1], p=hiddenmarkov[curr_state])
        observedstates[i] = 0.99 * observedstates[i-1] + (np.random.normal(1, .1) if hiddenstates[i] else np.random.normal(1, 20))

    print(hiddenstates)
    print(observedstates)
    #samples = np.random.randn(1000)
    return hiddenstates, observedstates

def plotstates(signal, labels):
    # Create a color map for labels
    unique_labels = np.unique(labels)
    label_to_color = {label: color for label, color in zip(unique_labels, plt.cm.tab10.colors)}

    # Build line segments
    points = np.array([np.arange(len(signal)), signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Map a color to each segment based on the label at the start of the segment
    segment_labels = labels[:-1]  # One label per segment
    colors = [label_to_color[label] for label in segment_labels]

    # Create the LineCollection
    lc = LineCollection(segments, colors=colors, linewidths=2)

    fig, ax = plt.subplots()
    ax.add_collection(lc)

    ax.set_xlim(0, len(signal))
    ax.set_ylim(signal.min() - 0.5, signal.max() + 0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Signal')
    ax.set_title('Continuous Signal Colored by Labels')

    # Optional: Create a manual legend
    for label, color in label_to_color.items():
        ax.plot([], [], color=color, label=label)
    ax.legend()

    plt.show()

hidden, observed = generateData(500)
plotstates(observed, hidden)

plt.plot(observed)

##learning the parameters