import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.parser_agent import filter_sections

# Hard-coded sample mimicking a real arXiv paper structure
SAMPLE_TEXT = """
Introduction
Deep learning has revolutionized computer vision...

Related Work
Previous approaches include LeNet and AlexNet...

Methodology
We propose a CNN with three convolutional layers.
Training was performed using SGD optimizer.
Learning rate was set to 0.01 with batch size 32.
We trained for 10 epochs on MNIST dataset.

Experiments
We evaluated our model on MNIST achieving 99.2% accuracy.
Batch size 64, epochs 10, learning rate 0.01.

Conclusion
In this paper we proposed...

References
[1] LeCun et al...
"""

result = filter_sections(SAMPLE_TEXT)
print(result)
print("---")
print("PASS" if "SGD" in result and "References" not in result else "FAIL")