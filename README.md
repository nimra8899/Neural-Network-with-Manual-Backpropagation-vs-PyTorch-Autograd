# Neural-Network-with-Manual-Backpropagation-vs-PyTorch-Autograd
This project demonstrates the implementation of a simple neural network using:

Manual forward and backward propagation (from scratch)
PyTorch automatic differentiation (autograd)
📌 Objective

To understand how gradients are computed manually and compare them with PyTorch's built-in autograd system.

🧠 Model Architecture
Input layer: 2 features
Hidden layer: 2 neurons (linear)
Output layer: 1 neuron (linear)
⚙️ Manual Implementation
Forward pass computed using basic equations
Gradients calculated manually using chain rule
Weights updated using Gradient Descent
🤖 PyTorch Implementation
Same architecture implemented using tensors
Autograd computes gradients automatically
Loss computed using Mean Squared Error
📊 Key Learning
Understanding how backpropagation works internally
Difference between manual gradients and autograd
Importance of vectorization in deep learning
▶️ How to Run
pip install -r requirements.txt
python manual_backprop.py
python pytorch_backprop.py
📌 Output

Both implementations update weights using gradient descent. Results should be similar, verifying correctness of manual gradients.

🚀 Future Improvements
Add activation functions (ReLU, Sigmoid)
Extend to multi-layer neural networks
Train on real datasets
