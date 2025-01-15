
---

# **NAND Neural Network Implementation from Scratch**

## **Objective**
This repository contains a Python implementation of a **feedforward neural network** developed from scratch. The neural network is trained to learn the **NAND operation**, which is a fundamental logic gate used extensively in digital electronics. The focus of this project is to provide a hands-on demonstration of the core components of neural networks, including **forward pass**, **backpropagation**, and **gradient descent optimization**.

---

## **Problem Definition**
### **Dataset**
- **Inputs (X)**: Binary combinations of two inputs:
  ```plaintext
  [[0, 0], [0, 1], [1, 0], [1, 1]]
  ```
- **Outputs (y)**: Corresponding NAND operation outputs:
  ```plaintext
  [[1], [1], [1], [0]]
  ```

### **Objective**
Train a neural network to accurately predict the output of the NAND operation for any combination of two binary inputs.

---

## **Neural Network Architecture**
- **Input Layer**: 2 neurons to represent the two binary inputs.
- **Hidden Layer**: 3 neurons with **Sigmoid Activation Function**.
- **Output Layer**: 1 neuron with **Sigmoid Activation Function** for binary classification.

---

## **Methodology**
### **Forward Pass**
1. Input values are passed to the hidden layer, where the weighted sum of inputs is computed.
2. The activation function is applied to calculate the hidden layer's output.
3. The hidden layer's output is passed to the output layer, where the final weighted sum is computed and processed through the activation function to produce the prediction.

### **Backpropagation**
1. Compute the **error** between the predicted output and the actual output.
2. Propagate the error backward through the network to calculate gradients of weights and biases.
3. Update the weights and biases using **gradient descent** to minimize the error.

### **Loss Function**
- **Mean Squared Error (MSE)** is used to quantify the difference between the predicted and actual outputs:
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_{\text{pred}} - y_{\text{true}})^2
  \]

### **Optimization**
- **Gradient Descent**: Updates weights and biases in the direction of the negative gradient to minimize the loss.

---

## **Setup Instructions**
### **Clone the Repository**
```bash
git clone https://github.com/OmBhutkar/Neural_Network_From_Scratch.git
```

### **Navigate to the Project Directory**
```bash
cd Neural_Network_From_Scratch
```

### **Run the Script**
Ensure you have Python 3.x and NumPy installed. Then execute:
```bash
python neural_network.py
```

### **Output**
The script will train the neural network for 10,000 epochs and print the loss every 1000 epochs. At the end, the predictions for the NAND operation will be displayed, approximating the following results:

| Input (X)      | Predicted Output |
|-----------------|------------------|
| `[0, 0]`       | `~1`            |
| `[0, 1]`       | `~1`            |
| `[1, 0]`       | `~1`            |
| `[1, 1]`       | `~0`            |

---

## **Key Results**
The neural network successfully learns the NAND operation by training on the given binary dataset. It demonstrates the use of basic neural network components for a simple binary classification task.

---

## **Technologies Used**
- **Python 3.x**
- **NumPy**: For matrix operations and gradient calculations.

---

## **File Structure**
- `neural_network.py`: Python script containing the implementation of the feedforward neural network for the NAND operation.
- `README.md`: Detailed documentation of the project.

---

## **Author**
[Om Bhutkar](https://github.com/OmBhutkar)

---

## **Acknowledgments**
This implementation is inspired by fundamental concepts in machine learning and neural networks. It serves as an educational project to deepen the understanding of how neural networks work under the hood, without relying on external deep learning libraries.

---
