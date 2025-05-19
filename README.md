
# CIFAR-10 Image Classification with Streamlit

This project is a Streamlit web application for classifying images from the CIFAR-10 dataset using a pre-trained deep learning model.

## Features

- Upload an image to get a classification prediction.
- Supports CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- Real-time prediction with a user-friendly interface.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cifar10-classification.git
   cd cifar10-classification
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

- Open the web application.
- Upload an image from one of the CIFAR-10 classes.
- The model will predict the class and display the result.

## Model

The model is trained using a custom architecture based on convolutional neural networks (CNNs) and residual blocks.

## Troubleshooting

If you encounter an error related to loading the model, ensure that the model file `cifar10_model.pth` is placed correctly in the `scripts` directory.

## License

This project is licensed under the MIT License.
