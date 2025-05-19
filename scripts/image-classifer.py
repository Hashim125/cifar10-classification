import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os

# Get the current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "cifar10_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Block class as per your training script
class Block(nn.Module):
    def __init__(self, num_convs, inputs, outputs):
        super(Block, self).__init__()
        self.num_convs = num_convs
        self.Linear1 = nn.Linear(inputs, num_convs)
        self.silu = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Prepare a layer to match the dimensions for the skip connection, if necessary.
        if inputs != outputs:
            self.match_dimensions = nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outputs)
            )
        else:
            self.match_dimensions = nn.Identity()

        # Add convolutional layers
        for i in range(num_convs):
            self.add_module(f'conv{i}', nn.Conv2d(inputs, outputs, kernel_size=3, padding=1))
            self.add_module(f'batch{i}', nn.BatchNorm2d(outputs))

    def forward(self, x):
        identity = self.match_dimensions(x)
        pooled = self.pool(x).flatten(1)
        a = self.Linear1(pooled)
        a = self.silu(a)

        last_output = 0
        for i in range(self.num_convs):
            conv_out = self._modules[f'conv{i}'](x)
            bn_out = self._modules[f'batch{i}'](conv_out)
            bn_out = self.silu(bn_out)
            scaling_factor = a[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            scaling_factor = scaling_factor.expand(-1, bn_out.size(1), bn_out.size(2), bn_out.size(3))
            scaled_conv_out = bn_out * scaling_factor
            last_output = scaled_conv_out if i == 0 else last_output + scaled_conv_out
        
        out = last_output + identity
        return out

# Define the Backbone class as per your training script
class Backbone(nn.Module):
    def __init__(self, conv_arch):
        super(Backbone, self).__init__()
        inputs = 3
        self.conv_arch = conv_arch
        for i, (num_convs, outputs) in enumerate(conv_arch):
            self.add_module(f'block{i}', Block(num_convs, inputs, outputs))
            inputs = outputs
        
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(outputs * 2 * 2, 1024),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        for i in range(len(self.conv_arch)):
            x = self._modules[f'block{i}'](x)
        x = self.last(x)
        return x

# Model initialization based on your configuration
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
model = Backbone(conv_arch)

# Load the saved model
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Streamlit App UI
st.title("🚀 CIFAR-10 Image Classification")
st.markdown("""
    **Upload an image from one of the following classes:**  
    - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck  
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Preprocess the image
        input_tensor = preprocess_image(image).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        # Display the prediction at the top
        st.markdown(f"### 🎯 **Prediction:** {label}")

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

    except UnidentifiedImageError:
        st.error("❌ Error: The uploaded file is not a valid image format.")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
