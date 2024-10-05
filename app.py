import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create an uploads directory
UPLOAD_FOLDER = '/tmp/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the SmoothAttention module
class SmoothAttention(nn.Module):
    def __init__(self, in_channels, out_channels, threshold=0.4):
        super(SmoothAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.threshold = threshold

    def forward(self, x):
        batch_size, C, H, W = x.size()

        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        attention_reshaped = attention.view(batch_size, H, W, H * W)
        attention_padded = F.pad(attention_reshaped, (0, 0, 1, 1, 1, 1), mode='replicate')

        chebyshev_distances = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                neighbor = attention_padded[:, i:i + H, j:j + W, :]
                distance = torch.max(torch.abs(neighbor - attention_reshaped), dim=-1)[0]
                chebyshev_distances.append(distance)

        chebyshev_distances = torch.stack(chebyshev_distances, dim=-1)
        max_chebyshev_distance = torch.max(chebyshev_distances, dim=-1)[0]

        smoothing_mask = (max_chebyshev_distance > self.threshold).float()

        smoothed_attention = torch.stack([
            attention_padded[:, i:i + H, j:j + W, :]
            for i in range(3) for j in range(3)
            if not (i == 1 and j == 1)
        ], dim=0).mean(dim=0)

        smoothing_mask = smoothing_mask.unsqueeze(-1).expand_as(attention_reshaped)

        final_attention = (1 - smoothing_mask) * attention_reshaped + smoothing_mask * smoothed_attention

        final_attention = final_attention.view(batch_size, H * W, H * W)

        proj_value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, final_attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        out = self.gamma * out + x
        return out

# Define the SmoothAttentionUNet model
class SmoothAttentionUNet(nn.Module):
    def __init__(self, num_classes):
        super(SmoothAttentionUNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.smooth_attention = SmoothAttention(512, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.smooth_attention(x)
        x = self.decoder(x)
        return x

# Load the trained model
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pth')
try:
    model = SmoothAttentionUNet(num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    model = None

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def tensor_to_image(tensor):
    image = tensor.squeeze().cpu().numpy()
    image = (image * 255).astype('uint8')
    return Image.fromarray(image)

@app.route('/segment', methods=['POST'])
def segment_image():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    logging.info("Received segmentation request")

    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"File saved at {filepath}")

        try:
            image = Image.open(filepath).convert("RGB")

            # Preprocess the image
            input_tensor = preprocess(image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.sigmoid(output) > 0.5

            # Convert the predicted mask to an image
            pred_image = tensor_to_image(pred_mask)

            # Save the image to an in-memory buffer
            img_io = io.BytesIO()
            pred_image.save(img_io, 'PNG')
            img_io.seek(0)

            # Clean up the uploaded file
            os.remove(filepath)

            return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='prediction.png')
        except Exception as e:
            logging.error(f"Error during image segmentation: {str(e)}")
            return jsonify({"error": str(e)}), 500
    else:
        logging.error("File upload failed")
        return jsonify({"error": "File upload failed"}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)