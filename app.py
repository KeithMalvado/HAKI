from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model_def import SRCNN, ResNetSR  # import model yang sudah dibuat

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
srcnn_model = SRCNN().to(device)
srcnn_model.load_state_dict(torch.load('model_srcnn.pt', map_location=device))
srcnn_model.eval()

resnet_model = ResNetSR().to(device)
resnet_model.load_state_dict(torch.load('model_resnetsr.pt', map_location=device))
resnet_model.eval()

transform = transforms.Compose([transforms.ToTensor()])

def restore_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    lr = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(lr).cpu().squeeze().permute(1, 2, 0).numpy()
    sr_img = Image.fromarray((sr * 255).astype('uint8'))
    return sr_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        model_type = request.form['model']
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            if model_type == 'SRCNN':
                sr_img = restore_image(srcnn_model, img_path)
            else:
                sr_img = restore_image(resnet_model, img_path)

            result_path = os.path.join(RESULT_FOLDER, 'result_' + file.filename)
            sr_img.save(result_path)

            return render_template('index.html',
                                   uploaded_img=img_path,
                                   result_img=result_path,
                                   model_used=model_type)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
