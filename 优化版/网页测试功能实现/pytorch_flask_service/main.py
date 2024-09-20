import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model_v3 import mobilenet_v3_small

app = Flask(__name__) # 实例化Flask
CORS(app)  # 解决跨域问题

weights_path = "./MB-v3-31-state_dict.pth" # 读取权重文件
class_json_path = "./class_indices.json" # 类别标签文件
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
model = mobilenet_v3_small(num_classes=4).to(device)
# load model weights
model.load_state_dict(torch.load(weights_path, map_location=device))

model.eval() # 只是测试，所以使用验证模式

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)) # 传入的是字节数据，所以以字节等方式读取
    if image.mode != "RGB": # 判断是否为RGB图像
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device) # 新增一个batch维度并转移到device设备


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0) # 将预测值转化为概率分布
        prediction = outputs.detach().cpu().numpy() # detach方法去除梯度信息，转移到CPU并转化为numpy格式。
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




