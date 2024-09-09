import torch
import pytorch_lightning as pl
import torchvision
from torchvision import  transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import os
from fastapi import FastAPI, UploadFile, File
import io


#モデルの定義
class Net(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.feature = torchvision.models.resnet18(pretrained=True)
    self.fc1 = nn.Linear(1000,2)

  def forward(self,x):
    h = self.feature(x)
    h = self.fc1(h)
    return h

#モデルのロード関数
weight_path = r'C:\Users\Junji Akiyama\Casting_Project\weight99%.pt'
# 事前に学習済みモデルをロード
model = Net()
model.load_state_dict(torch.load(weight_path))
model.eval()

#推論用前処理
predict_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FastAPIインスタンスを作成
app = FastAPI()

#推論用コード
@app.post("/casting_predict/")
async def predict_image(file: UploadFile = File(...)):
    # 画像データの読み込みと変換
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = predict_transform(image)

  # モデル推論
    with torch.no_grad():
        image = image.unsqueeze(0) #バッチサイズ1へ変換
        y = model(image) 
        y = F.softmax(y, dim=1)    #クラス確立を取得
        y = torch.argmax(y, dim=1).item() #最大確率のインデックスを取得

    # 予測結果を出力
    print(f"Predicted Class for {file_path}: {y}")
