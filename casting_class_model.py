#ライブラリインポート
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import  transforms , datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import os
import torchmetrics
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from torchsummary import summary
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import glob

#前処理 ※学習用データ拡張
#データ拡張の設定＋Resnet18に合わせた画像の準備
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.9), #50%の確率で水平反転
    transforms.RandomVerticalFlip(p=0.9),#50%の確率で垂直反転
    transforms.RandomRotation(degrees=50), #50度反転
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1), #色調調整
    transforms.RandomResizedCrop(size=(256,224)), #ランダムにリサイズとクロップ
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), #標準化の一つ前
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

#フォルダ分けされた画像データに対してアノテーション ※上のフォルダから0,1
dataset = datasets.ImageFolder(r'C:\Users\Junji Akiyama\Casting_Project\Dataset\train_data',transform)
# 画像ファイルパスとラベルを取得 ※アノテーションがcsv通りにできているかを確認
for img_path, label in dataset.imgs:
    class_name = dataset.classes[label]  # ラベルインデックスからクラス名を取得
    print(f"Image Path: {img_path}, Label: {label}, Class Name: {class_name}")

#データ分割
pl.seed_everything(0)
train ,val = torch.utils.data.random_split(dataset,[200,50]) 
#DataLoaderの準備
torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True,drop_last=True)
val_loader = torch.utils.data.DataLoader(val,batch_size=32)

#学習用データのクラスに偏りがあるため重みの調整
class_counts = [100, 150]  # クラス0が100枚、クラス1が150枚
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)# 各クラスのサンプル数に基づいてクラスウェイトを計算

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

  def training_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t,weight=class_weights)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=True, on_epoch=True, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t,weight=class_weights)
    self.log('val_loss', loss, on_step=False, on_epoch=True)
    self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=False, on_epoch=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(),lr=0.005,weight_decay=0.05)
    return optimizer

# 早期終了のコールバックを設定
early_stop_callback = EarlyStopping(
    monitor='val_loss',  # 監視するメトリック
    min_delta=0.00,      # これ以上の改善が見られない場合
    patience=30,          # 20エポック改善がない場合に停止
    verbose=True,
    mode='min'           # モニタリングするメトリックが小さいほど良い場合
)
torch.manual_seed(0)
net = Net()
logger = CSVLogger(save_dir='logs',name='my_exp') #学習ログを出力
trainer = pl.Trainer(max_epochs=256,deterministic=False,logger=logger,callbacks=early_stop_callback)
trainer.fit(net,train_loader,val_loader)
trainer.callback_metrics

#学習済みモデルのパラメータを保存
torch.save(net.state_dict(),'weight.pt')