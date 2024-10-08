【概要】
このプロジェクトでは、PyTorch Lightningを使ってResNet18をベースとした2クラス分類モデルの学習を行い、
さらにFastAPIを使用して、推論用APIを構築します。

【プロジェクトの構成】
学習フェーズ: 画像データに対してデータ拡張を行い、クラス不均衡を考慮したResNet18モデルを学習。
推論フェーズ: 学習済みのモデルを使用して、API経由で画像の分類を行う。

【前提条件】
PyTorchとPyTorch Lightningがインストール済み
FastAPIがインストール済み（インストールは pip install fastapi）
Uvicornがインストール済み（インストールは pip install uvicorn）
事前学習済みのResNet18モデルの重みファイルが存在する

【使用技術】
Python
PyTorch Lightning
torchvision
FastAPI
PIL (Python Imaging Library)

【構成】
データセットへのラベリング
モデルの定義
モデルの学習
重みの作成
推論前処理
FastAPIサーバーの構築
画像ファイルのアップロード処理
推論結果の返却

【説明】
・データセットへのラベリング
　ラベリングCSVデータをもとにデータセットに対して0,1フラグを割り振ります。

・モデルの定義
　ResNet18をベースにした2クラス分類用のニューラルネットワークを定義しています。self.feature は事前学習済みのResNet18を使用し、最終的に2つのクラスに分類します。

・モデルのロード
　事前学習済みの重みファイル weight.pt を指定してモデルをロードしています。モデルを eval モードに設定し、推論用に最適化しています。

・APIエンドポイント /casting_predict/
　FastAPIのエンドポイントを定義しています。
　以下の処理が行われます。
　1.クライアントからアップロードされた画像ファイルを受け取る。
　2.画像をPILで開き、predict_transform を適用して前処理する。
　3.事前に学習済みのモデルに画像を入力し、推論を行う。
　4.予測結果をクライアントに返却する。

・推論の実行
　with torch.no_grad() を使用して、推論時には勾配計算を無効化しています。
　画像はバッチサイズ1として入力され、推論結果は2クラスの確率がsoftmax関数を介して取得され、最大確率のクラスが最終結果として返されます。

・返却するデータ
　推論結果として、以下のJSON形式で返却します。
　{
  　"filename": "uploaded_image.jpg",
  　"predicted_class": 1
　}

【実行方法】
1.データの準備
フォルダにクラスごとの画像データを配置します。

2.スクリプトの実行
Pythonスクリプトを実行し、学習を開始します。
学習済みモデルはweight.ptとして保存されます。

3.サーバーの起動 FastAPIアプリケーションをUvicornで実行します。
uvicorn casting_class_fastapi:app --reload

4.画像ファイルをアップロードして推論を行う 
クライアントから画像ファイルをPOSTメソッドでエンドポイント /casting_predict/ に送信します。
 curl -X POST "http://127.0.0.1:8000/casting_predict/" \
   　-F "file=@path_to_image.jpg"

5.レスポンス APIは以下の形式のJSONレスポンスを返します。
　{
  "filename": "image.jpg",
　"predicted_class": 1　
 }

【注意事項】
事前に学習済みのモデルが正しくロードされることを確認してください。
APIサーバーは、画像ファイルのアップロードと推論を行うための環境が適切に整備されている必要があります。
