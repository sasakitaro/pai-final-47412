# Physical AI 応用編1 最終課題（Omnicampus ID: 47412）

## 課題テーマ
**GenesisシミュレータにおけるAI視覚認識・MLP軌道生成ポリシーを用いた7自由度ロボットアーム（Franka Emika Panda）の自律色別仕分けタスク**

---

## 1. 概要・目的
本リポジトリは、Physical AI 応用編1講座の最終課題提出用コード一式です。
物理シミュレータ **Genesis** 上に構築した7自由度ロボットアーム（Franka Emika Panda）と3色の対象オブジェクト（RED / GREEN / BLUE キューブ）に対し、以下の2つのAIモジュールを組み合わせた自律ピッキング＆色別仕分け（Pick & Place）システムを実装しています。

1. **AI視覚認識モジュール（`VisualPerceptionAI` in `ai_controller.py`）**
   - シミュレータ内カメラが撮影したRGB画像からHSV色空間セグメンテーションおよび画像モーメント抽出を行い、各色キューブの画像重心を検出して3次元ワークスペース座標と仕分け先ゴール座標を推定・計画します。
2. **ニューラル軌道生成ポリシー（`NeuralTrajectoryPolicy` in `ai_controller.py`）**
   - 始点・終点・移動フェーズを入力とする **PyTorch製 多層パーセプトロン（MLP）モデル** により、Minimum-Jerk（最小躍度）ベース軌道に対する滑らかな残差補正と障害物回避アーチ高さをリアルタイム推論し、Genesisの逆運動学（IK）ソルバおよびPD制御器を介して安定した把持・搬送を実現します。

---

## 2. ファイル構成
- `PAI_Final_Genesis_Colab.ipynb` : **Google Colab（無料NVIDIA T4 GPU）上でワンクリックで実行・動画出力できるノートブック**
- `run_genesis_sim.py` : Genesis シミュレーション環境の構築・ループ実行・HUDテレメトリ描画・MP4動画（75秒）出力スクリプト
- `ai_controller.py` : AI視覚認識モジュール（`VisualPerceptionAI`）およびPyTorch MLP軌道生成ポリシー（`NeuralTrajectoryPolicy`）
- `requirements.txt` : 依存ライブラリ一覧

---

## 3. 実行方法

### 方法A：Google Colab（NVIDIA GPU）で実行する場合（推奨・最短）
1. Google Colab で `PAI_Final_Genesis_Colab.ipynb` を開きます。
2. メニューの **「ランタイム」→「ランタイムのタイプを変更」** から **T4 GPU** を選択します。
3. **「ランタイム」→「すべてのセルを実行」** をクリックすると、自動的に環境構築・シミュレーション実行・動画生成（`PAI最終課題_47412.mp4`、1分15秒・等倍）が行われ、PCへダウンロードされます。

### 方法B：ローカル環境で実行する場合
```bash
pip install -r requirements.txt
python run_genesis_sim.py
```
実行完了後、カレントディレクトリに `PAI最終課題_47412.mp4`（長さ：1分15秒、等倍再生）が出力されます。
