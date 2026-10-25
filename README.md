# Physical AI 応用編1 最終課題（Omnicampus ID: 47412）

## 課題テーマ
**GenesisシミュレータにおけるAI視覚認識・MLP軌道生成ポリシーを用いた7自由度ロボットアーム（Franka Emika Panda）の自律色別仕分けタスク（異順列・交差マッチング対応）**

---

## 1. 概要・目的
本リポジトリは、Physical AI 応用編1講座の最終課題提出用コード一式です。
物理シミュレータ **Genesis** 上に構築した7自由度ロボットアーム（Franka Emika Panda）と3色の対象オブジェクト（RED / GREEN / BLUE キューブ）に対し、**ボックスの初期配置の色順序と指定場所（トレイ）の色順序が異なる「完全順列（シャッフル配置）」環境**においても、AIが双方の色と位置を認識・照合して正確にPlaceできる自律ピッキングシステムを実装しています。

1. **AI双方向視覚認識・色照合モジュール（`VisualPerceptionMatchAI` in `ai_controller_shuffled.py`）**
   - シミュレータ内カメラが撮影したRGB画像から、把持側キューブ（`[Slot 1: 青, Slot 2: 赤, Slot 3: 緑]`）と指定トレイ（`[Bin 1: 緑, Bin 2: 青, Bin 3: 赤]`）の双方をHSV色空間でセグメンテーションし、同一色ラベル同士を空間マッチング（`Slot 1 → Bin 2`, `Slot 2 → Bin 3`, `Slot 3 → Bin 1`）して交差搬送タスクを計画します。
2. **ニューラル対角軌道生成ポリシー（`NeuralTrajectoryPolicy` in `ai_controller_shuffled.py`）**
   - 始点・終点・移動フェーズ・水平横断距離を入力とする **PyTorch製 多層パーセプトロン（MLP）モデル** により、Minimum-Jerk（最小躍度）ベース軌道に対する滑らかな残差補正と、対角長距離搬送時の適応クリアランス高さをリアルタイム推論し、Genesisの逆運動学（IK）ソルバおよびPD力制御器を介して安定した把持・搬送を実現します。

---

## 2. ファイル構成

### ■ 発展版：異順列（シャッフル配置）交差マッチングタスク
- `PAI_Final_Genesis_Colab_shuffled.ipynb` : **Google Colab（無料NVIDIA T4 GPU）上で異順列タスクを実行し、`PAI最終課題_47412_shuffled.mp4` を生成・ダウンロードするノートブック**
- `run_genesis_sim_shuffled.py` : 異順列配置（Cubes: `[青, 赤, 緑]` / Bins: `[緑, 青, 赤]`）のGenesisシミュレーション実行＆HUD動画出力スクリプト
- `ai_controller_shuffled.py` : 双方向色認識・交差マッチングAI（`VisualPerceptionMatchAI`）＆ PyTorch MLP適応アーチ軌道ポリシー（`NeuralTrajectoryPolicy`）

### ■ 基本版：整列配置タスク
- `PAI_Final_Genesis_Colab.ipynb` : 基本版Colabノートブック（`PAI最終課題_47412.mp4` を出力）
- `run_genesis_sim.py` : 基本版Genesisシミュレーション実行スクリプト
- `ai_controller.py` : 基本版AI視覚認識＆MLP軌道生成モジュール
- `requirements.txt` : 依存ライブラリ一覧

---

## 3. 実行方法

### Google Colab（NVIDIA GPU）で実行する場合（推奨）
1. Google Colab で `PAI_Final_Genesis_Colab_shuffled.ipynb`（または基本版 `PAI_Final_Genesis_Colab.ipynb`）を開きます。
2. メニューの **「ランタイム」→「ランタイムのタイプを変更」** から **T4 GPU** を選択します。
3. **「ランタイム」→「すべてのセルを実行」** をクリックすると、シミュレーション実行・動画生成（`PAI最終課題_47412_shuffled.mp4`、1分15秒・等倍）が行われ、PCへ自動ダウンロードされます。
