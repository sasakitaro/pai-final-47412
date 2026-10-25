# Physical AI 応用編1 最終課題（Omnicampus ID: 47412）

## 課題テーマ
**GenesisシミュレータにおけるAI色認識・空間照合・不一致返却判断を備えた7自由度ロボットアーム（Franka Emika Panda）の段階的Pick & Placeシステム**

---

## 1. 概要・目的
本リポジトリは、Physical AI 応用編1講座の最終課題提出用コード一式です。
物理シミュレータ **Genesis** 上の7自由度ロボットアーム（Franka Emika Panda）とRGBカメラ、および **PyTorch製 多層パーセプトロン（MLP）軌道生成ポリシー** を組み合わせ、難易度を段階的に引き上げた以下の **3段階の自律ピッキングタスク（Part 1 〜 Part 3）** を実装・検証しています。

1. **Part 1：基本の整列色別仕分けタスク（`run_genesis_sim.py`）**
   - 初期配置 `[赤, 緑, 青]` の3色ボックスを視覚認識し、同順の指定場所 `[赤, 緑, 青]` へ順次自律Pick & Placeします。
2. **Part 2：位置シャッフル（完全順列）における交差マッチングタスク（`run_genesis_sim_shuffled.py`）**
   - ボックスの初期配置 `[Slot 1: 青, Slot 2: 赤, Slot 3: 緑]` と指定場所 `[Bin 1: 緑, Bin 2: 青, Bin 3: 赤]` の順番がすべて異なる環境において、AIが双方の色と位置を照合し、対角交差軌道（`Slot 1 → Bin 2`, `Slot 2 → Bin 3`, `Slot 3 → Bin 1`）で正確にPlaceします。
3. **Part 3：不一致色の配置却下＆元の位置への返却タスク（`run_genesis_sim_reject.py`）**
   - 3色のうち **2つ目の色（`Slot 2: 赤`）に対応する置き場所が存在しない環境**（指定場所が `[Bin 1: 緑, Bin 2: 黄, Bin 3: 青]`）において、ロボットが `Bin 2（黄）` の色不一致を検知し、**置かずに元の位置（`Slot 2`）へ持ち帰って正確に戻す（Return-to-Origin）** 高度な例外処理タスクを実行します。

---

## 2. ファイル構成

### ■ Part 3：不一致色の配置却下＆元の位置への返却タスク（最高難易度）
- `PAI_Final_Genesis_Colab_reject.ipynb` : Google Colab（NVIDIA GPU）用ノートブック（`PAI最終課題_47412_reject.mp4` を出力）
- `run_genesis_sim_reject.py` : 2つ目の色（赤）の置き場所が無い環境での不一致検知・元の位置への返却シミュレーション
- `ai_controller_reject.py` : 色照合・不一致返却意思決定AI（`VisualRejectAndReturnAI`）＆ PyTorch MLP軌道ポリシー

### ■ Part 2：異順列（位置シャッフル）交差マッチングタスク
- `PAI_Final_Genesis_Colab_shuffled.ipynb` : Google Colab用ノートブック（`PAI最終課題_47412_shuffled.mp4` を出力）
- `run_genesis_sim_shuffled.py` : 異順列配置の交差Pick & Placeシミュレーション
- `ai_controller_shuffled.py` : 双方向色認識・交差マッチングAI（`VisualPerceptionMatchAI`）

### ■ Part 1：基本の整列色別仕分けタスク
- `PAI_Final_Genesis_Colab.ipynb` : Google Colab用ノートブック（`PAI最終課題_47412.mp4` を出力）
- `run_genesis_sim.py` : 基本版シミュレーション実行スクリプト
- `ai_controller.py` : 基本版AI視覚認識＆MLP軌道生成モジュール

### ■ 全体動画結合スクリプト（2倍速・3分以内・前後スライド付き）
- `combine_three_videos.py` : 冒頭と末尾に `PAI応用1_最終課題_スライド.jpg`（各4秒）を配置し、Part 1〜Part 3の3つの動画を **2倍速（2x Speed明記）** で1つに繋ぎ合わせ、約2分00秒の全体提出動画 `PAI最終課題_47412_combined.mp4` を生成するスクリプト。
- `requirements.txt` : 依存ライブラリ一覧
