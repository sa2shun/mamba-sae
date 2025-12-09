# Mamba/Transformer SAE実験パイプライン

Mamba-1.4BとTransformer（Pythia-1.4Bなど）の中間state h_tと更新量Δh_tに対してSAE（Sparse Autoencoder）を学習し、比較するための実験パイプラインです。

すべての成果物（chunk, checkpoint, log, 解析結果）と使用したYAML設定のコピーを、`output_root`で指定した同一ディレクトリ（例: `/raid/${USER}/mamba-gpt-saes/runs`）以下にまとめて保存できます。

## ディレクトリ構造

```
<output_root>/
  mamba_1.4b/
    layer_12/
      state/
        chunk_000.pt
        chunk_001.pt
        ...
        meta.json
        sae_checkpoint.pt
        train_log.jsonl
        analysis/
          sparsity_distribution.png
          half_life_distribution.png
          contexts_feature_*.txt
      delta/
        ...
    layer_24/
      ...
    layer_36/
      ...
  pythia_1.4b/
    ...
```

## セットアップ

必要なパッケージをインストール：

```bash
pip install torch transformers datasets tqdm matplotlib numpy pyyaml
```

## 使用方法

### 1. 特徴抽出

```bash
python extract_features.py --config configs/mamba_layer24_delta.yaml
```

オプション：
- `--layer`: 特定の層のみを処理（例: `--layer 24`）
- `--signal`: signalタイプを指定（`state` or `delta`）

### 2. SAE学習

```bash
python train_sae.py --config configs/mamba_layer24_delta.yaml
```

### 3. 解析

```bash
python analyze_sae.py --config configs/mamba_layer24_delta.yaml
```

オプション：
- `--skip-half-life`: half-life計算をスキップ

## 設定ファイル

設定ファイルはYAML形式で、以下の項目を含みます：

- `output_root`: 出力先ルート。`~` や `${USER}` を展開（例: `/raid/${USER}/mamba-gpt-saes/runs`）。省略時はカレントの`runs/`。
- `model_name`: モデル名（例: `state-spaces/mamba-1.4b-hf`）
- `model_type`: `mamba` または `transformer`
- `layers`: 対象層のリスト（例: `[12, 24, 36]`）
- `signal`: `state` または `delta`
- `dataset`: データセット設定
- `sae`: SAEのハイパーパラメータ
  - `mode`: `l1` または `k_sparse`
  - `hidden_factor`: オーバーコンプリート率
  - `l1_lambda`: L1正則化係数（`mode=l1`の場合）
  - `k_frac`: 活性化特徴の割合（`mode=k_sparse`の場合）
- `train`: 学習設定
- `extract`: 特徴抽出設定
- `devices`: 使用するGPU IDのリスト

## SAEモード

### L1 SAE

```yaml
sae:
  mode: l1
  l1_lambda: 3e-3
```

### k-sparse SAE

```yaml
sae:
  mode: k_sparse
  k_frac: 0.1  # 10%の特徴を活性化
```

## 出力

各スクリプト実行時に、使用したYAML設定ファイルを`<output_root>/<モデル名ディレクトリ>/`へコピーして再現性を確保します。

### 特徴抽出

- `chunk_*.pt`: 特徴データのchunkファイル
- `meta.json`: メタ情報

### SAE学習

- `sae_checkpoint.pt`: 学習済みSAEのチェックポイント
- `train_log.jsonl`: 学習ログ

### 解析

- `analysis/sparsity_distribution.png`: スパース性分布のヒストグラム
- `analysis/half_life_distribution.png`: half-life分布（state-SAEの場合）
- `analysis/contexts_feature_*.txt`: 各featureのtop activation contexts

## 注意事項

- GPUメモリが不足する場合は、`batch_size`や`chunk_size`を調整してください
- 大規模なデータセットを使用する場合は、`max_docs`や`max_tokens_total`で制限を設定してください
- DataParallelを使用しているため、モデルにアクセスする際は`.module`を使用してください
