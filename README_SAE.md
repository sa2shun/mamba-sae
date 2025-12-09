# Mamba/Transformer SAE実験パイプライン

Mamba-1.4BとTransformer（Pythia-1.4Bなど）の中間state h_tと更新量Δh_tに対してSAE（Sparse Autoencoder）を学習し、比較するための実験パイプラインです。

すべての成果物（chunk, checkpoint, log, 解析結果）と使用したYAML設定のコピーを、`output_root`で指定した同一ディレクトリ（例: `/raid/${USER}/mamba-gpt-saes/runs`）以下にまとめて保存できます。

## ディレクトリ構造

```
<output_root>/
  mamba_1.4b/
    layer_12/
      state/
        stats.pt
        chunk_000.pt
        chunk_001.pt
        ...
        meta.json
        grid/
          hf4_k0p05_l10p0001_eq0p001_seed0/
            sae_checkpoint.pt
            train_log.jsonl
            epoch_summary.jsonl
            sample_metrics.pt
            summary.json
            analysis/
              sparsity_distribution.png
              half_life_distribution.png
              contexts_feature_*.txt
        grid_results.jsonl
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
pip install torch transformers datasets tqdm matplotlib numpy pyyaml pandas seaborn
```

## 使用方法（グリッド実験フロー）

1. 特徴抽出（例: Mamba 全層、state & delta）
   ```bash
   python extract_features.py --config configs/mamba_full_grid.yaml
   python extract_features.py --config configs/pythia_full_grid.yaml
   ```
   - `--layer` で個別層のみ実行
   - `--signal` で state / delta / resid_pre_mlp / resid_post_mlp を個別に指定可能（resid_* は transformer のみ）

2. SAEグリッド学習（81ハイパラ × 3 seeds）
   ```bash
   python train_sae.py --config configs/mamba_full_grid.yaml
   # テストランなら一部だけ:
   python train_sae.py --config configs/mamba_full_grid.yaml --limit-grid 2
   ```
   - グリッド結果は `<output_root>/<model>/layer_<L>/<signal>/grid/` 以下に run ごと保存
   - サマリー: `<output_root>/<...>/grid_results.jsonl`

3. 単体/グリッド解析
   - 最良（または指定）run の詳細解析:
     ```bash
     python analyze_sae.py --config configs/mamba_full_grid.yaml --layer 12 --signal state --run-name hf4_k0p05_l10p0001_eq0p001_seed0
     ```
   - ハイパラ heatmap / good-regime 抽出:
     ```bash
     python analysis/grid_reports.py --root /raid/.../mamba_1.4b_hf/layer_12/state --metric recon_loss --l1 0.0001 --eq 0.001
     ```
   - 辞書類似度（深さ/モデル/信号の比較）:
     ```bash
     python analysis/compare_saes.py --ckpt-a <runA>/sae_checkpoint.pt --ckpt-b <runB>/sae_checkpoint.pt --top-k 50
     ```

## 設定ファイル

設定ファイルはYAML形式で、主なキーは以下の通り：

- `output_root`: 出力先ルート。`~` や `${USER}` を展開。
- `model_name` / `model_type`: 例 `state-spaces/mamba-1.4b-hf`（mamba） / `EleutherAI/pythia-1.4b-deduped`（transformer）
- `layers`: 対象層のリスト（mamba: `[4,12,24,36,44]`, pythia: `[2,6,12,18,24]`）
- `signals`: `state` / `delta` / `resid_pre_mlp` / `resid_post_mlp` のリスト（resid_* は transformer のみ）。`signal` 単体指定も可。
- `dataset`:
  - `max_tokens_total` (推奨 1–2M): 学習・正規化に使うサンプル数
  - `stats_tokens`: mean/std計算用サンプル数
  - `max_docs`, `name`, `config`, `split`
- `sae`:
  - `hidden_factor`: コード幅倍率（例 `[2,4,8]`）
  - `k_frac`: k-sparse割合（例 `[0.02,0.05,0.1]`）
  - `l1_lambda`: L1係数（0 も含めて k-sparse 併用可）
  - `eq_alpha`: firing-rate equalization係数
  - `eq_target`: firing target（省略時は `k_frac`）
- `train`:
  - `seeds`: 例 `[0,1,2]`
  - `batch_size`, `num_epochs`(固定5), `val_split`(0.1), `lr`(1e-3), `weight_decay`(0.01)
  - `warmup_steps`(1000), `clip_grad_norm`(1.0), `log_every`
  - `num_workers`, `pin_memory`
- `extract`: `max_length`, `batch_size`, `chunk_size`
- `devices`: 使用GPU IDリスト

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
