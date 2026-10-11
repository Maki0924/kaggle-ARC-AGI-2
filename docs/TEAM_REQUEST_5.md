# 依頼5: Qwen A/B の確認セット(12問)

hybrid5 の提出ありがとうございました。並行して、NVARC の出力を固定して Qwen3.8 だけを回す比較(`docs/A100_PLAN.md`)を Kaggle の L4×4 で実施しました。探索36問の結果(`docs/PROGRESS.md` 3.1)は、トークン予算 32k ではほぼ無駄、64k なら medium で +6.0(損失 1.0)、xhigh で +3.5(損失 0)でした。1出力1サンプルなので、取り置いていた確認12問で **64k の2条件を各2サンプル**回して、結果が再現するかを見たいです。これを **1本** お願いします。

自分の週クォータは探索の実行で使い切っているため、お願いしています。

**先に `git pull` で最新の `main` を取ってください。**

## お願いする実行

| フォルダ | 中身 | 見たいこと |
|---|---|---|
| `kaggle/arc2-qwen-ab-l4-confirm` | 確認12問 / 16出力に対して、xhigh 64k → medium 64k の順に各2サンプル。NVARC は動かさず(固定入力を埋め込み済み)、Qwen のサーバーだけ立てる | 条件ごとの純増(新規正解 − NVARC の正解の喪失)が探索と同じ傾向か |

- GPU クォータを約 **8〜9時間**(L4×4 なので計上は2倍)使います。10.5時間で新しいサンプルの開始を止めるので、12時間の上限には収まります
- 競技データを添付していますが(L4×4 を得るため)、正解ファイルはツールの実行環境から読めないようにしてあり、起動時に自己検査します。**提出用のファイルは作りません**

## 投入

```bash
git pull
cd kaggle/arc2-qwen-ab-l4-confirm
# kernel-metadata.json の "id" を "<ユーザー名>/arc2-qwen-ab-l4-confirm" に書き換え
kaggle kernels push -p .
```

## 結果の共有

```bash
kaggle kernels output <ユーザー名>/arc2-qwen-ab-l4-confirm -p results/qwen_ab/<YYYYMMDD>_<ユーザー名>
```

- `results/qwen_ab/<YYYYMMDD>_<ユーザー名>/` を PR にしてください(`runs/*/raw/` と `ab/` は `.gitignore` 済みなので入りません)
- 採点はこちらでやります(`python a100/score.py results/qwen_ab/<...>/runs/*`)。試す場合は `uv run --no-project python a100/score.py <run dir>` で動きます
- `docs/TEAM_REQUEST.md` と同じく、投入日時・最終ステータス・消費クォータを `run_info.md` に書いてもらえると助かります

今回もコンペへの提出はしないでください。
