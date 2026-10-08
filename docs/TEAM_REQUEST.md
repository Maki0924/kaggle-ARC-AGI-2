# チームメンバーへの依頼: Kaggle での確認実行

ARC Prize 2026 - ARC-AGI-2 で使っているハイブリッド版(NVARC + Qwen3.8-27B)の確認実行を、あなたの Kaggle アカウントの GPU クォータで回してもらいたいです。こちらのアカウントは今週のクォータを使い切りました(更新は 10/10 09:00 JST)。

背景は `docs/PROGRESS.md` にまとめてあります。

## お願いしたいこと(所要: 作業15分 + 待ち2時間前後)

### 0. 事前に教えてほしいこと

- Kaggle のユーザー名
- このコンペでの自分の提出回数(チームマージの条件確認のため)
- 今週の GPU クォータの残り(Kaggle 画面右上のプロフィール →「Your Work」付近、または `kaggle quota`)。**L4×4 は2倍消費**なので、1回の確認実行(実時間約1.3時間)で約2.6時間減ります

### 1. チームへの参加

- コンペのページ → Team タブで、こちらからのマージ依頼を承認してください(締切 10/26)
- **このコンペへの提出はしないでください**。提出枠はチームで1日1回の共有です

### 2. 準備(初回のみ)

- Kaggle の Settings → API → Create New Token でトークンを作成し、環境変数 `KAGGLE_API_TOKEN` に設定
- Kaggle CLI: `pip install kaggle`(または `uvx kaggle`)
- このリポジトリを clone

```bash
git clone https://github.com/Maki0924/kaggle-ARC-AGI-2.git
cd kaggle-ARC-AGI-2
```

- vLLM の wheel 一式は公開データセット `koumeimaki/vllm019-cp311-cu128-wheelhouse` にあり、ノートブックに自動で添付されます(追加の設定は不要)

### 3. 確認実行の投入

`kaggle/arc2-hybrid1-nvarc-qwen38/kernel-metadata.json` の `id` を自分のユーザー名に書き換えて投入します。

```bash
cd kaggle/arc2-hybrid1-nvarc-qwen38
# "id": "koumeimaki/arc2-hybrid1-nvarc-qwen38" → "<あなたのユーザー名>/arc2-hybrid1-nvarc-qwen38"
kaggle kernels push -p .
```

- 公開評価の8問だけを回す設定になっています(本番用の隠しデータは使いません)
- 実行は L4×4、Python 3.11 イメージ、インターネット無効で自動設定されます
- キュー待ちが数時間になることがあります。状態確認: `kaggle kernels status <ユーザー名>/arc2-hybrid1-nvarc-qwen38`

### 4. 結果の回収

完了(`COMPLETE`)したら出力を取得して、リポジトリに PR してください。

```bash
kaggle kernels output <ユーザー名>/arc2-hybrid1-nvarc-qwen38 -p out
mkdir -p results/hybrid/v11_<日付>
cp out/arc2-hybrid1-nvarc-qwen38.log out/freeform_samples.jsonl out/freeform_receipt.json \
   out/submission.json out/nvarc_submission.json out/nvarc_confidence.json out/vllm_server.log \
   results/hybrid/v11_<日付>/
git checkout -b results/hybrid-v11
git add results/hybrid && git commit -m "Hybrid v11 commit-run results" && git push -u origin results/hybrid-v11
```

`ERROR` で終わった場合も、ログを同じ手順で上げてもらえれば原因を調べます。

## この実行で確認すること

| 項目 | 見るもの |
|---|---|
| fp8 KV キャッシュ + TRITON_ATTN で vLLM が起動するか | `vllm_server.log` |
| 同時16での生成速度 | ログの `decode_tokens_per_second`(前回は同時8で約240〜280) |
| 打ち切り時の回収(推論全体を再投入)で正解が出るか | `freeform_samples.jsonl` の `length+recovered` |
| 合流前後の正解数 | ログ末尾の `on 8 handoff tasks: NVARC x -> hybrid y` |

## 注意

- 同じコンペの L4×4 は混雑していて、キュー待ちが長いことがあります
- 実行中に同じノートブックを再投入すると前の実行が止まります
- 不明点があれば遠慮なく聞いてください
