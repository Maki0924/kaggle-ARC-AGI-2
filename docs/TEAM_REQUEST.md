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

### 4. 結果の共有(PR で push)

実行が終わったら(`COMPLETE` でも `ERROR` でも)、出力をリポジトリに PR で上げてください。

#### 置き場所と名前

```
results/hybrid/<バリアント>_<YYYYMMDD>_<ユーザー名>/
```

例: `results/hybrid/hybrid1_20261009_taro/`

#### 入れるファイル

| ファイル | 内容 | 必須 |
|---|---|---|
| `arc2-hybrid1-nvarc-qwen38.log` | ノートブック全体のログ | ○ |
| `vllm_server.log` | vLLM サーバーのログ(起動に失敗した場合の原因はここ) | ○ |
| `freeform_samples.jsonl` | Qwen の回答1件ごとの記録 | ある場合 |
| `freeform_receipt.json` / `freeform_status.json` | Qwen 部分の集計 | ある場合 |
| `nvarc_submission.json` / `nvarc_confidence.json` | NVARC の答えと確信度 | ある場合 |
| `submission.json` | 合流後の最終出力 | ある場合 |
| `run_info.md` | 下の雛形を埋めたメモ | ○ |

**入れないもの**: `__pycache__/`、`unsloth_compiled_cache/`、`*.py`、`*.ipynb`、`freeform_previous_runs/`(どれもこちらで再生成できます)。

#### 手順

```bash
U=<ユーザー名>; D=results/hybrid/hybrid1_$(date +%Y%m%d)_$U
kaggle kernels output $U/arc2-hybrid1-nvarc-qwen38 -p out
mkdir -p $D
for f in arc2-hybrid1-nvarc-qwen38.log vllm_server.log freeform_samples.jsonl freeform_receipt.json \
         freeform_status.json nvarc_submission.json nvarc_confidence.json submission.json; do
  [ -f out/$f ] && cp out/$f $D/
done
# run_info.md を下の雛形から作成して $D に置く
git checkout -b results/hybrid1-$(date +%Y%m%d)-$U
git add $D && git commit -m "Hybrid1 commit-run results ($(date +%Y-%m-%d), $U)"
git push -u origin HEAD
gh pr create --title "結果: hybrid1 確認実行 $(date +%m/%d) ($U)" --body-file $D/run_info.md
```

`gh` がなければ、GitHub の画面から PR を作って本文に `run_info.md` の内容を貼ってください。**マージはしないでください**(こちらで確認してからマージします)。

#### `run_info.md` の雛形

```markdown
# hybrid1 確認実行の結果

- 実行者: <ユーザー名>
- ノートブック: <ユーザー名>/arc2-hybrid1-nvarc-qwen38 version <番号>
- 投入日時 / 開始日時 / 終了日時 (JST): <...> / <...> / <...>
- 最終ステータス: COMPLETE / ERROR
- 消費した GPU クォータ(実行前後の `kaggle quota` の差): <...> 時間

## ログ末尾の要約(該当行をそのまま貼る)
- `NVARC done at ... h`:
- `vLLM ready: ...`:
- `decode_tokens_per_second`:
- `outputs changed by Qwen: ...`:
- `on 8 handoff tasks: NVARC x -> hybrid y`:

## 気づいたこと(任意)
```

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
