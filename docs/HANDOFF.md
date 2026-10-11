# 引き継ぎ(2026-10-12 時点)

ARC Prize 2026 - ARC-AGI-2(Kaggle)の作業を別のエージェントに引き継ぐためのまとめ。経緯の詳細は `docs/PROGRESS.md`、数値は `docs/EXPERIMENTS.md`、Qwen 比較の計画と結果は `docs/A100_PLAN.md`。この文書は「今どこにいて、次に何をするか」だけを書く。

## 1. 前提と約束ごと

- 締切: 最終提出 **2026-11-02 23:59 UTC**(JST 11/3 08:59)。参加・チームマージは 10/26。提出は **チームで1日1回**、採点待ち(PENDING)は同時に1件まで。最終選択は2本
- 本番環境: Kaggle ノートブック、12時間、ネット不可、L4×4(96GB)。L4×4 はクォータを2倍計上。週クォータは土曜 09:00 JST にリセット
- 目標: 銀(10/8 時点 32.22)〜金(37.36)。NVARC 系の平均 28.2 から +4 で銀
- チーム: 自分(koumeimaki)と harukana。**提出とクォータを使う実行は、依頼した内容以外はメンバーに頼まない**。依頼は `docs/TEAM_REQUEST_*.md` に書いて PR にする
- 進め方: 数値を早く出すことを優先し、確認待ちで止まらない。ユーザーへの返答は**日本語**
- 作業の流れ: ブランチ → PR → **squash マージ**。メンバーからの PR はマージせず開けておく(本人がマージする)
- レビュー: Codex CLI を使うときは `codex exec -m gpt-6-astra -c model_reasoning_effort="high" --sandbox read-only`(既定の effort は low なので必ず指定)
- ローカル: RTX 4090(24GB)、`uv` / `uvx kaggle` あり(素の python / pip はない。`uv run --no-project python ...` で実行)。Docker イメージ `arc2-nvarc:local` が NVARC 再現環境。大きなファイルは `C:\Users\makik\arc2-local\`(リポジトリは OneDrive 配下なので置かない)
- セキュリティ: Kaggle API トークンをコミット・表示しない。`kaggle.json` は .gitignore 済み

## 2. 解法の全体像

1. **NVARC**(公開解法: Qwen3-4B + タスク別 LoRA の TTT)で全240問を解き、出力と確信度(候補数・票数)を保存。dev80 で 24.00/80、本番 public 25〜31(平均 28.2、シードでばらつく)
2. **Qwen3.8-27B-FP8**(vLLM 0.19、TP=4、fp8 KV + TRITON_ATTN、MTP、Python ツールあり)を、NVARC が低確信度の出力にだけ回す
3. **合流**: Qwen が自然に考え終わり("stop")有効なグリッドを出したものだけ採用。候補ゼロなら attempt_1 に、候補が弱ければ attempt_2 に。強制確定・打ち切り回収の答えは捨てる

実体は `scripts/build_hybrid.py` が生成する `kaggle/arc2-hybrid5-nvarc-qwen38-medium/`(現行の本番候補)。

## 3. 現在地(数値)

### 本番(public LB)

| 版 | 内容 | public |
|---|---|---|
| NVARC 系 5本 | baseline / fast1 / seed3 / seed4 | 31.11 / 29.03 / 25.28 / 26.94 / 28.61(平均 28.2) |
| hybrid v9 ×3 | NVARC + Qwen xhigh、考え終わりのみ採用 | 28.61 / 27.78 / 26.11(平均 27.5) |
| **hybrid5** ×1 | NVARC + Qwen **medium 32k**、同時16、強制確定は不採用 | **29.44**(10/10 提出、採点に約2日) |

### Qwen A/B(固定 NVARC 入力、探索36問 / 44出力、Kaggle L4×4、10/10)

NVARC の出力を固定して Qwen だけを回し、本番と同じ合流で「何問増えて何問失うか」を数えた。NVARC 単体 6.83。詳細は `docs/A100_PLAN.md` 7.2。

| 条件 | 所要 | 純増 | 増えたタスク | 損失 | stop | 判定 |
|---|---|---|---|---|---|---|
| medium 32k | 1.3h | +1.0 | 1 | 0 | 5/44 | × |
| xhigh 32k | 1.1h | +1.0 | 1 | 0 | 5/44 | × |
| medium 64k | 4.5h | **+6.0** | 9 | 1.0 | 17/44 | 損失で保留 |
| xhigh 64k | 3.3h | **+3.5** | 4 | 0 | 14/44 | **合格** |

結論: **効くのは推論強度ではなくトークン予算**。32k は推論強度によらずほぼ予算切れ。64k で初めて複数問が増える。medium は平均 12.8 ターン(本番 hybrid5 の10ターン上限を超える)、xhigh は 6.1 ターン。

### 分かっていること

- NVARC 側の調整(選択・ビュー・シード・TTT 設定)は効かない。打ち止め
- Qwen が NVARC の正解を上書きする損失は実在する(hybrid4 の強制確定、medium 64k の b5ca7ac4)
- 公開評価8問の確認実行では毎回同じ1問しか増えず、設定を区別できない → 固定入力の A/B(上記)で測る
- 本番で Qwen の実生成に使えるのは約3〜3.5時間。同時16で 44出力 × 64k に 3.3〜4.5時間かかったので、64k を回せるのは 30〜45出力程度

## 4. 次にやること(優先順)

1. **確認12問の A/B**: `kaggle/arc2-qwen-ab-l4-confirm`(xhigh 64k → medium 64k、各2サンプル、約8時間)。依頼文は `docs/TEAM_REQUEST_5.md`。自分の週クォータは探索で使い切った見込みなので、残りを確認してから自分で投入するかメンバーに頼む。結果は `uvx kaggle kernels output <id> -p results/qwen_ab/<日付>` → `uv run --no-project python a100/score.py results/qwen_ab/<日付>/runs/*`。判定基準は `docs/A100_PLAN.md` 第5節(確認: +1 以上、損失ゼロ)
2. **hybrid6 の作成**: `scripts/build_hybrid.py` に hybrid5 を元にしたバリアントを追加。Qwen の予算を 64k、ターン上限を外す(または大きくする)、推論強度は確認結果で決める(現時点の本命は xhigh)。64k を全対象に回せないので、対象を層(候補ゼロ / 候補1つ / 2番目の票が弱い)で絞るか、時間で打ち切る順序を決める。A/B の層別純増は「候補1つ(W)」に集中していた
3. **hybrid6 の確認実行**(公開評価8問、メンバーのクォータ)→ 本番提出。本番は NVARC の出来でばらつくので 2〜3本の平均で判断
4. それまでの本番枠は **hybrid5 を毎日1本**出して平均を取る(メンバーに依頼)
5. 10/22 までに成功済みの候補を2本確保。10/27 以降は大きな変更をしない

後回しにしているもの: SGLang 移行(`docker/sglang-test/`、4090 での起動検証途中、未コミットの変更あり)、大学の A100(計画は `docs/A100_PLAN.md`、優先1・2は Kaggle で済んだので残りの条件用)。

## 5. 主なファイルとコマンド

| パス | 内容 |
|---|---|
| `src/nvarc/` | NVARC パイプライン |
| `scripts/build_hybrid.py` → `kaggle/arc2-hybrid*/` | ハイブリッド本番ノートブックの生成(公開ランナーへのパッチ) |
| `scripts/build_fixed_nvarc.py` → `a100/fixed_nvarc/` | 固定 NVARC 入力(探索36問 / 確認12問の割り当て、dev80 で 24.00 を再現) |
| `scripts/build_a100_harness.py` → `a100/runner_parts.py` | hybrid5 のランナーからプロンプト・グリッド解析・ツール実行環境を抜き出す |
| `a100/qwen_ab.py` | Qwen だけを回す(総トークン予算、`/tokenize` で正確なプロンプト長、強制確定なし、再開可、正解ファイルの読み取りを監査フックで禁止) |
| `a100/score.py` | 本番と同じ合流を固定順で再生し、出力単位で純増・損失を出す。未完走は `--partial` |
| `a100/serve.sh` | vLLM サーバー起動(`MODEL VENV TP SEQS KVFP8 MTP CTX MEM LOGDIR`)。`servers.json` を書く |
| `scripts/build_qwen_ab_notebook.py [explore\|confirm]` → `kaggle/arc2-qwen-ab-l4[-confirm]/` | 上のハーネスを Kaggle L4×4 で回すノートブック |
| `results/qwen_ab/20261010/` | 探索 A/B の結果(samples / turns / merged_submission / scores) |
| `results/hybrid/` | メンバーの確認実行の結果(hybrid1 / 2 / 4 / 5) |
| `scripts/kaggle_watch.py` | 公開ノートブック・ディスカッション・LB の差分取得(セッション開始時に実行) |
| `docs/TEAM_REQUEST*.md` | メンバーへの依頼(1〜5) |

よく使うコマンド:

```bash
uvx kaggle competitions submissions arc-prize-2026-arc-agi-2   # 提出の状態
uvx kaggle kernels status koumeimaki/<kernel>                  # ノートブックの状態
uvx kaggle kernels push -p kaggle/<dir>                        # 投入(クォータ消費に注意)
uvx kaggle kernels output koumeimaki/<kernel> -p results/...   # 出力の取得
uv run --no-project python scripts/build_hybrid.py             # ノートブック生成
uv run --no-project python a100/score.py <run dir> [--partial]
```

## 6. はまりどころ

- Kaggle の L4×4 は**競技データの添付が必要**(ないと GPU 2枚)。A/B ノートブックでは正解ファイルがツールから読めないよう `qwen_ab.py` の FILE_GUARD(監査フック)で防ぎ、起動時に自己検査する
- Kaggle の Python 3.11 イメージは TensorFlow が入っていて vLLM のモデル検査サブプロセスが落ちる → `pip uninstall -y tensorflow` と `USE_TF=0`
- vLLM 0.19 は自作 wheelhouse(`koumeimaki/vllm019-cp311-cu128-wheelhouse`)から `--target /tmp/vllm_env` に入れ、`python -S` で起動。sklearn の numpy ABI 衝突はシムで回避
- イメージに curl がない(readiness は python urllib)。`.sh` は LF(`.gitattributes` 済み)
- `!` セルは1行で書く(行継続が効かない)
- 提出は採点待ちが長い(hybrid5 は約2日)。v9 では「Kaggle Error」表示の後に採点された例もある
- PowerShell で `git commit -F -` は使えない(stdin なし)。メッセージはファイルに書いて `-F <file>`
- Python のパッチスクリプトはバックスラッシュで事故りやすい。ファイル編集は Edit ツールで
