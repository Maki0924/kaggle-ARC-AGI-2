# 依頼2: hybrid2 と hybrid1 再実行の確認実行

前回(PR #20)の確認実行ありがとうございました。fp8 KV が起動し、8問で NVARC 3.33 → ハイブリッド 4.33 になりました。続けて2本お願いします。手順と結果の共有形式は `docs/TEAM_REQUEST.md` と同じで、違うのはフォルダ名だけです。

**先に `git pull` で最新の `main` を取ってください**(PR #21 で両方のノートブックが更新されています)。

## お願いする実行

| 順 | フォルダ | 中身 | 見たいこと |
|---|---|---|---|
| 1 | `kaggle/arc2-hybrid2-nvarc-qwen38-hint` | hybrid1 に加えて、Qwen のプロンプトに NVARC の候補グリッドを付け「確かめて、選ぶか直すか自分で答えて」と頼む版 | 合流前後の正解数が hybrid1(3.33 → 4.33)より伸びるか、Qwen が考え終わる割合が増えるか |
| 2 | `kaggle/arc2-hybrid1-nvarc-qwen38` | 前回と同じ版(最新版では、捨てた回収回答を記録して採点する処理を追加) | 打ち切りから回収した答えの正解率(ログ末尾の `discarded forced/recovered answers: N, correct: M`) |

1本あたり GPU クォータを約2.2時間使います(2本で約4.5時間)。**2本は同時に投入して構いません**(同時実行の上限は2本)。

## 投入

```bash
git pull
cd kaggle/arc2-hybrid2-nvarc-qwen38-hint
# kernel-metadata.json の "id" を "<ユーザー名>/arc2-hybrid2-nvarc-qwen38-hint" に書き換え
kaggle kernels push -p .
cd ../arc2-hybrid1-nvarc-qwen38
# "id" を "<ユーザー名>/arc2-hybrid1-nvarc-qwen38" に書き換え(前回と同じ。バージョン2になります)
kaggle kernels push -p .
```

## 結果の共有

`docs/TEAM_REQUEST.md` の「4. 結果の共有」と同じ形式で、1本につき1つの PR にしてください。

- hybrid2: `results/hybrid/hybrid2_<YYYYMMDD>_<ユーザー名>/`、ノートブック全体のログのファイル名は `arc2-hybrid2-nvarc-qwen38-hint.log`
- hybrid1: `results/hybrid/hybrid1_<YYYYMMDD>_<ユーザー名>_v2/`
- 新しく出力される `qwen_discarded.jsonl`(捨てた回収回答の記録)も入れてください
- `run_info.md` のログ要約に、`discarded forced/recovered answers: ...` の行も追加してください

今回もコンペへの提出はしないでください。
