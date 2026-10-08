# 依頼4: hybrid5 の確認実行

hybrid4 の結果から、Qwen は推論強度 medium のほうが短時間で考え終わり、新しい正解も medium から出ることが分かりました。そこで、Qwen の1回目のパスを medium にした版(hybrid5)を作りました。これを **1本** お願いします。手順と結果の共有形式は `docs/TEAM_REQUEST.md` と同じで、違うのはフォルダ名だけです。

**先に `git pull` で最新の `main` を取ってください**(PR #30 で追加されています)。

## お願いする実行

| フォルダ | 中身 | 見たいこと |
|---|---|---|
| `kaggle/arc2-hybrid5-nvarc-qwen38-medium` | NVARC は今までどおり。Qwen の1回目のパスを推論強度 medium・最大3.2万トークン・30分・同時16に。強制確定の答えは採用しない | Qwen が考え終わる出力の数、合流前後の正解数、NVARC の正解を失っていないか |

GPU クォータを約2.2時間使います。

## 投入

```bash
git pull
cd kaggle/arc2-hybrid5-nvarc-qwen38-medium
# kernel-metadata.json の "id" を "<ユーザー名>/arc2-hybrid5-nvarc-qwen38-medium" に書き換え
kaggle kernels push -p .
```

## 結果の共有

`docs/TEAM_REQUEST.md` の「4. 結果の共有」と同じ形式で PR にしてください。

- 置き場所: `results/hybrid/hybrid5_<YYYYMMDD>_<ユーザー名>/`
- ノートブック全体のログのファイル名は `arc2-hybrid5-nvarc-qwen38-medium.log`
- `qwen_discarded.jsonl` もあれば入れてください

今回もコンペへの提出はしないでください。
