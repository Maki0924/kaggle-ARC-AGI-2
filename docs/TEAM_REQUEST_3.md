# 依頼3: hybrid4 の確認実行

hybrid1・hybrid2 の確認実行ありがとうございました。分かったのは「Qwen が NVARC の解けない難問を 5.7万トークン・1時間以内に考え終わらない」ことです。そこで、Qwen に長く考えさせる版(hybrid4)を作りました。これを **1本** お願いします。手順と結果の共有形式は `docs/TEAM_REQUEST.md` と同じで、違うのはフォルダ名だけです。

**先に `git pull` で最新の `main` を取ってください。**

## お願いする実行

| フォルダ | 中身 | 見たいこと |
|---|---|---|
| `kaggle/arc2-hybrid4-nvarc8-qwen38-long` | NVARC を8ビューにして時間を空け、Qwen は1出力あたり最大12万トークン・2時間(コンテキスト131k、同時8) | Qwen が考え終わる出力が増えるか(前回は8件中0〜1件)、合流前後の正解数 |

**GPU クォータを約6時間使います**(実時間で最大約3時間、L4×4 は2倍消費)。長く考えさせる分、これまでより重い実行です。残りが6.5時間未満なら投入しないでください。

## 投入

```bash
git pull
cd kaggle/arc2-hybrid4-nvarc8-qwen38-long
# kernel-metadata.json の "id" を "<ユーザー名>/arc2-hybrid4-nvarc8-qwen38-long" に書き換え
kaggle kernels push -p .
```

## 結果の共有

`docs/TEAM_REQUEST.md` の「4. 結果の共有」と同じ形式で PR にしてください。

- 置き場所: `results/hybrid/hybrid4_<YYYYMMDD>_<ユーザー名>/`
- ノートブック全体のログのファイル名は `arc2-hybrid4-nvarc8-qwen38-long.log`
- `qwen_discarded.jsonl` もあれば入れてください
- `run_info.md` のログ要約に、`discarded forced/recovered answers: ...` の行と、`freeform_samples.jsonl` の各行の `finish` と `completion_tokens` の一覧を追加してください

今回もコンペへの提出はしないでください。

## 補足

ここ2日ほど、Kaggle で多くのチームの提出がキュー待ちの末にエラーになっています(ディスカッション 747088)。確認実行(コミット)も待ちが長くなる可能性があります。
