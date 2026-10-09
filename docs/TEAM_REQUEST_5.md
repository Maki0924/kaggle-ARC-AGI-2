# 依頼5: hybrid5 の本番提出

hybrid5 の確認実行(PR #32)で、NVARC の正解を失わずに上積みが出ることを確認できました。この版を **本番(隠しテスト)に提出** してください。harukana さんのアカウントにある完走済みの version 1 から提出すれば、確認実行をやり直す必要はありません。

## いつ出すか

**チームの前の提出の採点が終わってから**出してください。Kaggle の新しい制限で、チームとして待機・実行中にできる提出は1本までです。

- 確認方法: `kaggle competitions submissions arc-prize-2026-arc-agi-2` で、一番上の行が `PENDING` でなく `COMPLETE` または `ERROR` になっていること
- 提出はチームで1日1回(UTC 0時でリセット)。その日の枠が残っているかは `kaggle competitions submission-limits arc-prize-2026-arc-agi-2` の `Remaining today` で確認

こちらでも前の提出の採点が終わったら知らせます。

## 提出方法(どちらか)

**画面から**

1. https://www.kaggle.com/code/harukana/arc2-hybrid5-nvarc-qwen38-medium を開く
2. 右上のバージョン選択で **version 1** を選ぶ
3. 「Submit to Competition」を押す。説明欄には `hybrid5: NVARC + Qwen3.8 medium (C=16, forced_final excluded)` と入れてください

**コマンドから**

```bash
kaggle competitions submit arc-prize-2026-arc-agi-2 \
  -k harukana/arc2-hybrid5-nvarc-qwen38-medium -v 1 -f submission.json \
  -m "hybrid5: NVARC + Qwen3.8 medium (C=16, forced_final excluded)"
```

## 提出後

- 本番は12時間近くかかります(キュー待ちは別)。採点の結果はこちらで確認するので、提出した日時だけこの PR にコメントしてください
- 提出は1回だけにしてください(重複して出さない)
