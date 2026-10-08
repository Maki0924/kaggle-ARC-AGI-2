# hybrid2 確認実行の結果

- 実行者: harukana
- ノートブック: [harukana/arc2-hybrid2-nvarc-qwen38-hint version 1](https://www.kaggle.com/code/harukana/arc2-hybrid2-nvarc-qwen38-hint/versions/1)
- 投入日時 (JST): 2026-10-08T18:20:12+09:00
- 開始: RUNNING 初回観測 2026-10-08T09:20:55.002141+00:00（UTC、実開始時刻はAPIから取得できず）
- 終了状態の観測日時 (JST): 2026-10-08T19:28:13+09:00（実終了時刻とは差がある）
- 最終ステータス: COMPLETE
- 消費した GPU クォータ: 2.20 時間（アカウント全体の差分。実行前 used 6.27h / remaining 23.73h、実行後 used 8.47h / remaining 21.53h。同時実行があればその消費も含む）
- ソースコミット: `9315639cc3f604ff8858464ca2be4c22f4eb74c8`
- 実行設定: 公開評価8問、L4×4指定、Python 3.11イメージ指定、インターネット無効
- ノートブック SHA-256: `5a6562427efe776ab6e4079643fed65ab3f5f961e6fd6c5f0190fe8392249222`
- 変更: kernel-metadata.json の所有者のみ harukana に変更
- この作業でコンペ提出は実施していません。

## ログ末尾の要約

### `NVARC done at`

```text
NVARC done at 0.37 h
```

### `vLLM ready:`

```text
[+ 1956.0s] vLLM ready: True | tool flags active: True
```

### `decode_tokens_per_second`

```text
 "decode_tokens_per_second": 187.2,
 "peak_decode_tokens_per_second": 316.9,
```

### `outputs changed by Qwen:`

```text
outputs changed by Qwen: 160
```

### `handoff tasks:`

```text
on 8 handoff tasks: NVARC 3.67  ->  hybrid 3.67
```

### `discarded forced/recovered answers:`

```text
discarded forced/recovered answers: 5, correct: 1 [('8b9c3697', 0)]
```

## 出力ファイル

- `arc2-hybrid2-nvarc-qwen38-hint.log`
- `vllm_server.log`
- `freeform_samples.jsonl`
- `freeform_receipt.json`
- `freeform_status.json`
- `nvarc_submission.json`
- `nvarc_confidence.json`
- `submission.json`
- `qwen_discarded.jsonl`

未生成・未取得: なし

依頼: https://github.com/Maki0924/kaggle-ARC-AGI-2/pull/22

## 補足

提出履歴APIでは依頼者の履歴と一致する7件が返りました。個人の提出回数とチームマージ状態はこのAPIだけでは確定していません。
