# hybrid5 確認実行の結果

- 実行者: harukana
- ノートブック: [harukana/arc2-hybrid5-nvarc-qwen38-medium version 1](https://www.kaggle.com/code/harukana/arc2-hybrid5-nvarc-qwen38-medium/versions/1)
- 投入日時 (JST): 2026-10-09T01:08頃+09:00（CLI投入時刻）
- 開始: RUNNING 初回観測 2026-10-08T16:15:00.649003+00:00（UTC、実開始時刻はAPIから取得できず）
- 終了状態の観測日時 (JST): 2026-10-09T02:25:04+09:00（実終了時刻とは差がある）
- 最終ステータス: COMPLETE
- 消費した GPU クォータ: 3.62 時間（アカウント全体の差分。実行前 used 17.63h / remaining 12.37h、実行後 used 21.25h / remaining 8.75h。同時実行があればその消費も含む）
- ソースコミット: `cfa555e18acf058a0dce90348e3707213f01f512`
- 実行設定: 公開評価8問、L4×4指定、Python 3.11イメージ指定、インターネット無効
- ノートブック SHA-256: `d632c36fcb6477df8771748b6b3b89fce9e896b97c6ea570407de8372b19c2aa`
- 変更: kernel-metadata.json の所有者のみ harukana に変更
- この作業でコンペ提出は実施していません。

## ログ末尾の要約

### `NVARC done at`

```text
NVARC done at 0.46 h
```

### `vLLM ready:`

```text
[+ 2362.5s] vLLM ready: True | tool flags active: True
```

### `decode_tokens_per_second`

```text
 "decode_tokens_per_second": 135.1,
 "peak_decode_tokens_per_second": 232.4,
```

### `outputs changed by Qwen:`

```text
outputs changed by Qwen: 161
```

### `handoff tasks:`

```text
on 8 handoff tasks: NVARC 2.67  ->  hybrid 3.67
```

### `discarded forced/recovered answers:`

```text
discarded forced/recovered answers: 3, correct: 0 []
```

## 出力ファイル

- `arc2-hybrid5-nvarc-qwen38-medium.log`
- `vllm_server.log`
- `freeform_samples.jsonl`
- `freeform_receipt.json`
- `freeform_status.json`
- `nvarc_submission.json`
- `nvarc_confidence.json`
- `submission.json`
- `qwen_discarded.jsonl`

未生成・未取得: なし

依頼: https://github.com/Maki0924/kaggle-ARC-AGI-2/pull/31

## 補足

提出履歴APIでは依頼者の履歴と一致する7件が返りました。個人の提出回数とチームマージ状態はこのAPIだけでは確定していません。

## 回答ごとの終了理由とトークン数

| task_id | query_index | finish | completion_tokens |
| --- | --- | --- | --- |
| c4d067a0 | 0 | stop | 22738 |
| b6f77b65 | 0 | stop | 28885 |
| 7b3084d4 | 0 | length+recovered | 32768 |
| b6f77b65 | 2 | protocol_error:tool_call_markup_in_content | 46817 |
| b6f77b65 | 1 | context_budget | 50833 |
| 71e489b6 | 1 | length+recovered | 54258 |
