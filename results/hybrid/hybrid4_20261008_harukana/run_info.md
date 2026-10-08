# hybrid4 確認実行の結果

- 実行者: harukana
- ノートブック: [harukana/arc2-hybrid4-nvarc8-qwen38-long version 1](https://www.kaggle.com/code/harukana/arc2-hybrid4-nvarc8-qwen38-long/versions/1)
- 投入日時 (JST): 2026-10-08T20:34:35+09:00
- 開始: RUNNING 初回観測 2026-10-08T11:35:00.291179+00:00（UTC、実開始時刻はAPIから取得できず）
- 終了状態の観測日時 (JST): 2026-10-08T22:57:29+09:00（実終了時刻とは差がある）
- 最終ステータス: COMPLETE
- 消費した GPU クォータ: 7.00 時間（アカウント全体の差分。実行前 used 8.47h / remaining 21.53h、実行後 used 15.47h / remaining 14.53h。同時実行があればその消費も含む）
- ソースコミット: `bb4b4215445950f5cc7c0b6d255ba850d1f821dd`
- 実行設定: 公開評価8問、L4×4指定、Python 3.11イメージ指定、インターネット無効
- ノートブック SHA-256: `0e866432261a781d62c6d268d909ef83fca491cd587a1364b123986ea55739bc`
- 変更: kernel-metadata.json の所有者のみ harukana に変更
- この作業でコンペ提出は実施していません。

## ログ末尾の要約

### `NVARC done at`

```text
NVARC done at 0.32 h
```

### `vLLM ready:`

```text
[+ 1798.2s] vLLM ready: True | tool flags active: True
```

### `decode_tokens_per_second`

```text
 "decode_tokens_per_second": 144.0,
 "peak_decode_tokens_per_second": 309.8,
```

### `outputs changed by Qwen:`

```text
outputs changed by Qwen: 165
```

### `handoff tasks:`

```text
on 8 handoff tasks: NVARC 3.67  ->  hybrid 4.33
```

### `discarded forced/recovered answers:`

```text
discarded forced/recovered answers: 5, correct: 1 [('b6f77b65', 1)]
```

## 出力ファイル

- `arc2-hybrid4-nvarc8-qwen38-long.log`
- `vllm_server.log`
- `freeform_samples.jsonl`
- `freeform_receipt.json`
- `freeform_status.json`
- `nvarc_submission.json`
- `nvarc_confidence.json`
- `submission.json`
- `qwen_discarded.jsonl`

未生成・未取得: なし

依頼: https://github.com/Maki0924/kaggle-ARC-AGI-2/pull/25

## 補足

提出履歴APIでは依頼者の履歴と一致する7件が返りました。個人の提出回数とチームマージ状態はこのAPIだけでは確定していません。

## 回答ごとの終了理由とトークン数

| task_id | query_index | finish | completion_tokens |
| --- | --- | --- | --- |
| 4c7dc4dd | 1 | protocol_error:tool_call_markup_in_content | 52009 |
| 4c7dc4dd | 0 | stop | 89881 |
| c4d067a0 | 0 | context_budget | 87143 |
| 7b3084d4 | 0 | stop | 100196 |
| b6f77b65 | 1 | length+recovered | 91172 |
| b6f77b65 | 0 | context_budget | 96480 |
| 71e489b6 | 1 | length+recovered | 114221 |
| b6f77b65 | 2 | length+recovered | 109530 |
| 4c7dc4dd | 0 | protocol_error:tool_call_markup_in_content | 5193 |
| b6f77b65 | 0 | stop | 7492 |
| 71e489b6 | 1 | length+recovered | 20480 |
| 7b3084d4 | 0 | length+recovered | 20480 |
| b6f77b65 | 2 | stop | 23007 |
| c4d067a0 | 0 | stop | 25883 |
| 4c7dc4dd | 1 | stop | 43787 |
| b6f77b65 | 1 | stop | 47802 |
