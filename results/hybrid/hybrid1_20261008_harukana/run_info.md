# hybrid1 確認実行の結果

- 実行者: harukana
- ノートブック: [harukana/arc2-hybrid1-nvarc-qwen38](https://www.kaggle.com/code/harukana/arc2-hybrid1-nvarc-qwen38) version 1
- 投入日時 (JST): 2026-10-08 16:27:33
- 開始 / 終了日時: 下記ログ時刻参照。COMPLETE 初回観測は 2026-10-08 17:34:12 JST
- 最終ステータス: COMPLETE
- 消費した GPU クォータ: 2.18 時間（used 4.09h → 6.27h、remaining 25.91h → 23.73h。アカウント全体の差分なので同時実行があれば含む）
- ソースコミット: `788e025d331326edb6003efc96c430aa2e59f363`
- 変更: kernel-metadata.json の id の所有者のみ harukana に変更
- 確認環境: NVIDIA L4 ×4、Python 3.11.13、vLLM 0.19.0、インターネット無効
- コンペへの提出は実施していません。

## 結果

| 確認項目 | 結果 |
| --- | --- |
| fp8 KV + TRITON_ATTN の起動 | 成功。サーバーログに設定・バックエンドと ready=True を確認 |
| 公開8問の評価 | NVARC 3.33 → hybrid 4.33（複数出力の問題は出力ごとに按分） |
| 生成速度 | 平均166.4 tok/s、ピーク281.5 tok/s |
| 同時実行 | 設定16、実測ピーク7・平均5.6。16並列を埋めた速度比較にはなっていない |
| Qwen の有効回答 | 7記録中1件。c4d067a0 の query 0、自然終了 stop、attempt_2 に採用 |
| 打ち切り回収 | length+recovered は5件、全件 grid=false。今回の回収から有効グリッドは得られず |
| その他の終了 | context_budget が1件、grid=false |

## ログ末尾の要約

```text
NVARC done at 0.35 h
[+ 1887.3s] vLLM ready: True | tool flags active: True
 "decode_tokens_per_second": 166.4,
 "peak_decode_tokens_per_second": 281.5,
outputs changed by Qwen: 161
on 8 handoff tasks: NVARC 3.33  ->  hybrid 4.33
```

## 解釈上の注意

- `outputs changed by Qwen: 161` はQwenが161回答を生成した意味ではありません。NVARCと最終出力の差分を再集計すると、今回の対象8問内の変更は `c4d067a0` の1出力だけでした。残る160変更は対象外タスクです。receiptも `published_model_grids=1` と記録しています。
- 8問の結果だけで本番の全問性能・完走は保証できません。今回の成果物は公開評価の診断用です。
- 回収時、過去実行フォルダ内にも `submission.json` があり、自動監視が曖昧な選択を避けて停止しました。今回のルート直下の8ファイルだけを選び、過去実行・コード・キャッシュは除外して回収しました。
- 提出履歴APIでは依頼者の履歴と一致する7件が返りました。harukana個人の提出回数やチームマージ状態は、このAPIだけでは確定していません。

## 時刻と再現情報

- receipt `run_started_utc`: `2026-10-08T07:48:45.335056+00:00`
- receipt `run_finished_utc`: `2026-10-08T08:32:53.271072+00:00`
- ノートブック先頭セルの開始時刻: 2026-10-08T16:27:57.293177+09:00
- ノートブック SHA-256: `8591c52af0ca5078dd0b7f756920ecc2f0dbe692be08fd9be82b4c36195b394f`
