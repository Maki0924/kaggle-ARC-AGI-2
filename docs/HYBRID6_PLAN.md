# hybrid6: 64k推論と出力単位の対象選択

作成: 2026-10-13。確認12問の結果を見る前の設定・採用基準。

## 目的と現状

NVARCの固定出力にQwenを加え、12時間の本番制限内で純増を得る。
探索36問の結果を本番の改善点数に換算しない。公開評価はNVARCの学習データに含まれ、未見評価ではない。

層の定義は `a100/hybrid_policy.py` と `scripts/build_fixed_nvarc.py` に従う。
- Z: 候補ゼロ。
- O: 候補1つ。
- W: 候補2つ以上で第2候補の票が2以下。
- strong: 上記以外。

コホートのラベルはタスク単位（いずれかの出力にZがあればZ、次にO、次にW）。
本番の対象順序・新しい集計は出力単位。同じタスクでも出力の層は異なる。
HANDOFFの旧記載「候補1つ(W)」を訂正した。

## 実装した候補

| 項目 | 設定 |
|---|---|
| バリアント | hybrid6 = xhigh、hybrid6-medium = medium |
| NVARC | fast1、7.5時間（変更なし） |
| Qwen | FP8、vLLM 0.19、TP4、MTP、FP8 KV、TRITON_ATTN |
| 予算 | 全ターン合計65,536 completion tokens、context 131,072 |
| prompt計数 | サーバーの /tokenize、chat template・tools・effort込み |
| ターン | 最大64。強制確定・打ち切り回収は行わない |
| 同時数 | モデル16、Pythonツール8 |
| 対象順序 | 出力ごとW→O→Z、その中はv2昇順→v1昇順。IDは同順位時の固定順のみ |
| サンプル | 出力ごと1回。第2パスを自動実行しない |
| 時間制御 | notebook開始から11.25時間の既存hard wall。各出力のepisodeは最大3,600秒。既存の予測的admissionを利用 |
| 対象数 | 全対象を優先キューに置き、時間内に開始できるものだけ実行。キュー全件の完了を約束するための予算短縮は廃止 |
| 合流 | 自然なstopかつ強制確定・回収なし。有効な最終グリッドだけ採用 |
| 候補なし判定 | 実際の候補数n=0だけ。候補として生成された [[0]] は有効な答えとして保持 |
| 障害 | ツールが起動できなければNVARCを維持し、direct推論へ黙って切り替えない |
| 記録 | 従来receiptに加え qwen64_samples/ にターンごとの予算・応答・最終グリッドを保存 |

A/Bランナーは時間上限なし、本番は3,600秒/出力と全体wallあり。この差はL4の検証対象。
1時間で64kを完了できることはまだ保証しない。mediumの方が遅い場合は処理量も含めて比較する。
診断実行は従来の8問。正解率比較ではなく起動・構文・ツール・予算・保存の確認に限る。

## 保存済み結果のCPU再評価

入力: `results/qwen_ab/20261010/runs/`。36タスク、対象44出力、各1サンプル。
再評価: `results/qwen_ab/20261013_replay/`。歴史的なscores.jsonは変更しない。

| 合流・対象 | medium 64k（追加 / 損失 / 純増） | xhigh 64k（追加 / 損失 / 純増） |
|---|---|---|
| strict、全44出力 | 7.0 / 1.0 / +6.0 | 3.5 / 0 / +3.5 |
| strict、第2候補への上書きをv2≤1に限定 | 4.5 / 1.0 / +3.5 | 2.5 / 0 / +2.5 |
| strict、W→O→Z順の先頭32出力のみ | 5.5 / 1.0 / +4.5 | 2.5 / 0 / +2.5 |

出力単位の純増: mediumはW +5.5 / O +0.5 / Z 0、xhighはW +3.5 / O 0 / Z 0。
従来のタスク単位ラベルによるmediumのW +5.0 / O +0.5 / Z +0.5とは定義が異なる。
v2≤1は損失を防げず利益を減らすため棄却。本番は従来のv2≤2を維持する。
先頭32件の数字は保存済み回答を間引いた反実仮想であり、順序変更・バッチ変更後の実測や所要時間の保証ではない。
同時実行されたsample_wall_sの合計を経過時間として扱わない。

## 確認12問と採用基準

1. 既存の `kaggle/arc2-qwen-ab-l4-confirm` でxhigh64k→medium64k、各2サンプル。GPU時間の見込みは約8時間、L4×4のクォータ消費はその2倍。起動前に既存実行・残クォータを確認。
2. 条件の比較はまず `--samples 1 --policy strict`。第2サンプル込みは独立した追加予算として報告する。
3. 確認用採用ゲート: +1タスク相当以上、損失0。小標本での実務基準であり統計的保証ではない。
4. 両条件が通れば、L4で同じ総時間を与えた純増を優先。差が1タスク相当未満なら速く単純な方。
5. mediumだけ利益が大きく損失も出る場合、既存の保守的候補を維持する。確認12問を見て閾値を都合よく調整しない。
6. 両方落ちた場合、hybrid6は本番候補に昇格しない。既存の採点済み候補を維持し、失敗原因を先に確認する。
7. L4では全ターン合計予算、prompt計数、終了理由、時間内に完了した出力数、純増・損失、OOM・保存を確認。GPU検証前に本番へ提出しない。

## コマンド

```powershell
uv run --no-project python scripts/build_hybrid.py hybrid6
uv run --no-project python scripts/build_hybrid.py hybrid6-medium
uv run --no-project python scripts/build_qwen_ab_notebook.py confirm
uv run --no-project python -m unittest discover -s tests -v

# 既存実行・認証・クォータ確認後に投入（実際にGPUクォータを使う）
uvx kaggle kernels status koumeimaki/arc2-qwen-ab-l4-confirm
uvx kaggle kernels push -p kaggle/arc2-qwen-ab-l4-confirm
uvx kaggle kernels output koumeimaki/arc2-qwen-ab-l4-confirm -p results/qwen_ab/confirm_20261013

# <run> は取得した各条件のディレクトリ。全サンプルが揃わないと採点は拒否される
uv run --no-project python a100/score.py <run> --samples 1 --policy strict --report-dir results/qwen_ab/confirm_20261013/strict_s1
uv run --no-project python a100/score.py <run> --policy strict --report-dir results/qwen_ab/confirm_20261013/strict_s2
```

10/13時点: 11件のCPUテストが通過。Kaggle APIの認証が利用できず、確認実行の状態も取得できていない。今回のGPU実行・本番提出は未実施。
`qwen_ab.py` の計数・終了理由チェックを強化したため、旧コードの途中実行を新コードで再開しない（manifestの不一致を正しく拒否する）。

## 締切まで

- 10/13〜16: 確認12問を完了、条件を固定。
- 10/16〜19: L4の統合試験と時間内処理量確認。
- 10/22まで: 採点済みの本番候補2本を確保。採点待ちがあるため毎日の再提出を前提にしない。
- 10/27以降: 構成凍結。最終提出は11/2 23:59 UTC。

## 参照した公開情報

- [NVARC公式writeup](https://www.kaggle.com/c/arc-prize-2025/writeups/nvarc): 合成データによる事前学習。
- [Qwen3.8公式評価](https://arcprize.org/results/alibaba-qwen3-8-27b): effortで成績・指示が異なる。手元のツール付き評価とは条件が異なる。
- [公開の再現実験記録](https://github.com/jaywang172/ARC-Prize-2026---ARC-AGI-2/blob/main/research/ARC_RESEARCH_2026-08-29.md): 独自正解があってもlabel-freeな合流で利益が出るとは限らない。
- Kaggleの公開ノートブック・Discussionについては10/9の保存済み一覧と既存調査記録を参照。今回の最新本文取得はAPI認証とWeb取得の制約により完了していない。
