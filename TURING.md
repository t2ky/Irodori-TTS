# Turing（RTX 2080Ti）向けの手当

このブランチは上流 [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の
`3822056`（main, 2026-11）を起点に、本番の GPU（RTX 2080Ti, sm75）で
**同時 8 本の文をバッチで回して 1 文 0.8 秒以内**に返すための手当を積む。

- ブランチ名は `turing-<上流の起点の短いハッシュ>`。上流を取り込むときは新しい起点で切り直し、手当を載せ直す
- 上流の挙動は既定のまま変えない（手当は引数・環境変数で有効にする）。1 本ずつ・fp32 では上流と同じ結果になること
- 手当ごとに下の一覧へ 1 行足す（何を・なぜ・どう確かめたか）

## 前提（2080Ti）

- bf16 は遅い（3090 でも fp32 の 3〜4 倍遅かった）。fp16 のテンソルコアを使う
- FlashAttention 2・INT4 量子化版（sm80 以上）は使えない
- TTS に使える VRAM は約 8.5GB（emotion-api と同居）

## 手当の一覧

| # | 内容 | 理由 | 確認 |
|---|---|---|---|
| 1 | `irodori_tts/turing.py` の `TuringBatcher`: パディングをバッチの最長（16 の倍数）に切る、空のキャプションは ModernBERT を通さない、話者の状態を参照ごとにキャッシュ、条件のエンコードを 1 回にして継続長とサンプラーで共有、復号は文ごと | `split_rare_tokens` の経路は本文 256・キャプション 512 トークンまでパディングし、DiT は毎ステップ約 1,000 個の（ほぼマスクされた）文脈キーを見ていた。上流のバッチは継続長のために文ごとにエンコードし、サンプラーでもう一度エンコードする | fp32 で 1 本ずつの上流と長さ 8/8 一致・SNR 64〜71dB（上流のバッチ #42 と同じ水準）。まとめて復号は速くならず（48kHz の畳み込みが律速）、長さの違う文を一緒にすると末尾ににじむので文ごとに戻した |
| 2 | 精度 `fp16`（`resolve_runtime_dtype`） | Turing は fp16 のテンソルコアがあり bf16 は遅い | RMSNorm・RoPE は元から fp32 で計算。NaN なし、長さ 8/8 一致、メルの差 平均 0.07〜0.11dB |
| 3 | 復号器だけ fp16（`decoder_fp16`）。参照音声のエンコードはコーデックの精度のまま | コーデックごと fp16 にすると参照の話者状態がずれて長さが 6/8 変わった。復号だけなら fp32 比 SNR 約 59dB で 1.5 倍速い | 上に同じ |
| 4 | `irodori_tts/cuda_graphs.py` の `GraphedDiT`（`cuda_graphs`）: 形ごとに DiT の forward を CUDA グラフに、入力の置き場は全グラフで共有、フックがあるときは通常実行 | 1 本ずつは起動律速（1 回 約 1,500 カーネル・16ms） | fp32 でグラフなしと同じ出力（SNR・メルの差が一致） |
| 5 | `get_timestep_embedding` の `log(10000)` を Python の定数に | 毎回デバイス上に tensor を作る（H2D コピー）のはグラフの捕獲中にできない | 同じ fp32 の値。出力の変化なし |
| 6 | `compile_decoder` / `compile_blocks`: 復号器と DiT の各ブロックを `torch.compile(dynamic=True)` | Snake・norm・AdaLN などの要素ごとの演算の融合。グラフの中で動く | 復号 4 秒ぶん 15.0 → 11.6ms。長さ 8/8 一致 |
| 7 | `synthesize_batch(..., length_check=True)`: length_check.py の 2 回生成をバッチで。最後のステップの呼び出しだけ通常実行（compile もしない）でポインタのフックを付け、全行のポインタを 1 回で取る。縮める行だけをまとめて 2 回目に回し、復号は採用した行だけ | 上流の length_check は 1 本ずつ・フック前提で、グラフや compile 済みのブロックではフックが呼ばれない | fp32 で上流（1 本ずつ）と判断が完全に一致（8 文中 3 文を同じフレーム数に縮める）、長さ 8/8 一致。12・16 ステップでも 40 と同じ行を縮めた |
| 8 | `watermark_fp16`: SilentCipher を fp16 の autocast で | 1 本 約 10ms が文ごとに直列。行をまとめても速くならない（スペクトログラムの畳み込みが律速、試して消した） | 8 本 226 → 72ms。読めたペイロードと確信度が fp32 と同じ |
| 9 | `graph_max_rows`（既定 8 行）・`speaker_multiple`・`latent_multiple` | グラフは形ごとなので、参照音声の長さ・潜在の長さ・行数の組み合わせで数が増える。大きいバッチは計算律速でグラフの効果が 2〜4% しかない | — |
| 10 | `turing_server.py`: 今のローカルサーバと同じ `/synth`・`/health` で、同時に来たリクエストを窓（既定 10ms・最大 8 本）でまとめる。声は `IRODORI_VOICES_DIR` に置いて名前（`voice`）で指定できる | 本番の形（1 プロセスに 1 版） | 下の負荷試験 |
| 11 | 声を参照潜在（`<名前>.pt`、`bench/make_latent.py`）で登録できる。`IRODORI_ALLOW_REF_AUDIO=0` でアップロードを断る | 参照音声のエンコード（fp32 のコーデック）は 30〜120 秒で約 1.4GB の一時確保。別のプロセスと GPU を分けるときにこれが入らない（vLLM と同居する GPU で cuBLAS の確保に失敗した） | wav で渡したときと長さ一致・波形の差 0〜0.0015 |
| 12 | 話者状態のキャッシュのキーに参照潜在のパスも入れる | 入れないと潜在だけのリクエストが全部同じキーになり、最初の声が使われる | キャッシュが声ごとに分かれる |
| 13 | 済んだ文から順に返す（`on_row`、`IRODORI_ROW_STREAM`）。短い文から復号・透かし | 遅い GPU では復号と透かしがバッチの大半（RTX 2000 Ada、8 本で 670ms のうち 510ms） | 同時 8 本の平均 0.70 → 0.42 秒（最大は同じ） |
| 14 | ウォームアップ後にグラフを凍結（`freeze()`、`IRODORI_GRAPH_FREEZE`）: 入力の置き場を広げず、新しい形は捕獲しない。compile も `eager_on_recompile`（`IRODORI_COMPILE_FREEZE`）。ウォームアップは行数 8〜1 × 長さのばらついた文 | 置き場を広げると捕獲済みのグラフが全部捨てられ（最長 34 字のウォームアップの後に長い文が 1 本来るだけで起きた）、捕獲 1 回は 0.5 秒前後の待ちになる | 下の Ada の節 |
| 15 | サーバの既定で cuDNN の attention を切る（`IRODORI_SDPA_CUDNN=0`、上流の docs/parameters.md の推奨） | sm80 以上では cuDNN が選ばれ、**新しい長さのたびに実行計画を作る**。1 回目だけ +0.7 秒（Turing では使われないので 2080Ti では出ない） | 6 字 0.81 → 0.15 秒、46 字 0.95 → 0.27 秒。2 回目の速さは同じ |
| 16 | `IRODORI_CUDA_MEMORY_MB`: PyTorch の確保の上限（`set_per_process_memory_fraction`）。`PYTORCH_ALLOC_CONF=expandable_segments:True` と組む | 長さがばらつくとキャッシュが伸び続ける（グラフなしで 5.3GB まで）。同居する vLLM の再起動には空きが要る | 4096 で起動後 3.5GB・負荷中の最大 4.1GB（プロセス全体）、速さは同じ、OOM なし |
| 17 | `graph_encoder`（`IRODORI_GRAPH_ENCODER`）: テキストエンコーダ（ModernBERT）を `cuda_graphs.GraphedCall` で形ごとにグラフに。`warm_encoder` が行数 1〜8 × トークン 16 刻み〜96（`IRODORI_ENCODER_GRAPH_TOKENS`、約 190 字）の 48 個を起動時に捕まえて凍結する。それより長い文は普通に実行。256 まで捕まえると出力の置き場で約 370MB 食い、確保の上限の中で長い返答がメモリ不足になった | 1 回に約 870 個の小さなカーネルで、行数によらず起動律速。遅い CPU では 1 行で 46ms（Xeon E5-2695 v2） | 出力は同じ（差 0.0）。3090 で 14.8 → 5.8ms、捕獲 9 秒 |
| 18 | `fast_watermark`（`IRODORI_FAST_WATERMARK`、`fast_watermark.py`）: SilentCipher の encode_wav を同期なしで書き直し。リサンプルのフィルタ・窓・メッセージを使い回し、無音の判定は GPU 上、復号した音声を CPU に戻さず渡す | 元は 1 行の中で何度も CPU が GPU を待つ（3090 で 1 行 27ms のうち GPU 10ms）。窓は毎回 CPU から転送、フィルタは毎回作り直し、既定の SDR で毎回 print | 元と完全一致（fp32・fp16 とも差 0.0、ペイロードも同じ）。リサンプルのフィルタは `functional.resample` と同じ作り方にしないと 1e-4 ずれる（`transforms.Resample` は CPU で作る） |
| 19 | 文ごとに返す処理の流れ作業化（`_finish_rows_pipelined`）: 行ごとに非同期で pinned メモリへ書き戻しを予約して次の行を投げ、書き戻しが済んだ行から返す。`sync_timings=False` で段の区切りの同期もやめる（サーバ） | 行ごと・段ごとに GPU の完了を待つと、その間 CPU が次を投げられない | 元の設定同士の揺れ（fp16、最大 0.00195）と同じ差 |
| 20 | グラフの捕獲の試し実行に使うストリームを 1 本に（`_side_stream`、GraphedDiT と GraphedCall の両方） | cuBLAS はストリームごとに作業領域を持つので、捕獲のたびに新しいストリームを作ると 1 回あたり約 6MB 漏れていた | テキストエンコーダ 48 個で +286MB → +30MB |
| 21 | `warm_dit`（サーバは `IRODORI_GRAPHS=1` で声が登録されているとき）: 拡散のグラフを行数 1〜4 × トークン 16 刻み〜64 × 潜在の長さ（`latent_multiple` 刻み）〜15 秒ぶんで網羅して捕まえ、凍結。全部の声を最長の声の長さにそろえる（`speaker_pad_to`）。グラフの上限は 512（`IRODORI_GRAPH_MAX_GRAPHS`） | 凍結後、ウォームアップの文と形が違う文はすべて普通に実行になっていた（2080 で拡散の呼び出しの 100%）。Turing では普通の実行がグラフの約 3 倍 | 2080 の返答 1 件 0.385 → 0.262 秒。389 個を 81 秒で捕獲（起動が長くなる） |

## 計測（ps00 の 3090、`bench/bench.py`、参照 34 秒、チャットの 8 文、透かしなし）

| | MF 1 本ずつ | MF 8 本 | Small 1 本ずつ | Small 8 本 |
|---|---|---|---|---|
| 上流（fp32） | 0.171 | 1.052 | 0.842 | 5.83 |
| 1 | 0.106 | 0.465 | 0.725 | 4.59 |
| 1〜3（fp16） | 0.108 | 0.254 | 0.718 | 1.53 |
| 1〜4 | 0.086 | 0.248 | 0.335 | 1.47 |
| 1〜6 | **0.070** | **0.199** | **0.246** | **1.03** |

秒。8 本は 1 回の呼び出しの時間（全部の文がそろうまで）。透かし（SilentCipher）を入れると 8 本で +約 100ms（1 本 約 10〜13ms、文ごとに直列）。

```bash
bench/run.sh --model mf --precision fp16 --decoder-fp16 --graphs --compile-decoder --compile-blocks \
  --fast --batch 1,8 --stages --compare bench/out/mf_fp32_b1.npz   # 基準は --save で先に作る
```

長さ補正・透かしを入れた本番相当（Small）:

| Small・8 本（fp16・手当 1〜8・長さ補正あり、透かしなし） | 8 本で 1 回 |
|---|---|
| 40 ステップ | 1.35 |
| 16 ステップ・sway −1 | 0.74 |
| 12 ステップ・sway −1 | 0.60 |
| 12 ステップ | 0.57 |

## 品質の目安（`bench/eval_quality.py`、8 文 × 3 シード）

faster-whisper large-v3 の CER と、WavLM-SV（microsoft/wavlm-base-plus-sv）の参照音声とのコサイン類似度。
「うん。」は長さ補正なしだと必ず余計な発話が付く（CER 100〜350%）ので除いて比べた。

- fp16 の高速経路は上流の fp32 と同等（Small 40: CER 2.5% 対 2.7%、類似度 0.896 対 0.896）
- ステップ数・sway・CFG の掛け方を変えても CER は 2.5〜2.7% で差がない。類似度は 40: 0.896、24: 0.891、16: 0.890、12: 0.883、16 sway: 0.893、12 sway: 0.896、CFG t≥0.7: 0.885、alternating: 0.848。短い 2 文を除くとすべて 0.92〜0.93 で、この指標では見分けられない（同じ話者なら 0.91〜0.95 に集まる）。**決めるのは耳**
- 長さ補正で「うん。」を含む全体の CER が 29% → 2.2%

## 負荷試験（`turing_server.py` ＋ `bench/load.py`、3090、透かしあり、声は名前で指定）

| | 同時 8 本 平均／最大 | 1 本 | ポアソン |
|---|---|---|---|
| MF | 0.316 ／ 0.492 | 0.087 | 8/s: p95 0.202、16/s: p95 0.359 |
| Small 40 ステップ | 1.574 ／ 1.765 | 0.322 | 4/s: p95 2.30、8/s: 処理が追いつかない |
| Small 16 ステップ・sway | 0.938 ／ 1.142 | 0.208 | 4/s: p95 0.62、8/s: p95 1.85 |

秒。リクエストごとに参照音声（34 秒・base64 で約 3MB）を送ると、同時 8 本が 0.49 秒・バッチが平均 6.2 本に崩れた（解析がイベントループで直列）。本番は声を登録して名前で指定する。

```bash
IRODORI_MODEL=small-mf IRODORI_PORT=8099 IRODORI_VOICES_DIR=<dir> bench/serve.sh   # 止めるのは bench/stop_server.sh 8099
python3 bench/load.py --voice <name> burst --n 8 --rounds 10
```

## 罠

- `pkill -f <スクリプト名>` は呼んだシェル自身のコマンド行にも一致して、シェルごと落ちる。止めるのはポートから PID を引く `bench/stop_server.sh`
- コーデックを fp16 にすると参照音声のエンコードも fp16 になり、話者状態がずれて長さが変わる。fp16 にするのは復号器だけ
- compile 済みのブロックは attention を取り込むので、あとから attention に付けたフックが呼ばれない
- CUDA グラフの入力の置き場を広げると捕獲済みのグラフは無効になり、空になったプールにはもう捕獲できない（新しいプールを作る）。温めは大きい形から
- 透かしを入れていない音声でも SilentCipher の検出が「あり」になることがある（2/8）。短い音声では fp32 でも 1 バイト目の読み違い・検出漏れがある。上流の性質

## Ada（RTX 2000 Ada、16GB・70W・224GB/s、CPU は i7-7700）

この GPU には vLLM（8B、`--gpu-memory-utilization=0.625`）が同居する。vLLM は起動時に 10.2GB の空きを要求するので、
**TTS はプロセス全体で 6.1GB を超えてはいけない**。GPU を要求できない（`nvidia.com/gpu` は vLLM が取っている）ので、
GPU を要求せずに nvidia ランタイムで同じ GPU を使う。

推奨: MF・fp16・`IRODORI_GRAPHS=0`・compile あり・`IRODORI_SDPA_CUDNN=0`・`IRODORI_CUDA_MEMORY_MB=4096`・
`PYTORCH_ALLOC_CONF=expandable_segments:True`・声は参照潜在・`IRODORI_ALLOW_REF_AUDIO=0`。

- グラフはこの GPU ではほぼ効かない（1 本 0.114 → 0.116 秒）。形の組み合わせ（行数 × 潜在の長さ × トークン数）が多く、捕獲しても当たらない。切ると VRAM が減る
- 復号（DACVAE）は 4 秒ぶん 36ms（compile あり）で、畳み込みの計算で律速。cudnn.benchmark は効かない
- bf16 は fp16 と同じ速さの見込み（試していない）。fp16 で NaN が出ないのでそのまま

返答まるごと（`bench/load.py replies`、`bench/replies.json` の 12 返答＝平均 2.7 文・音声 8.8 秒、文を同時に送り最後の文が返るまで。透かし・ウォームアップ済み）:

| | 平均 | p95 | 最大 |
|---|---|---|---|
| MF・返答 1 件ずつ | 0.25〜0.30 | 0.31〜0.36 | 0.37 |
| MF・同時 4 件 | 0.61〜0.63 | 0.94〜0.96 | 0.97 |
| MF・文を毎秒 3 本（`texts_long.json`、最大 131 字） | 0.34 | 0.68 | 0.93 |
| Small 40 ステップ・1 件 / 4 件 | 1.54 / 4.03 | 1.95 / 6.03 | 1.95 / 6.12 |
| Small 16 ステップ・sway −1・1 件 / 4 件 | 0.82 / 2.22 | 1.08 / 3.16 | 1.08 / 3.24 |

**vLLM と同時に負荷がかかると時分割で互いに遅くなる**（MPS なし）: vLLM の 1 トークン 24 → 33ms（TTS に毎秒 3 本）、
TTS の返答 1 件 0.26 → 0.97 秒・同時 4 件 0.62 → 2.85 秒（vLLM に同時 4 本を流し続けたとき）。TTS が待っているだけなら vLLM は遅くならない。

```bash
IRODORI_MODEL=small-mf IRODORI_VOICES_DIR=<dir> IRODORI_GRAPHS=0 IRODORI_CUDA_MEMORY_MB=4096 \
  PYTORCH_ALLOC_CONF=expandable_segments:True IRODORI_ALLOW_REF_AUDIO=0 python turing_server.py
python3 bench/load.py --url http://127.0.0.1:8097 --voice <name> replies --replies bench/replies.json --concurrent 4
```

- torch 2.10 の inductor のキャッシュは別のプロセスから読めない（`DefaultsSource.__init__()` の TypeError）。起動のたびにコンパイルし直す（起動 11 秒＋ウォームアップ 30 秒）

## 遅い CPU 向けの手当（17〜21）の効果（返答まるごと、`bench/load.py replies`、秒）

| | 返答 1 件 | 同時 4 件 | 文を毎秒 3 本（長め） |
|---|---|---|---|
| RTX 2080（Xeon E5-2695 v2）・グラフあり、手当の前 | 0.441 | 0.963 | 0.616 |
| 同・17〜20 | 0.385 | 0.853 | 0.524 |
| 同・17〜21 | **0.262** | **0.772** | **0.471** |
| 同・17〜21 ＋ `IRODORI_CUDA_MEMORY_MB=4608` | 0.263 | 0.836 | 0.474 |
| RTX 2000 Ada（i7-7700）・グラフなし、手当の前 | 0.255 | 0.625 | 0.335 |
| 同・17〜20 | 0.238 | 0.609 | 0.332 |

- GPU の計算（復号・拡散・透かし）はどちらのカードもほぼ同じで、差は CPU（小さなカーネルの投げ込みと同期）だった
- 2080 の VRAM（17〜21、上限なし）: 起動後 5.1GB、負荷後 7.3GB。上限 4608 で負荷後 6.6GB
- 起動: 2080 で 134 秒（拡散のグラフ 81 秒・テキストエンコーダ 4 秒）。Ada はグラフなしなので 30〜40 秒
