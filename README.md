# KP

Kimi K3 を動かすための **完全自作の推論エンジン**（自作CUDAカーネル）。
外部クレートは使わず、JSON・設定・（今後）safetensors・tokenizer・CUDAドライバ呼び出しまで自前で書く。

## 現状

- [x] `src/json.rs` 自作JSONパーサ
- [x] `src/config.rs` K3 の config.json を型付きで読む（KDA/MLA・dense/MoE のレイヤー表、MXFP4量子化情報）
- [x] `kp inspect <config.json>` 構造とroutedエキスパートのサイズを表示

```
cargo test
cargo run -- inspect models/kimi-k3/config.json
```

`models/` は git 管理外。`moonshotai/Kimi-K3` の `config.json` を `models/kimi-k3/` に置く。

## Kimi K3 の構造（config.json と Hugging Face メタデータから）

| 項目 | 値 |
|---|---|
| 層数 | 93（KDA 69 + Gated MLA 24、先頭1層だけ dense FFN） |
| hidden / vocab / context | 7168 / 163,840 / 1,048,576 |
| routed experts | 896個、トークンあたり16個。LatentMoE（7168→3584 に落として 3072 の中間次元） |
| shared experts | 2個 |
| router | sigmoid + noaux_tc（top-k、renormalize） |
| routed experts の格納形式 | MXFP4（4bit + グループ32ごとに1バイトのスケール） |
| それ以外 | BF16（アテンション・shared experts・dense・lm_head など約57B） |

- routed experts の総数は `3 × 3584 × 3072 × 896 × 92 = 2,722,740,830,208`。HF の safetensors メタデータの U8 要素数と完全に一致する（`config::tests` で検査）。
- チェックポイント全体は safetensors 96個、約1453.7 GiB。
- トークンあたりに読む routed expert は約24.06 GiB（MXFP4）。これに always-active の BF16 部分（約55B）が加わる。

## ロードマップ

1. [ ] safetensors リーダ（ヘッダ解析、Windows の `CreateFileMapping` を直叩きした mmap）
2. [ ] tokenizer（`tiktoken.model` の BPE と特殊トークン、chat template）
3. [ ] CUDAドライバ呼び出し（`nvcuda.dll` 直叩き）と `.cu` のビルド（`build.rs` から nvcc）
4. [ ] MXFP4 dequant＋GEMV カーネル
5. [ ] KDA カーネル（chunked prefill と1トークン decode）
6. [ ] Gated MLA カーネル
7. [ ] LatentMoE（router、shared experts、routed experts）
8. [ ] SSD ストリーミングとエキスパートキャッシュ（VRAM/RAM/SSD の3層）
9. [ ] forward 全体、サンプラ、OpenAI互換API

## ライセンス

MIT。Kimi K3 の重みは Moonshot AI の Kimi K3 ライセンスに従う。
