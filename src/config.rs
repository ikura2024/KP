//! Kimi K3 の config.json を型付きで読む。
//!
//! 注意: config 内のレイヤー番号 (`kda_layers` / `full_attn_layers`) は **1始まり**。
//! KP 内部では 0 始まりの `layers[i]` に直して持つ。

use crate::json::{ParseError, Value};
use std::fmt;

/// アテンションの種類。K3 は KDA(線形) 69層 + Gated MLA(全注意) 24層 の混成。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    Kda,
    Mla,
}

/// FFN の種類。先頭 `first_k_dense_replace` 層だけ dense、残りは MoE。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind {
    Dense,
    Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpec {
    pub attn: AttnKind,
    pub ffn: FfnKind,
}

#[derive(Debug)]
pub enum ConfigError {
    Json(ParseError),
    Missing(String),
    WrongType { path: String, expected: &'static str },
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Json(e) => write!(f, "{e}"),
            ConfigError::Missing(p) => write!(f, "missing field: {p}"),
            ConfigError::WrongType { path, expected } => write!(f, "field {path}: expected {expected}"),
            ConfigError::Invalid(m) => write!(f, "invalid config: {m}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<ParseError> for ConfigError {
    fn from(e: ParseError) -> Self {
        ConfigError::Json(e)
    }
}

#[derive(Debug, Clone)]
pub struct K3Config {
    // 全体
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub attn_res_block_size: usize,
    pub num_nextn_predict_layers: usize,
    // dense FFN
    pub dense_intermediate_size: usize,
    // MLA (全注意層)
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub mla_use_nope: bool,
    pub mla_use_output_gate: bool,
    // KDA (線形注意層)
    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
    pub kda_short_conv_kernel: usize,
    pub kda_gate_lower_bound: f64,
    pub kda_use_full_rank_gate: bool,
    // MoE (LatentMoE)
    pub moe_intermediate_size: usize,
    pub routed_expert_hidden_size: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub num_shared_experts: usize,
    pub first_k_dense_replace: usize,
    pub moe_renormalize: bool,
    pub routed_scaling_factor: f64,
    pub router_activation: String,
    pub topk_method: String,
    pub latent_moe_use_norm: bool,
    // routed experts の量子化 (MXFP4)
    pub expert_quant_format: String,
    pub expert_quant_bits: usize,
    pub expert_quant_group_size: usize,
    // レイヤー構成 (0始まり)
    pub layers: Vec<LayerSpec>,
}

fn field<'a>(obj: &'a Value, parent: &str, key: &str) -> Result<&'a Value, ConfigError> {
    obj.get(key).ok_or_else(|| ConfigError::Missing(format!("{parent}.{key}")))
}

fn wrong(parent: &str, key: &str, expected: &'static str) -> ConfigError {
    ConfigError::WrongType { path: format!("{parent}.{key}"), expected }
}

fn usize_of(obj: &Value, parent: &str, key: &str) -> Result<usize, ConfigError> {
    field(obj, parent, key)?.as_usize().ok_or_else(|| wrong(parent, key, "non-negative integer"))
}

fn f64_of(obj: &Value, parent: &str, key: &str) -> Result<f64, ConfigError> {
    field(obj, parent, key)?.as_f64().ok_or_else(|| wrong(parent, key, "number"))
}

fn bool_of(obj: &Value, parent: &str, key: &str) -> Result<bool, ConfigError> {
    field(obj, parent, key)?.as_bool().ok_or_else(|| wrong(parent, key, "bool"))
}

fn str_of(obj: &Value, parent: &str, key: &str) -> Result<String, ConfigError> {
    field(obj, parent, key)?.as_str().map(str::to_string).ok_or_else(|| wrong(parent, key, "string"))
}

fn usize_list(obj: &Value, parent: &str, key: &str) -> Result<Vec<usize>, ConfigError> {
    let arr = field(obj, parent, key)?.as_array().ok_or_else(|| wrong(parent, key, "array"))?;
    arr.iter().map(|v| v.as_usize().ok_or_else(|| wrong(parent, key, "array of non-negative integers"))).collect()
}

impl K3Config {
    pub fn from_json_str(src: &str) -> Result<K3Config, ConfigError> {
        let root = Value::parse(src)?;
        let text = root.get("text_config").unwrap_or(&root);
        let t = "text_config";

        let num_layers = usize_of(text, t, "num_hidden_layers")?;
        let first_k_dense_replace = usize_of(text, t, "first_k_dense_replace")?;
        let moe_layer_freq = usize_of(text, t, "moe_layer_freq")?;
        if moe_layer_freq != 1 {
            return Err(ConfigError::Invalid(format!("moe_layer_freq={moe_layer_freq} is not supported (only 1)")));
        }

        // KDA / MLA の割り当て（config は1始まり）
        let lin = field(text, t, "linear_attn_config")?;
        let l = "text_config.linear_attn_config";
        let kda = usize_list(lin, l, "kda_layers")?;
        let full = usize_list(lin, l, "full_attn_layers")?;
        let mut attn: Vec<Option<AttnKind>> = vec![None; num_layers];
        for (list, kind) in [(&kda, AttnKind::Kda), (&full, AttnKind::Mla)] {
            for &n in list.iter() {
                if n == 0 || n > num_layers {
                    return Err(ConfigError::Invalid(format!("layer number {n} out of range 1..={num_layers}")));
                }
                if attn[n - 1].is_some() {
                    return Err(ConfigError::Invalid(format!("layer {n} listed more than once")));
                }
                attn[n - 1] = Some(kind);
            }
        }
        let mut layers = Vec::with_capacity(num_layers);
        for (i, a) in attn.into_iter().enumerate() {
            let attn = a.ok_or_else(|| ConfigError::Invalid(format!("layer {} has no attention kind", i + 1)))?;
            let ffn = if i < first_k_dense_replace { FfnKind::Dense } else { FfnKind::Moe };
            layers.push(LayerSpec { attn, ffn });
        }

        // routed experts の量子化
        let q = field(text, t, "quantization_config")?;
        let group0 = q
            .get("config_groups")
            .and_then(|g| g.get("group_0"))
            .ok_or_else(|| ConfigError::Missing(format!("{t}.quantization_config.config_groups.group_0")))?;
        let qpath = "text_config.quantization_config.config_groups.group_0";
        let weights = field(group0, qpath, "weights")?;
        let wpath = "text_config.quantization_config.config_groups.group_0.weights";

        Ok(K3Config {
            hidden_size: usize_of(text, t, "hidden_size")?,
            num_layers,
            vocab_size: usize_of(text, t, "vocab_size")?,
            max_position_embeddings: usize_of(text, t, "max_position_embeddings")?,
            rms_norm_eps: f64_of(text, t, "rms_norm_eps")?,
            hidden_act: str_of(text, t, "hidden_act")?,
            attn_res_block_size: usize_of(text, t, "attn_res_block_size")?,
            num_nextn_predict_layers: usize_of(text, t, "num_nextn_predict_layers")?,
            dense_intermediate_size: usize_of(text, t, "intermediate_size")?,
            num_heads: usize_of(text, t, "num_attention_heads")?,
            num_kv_heads: usize_of(text, t, "num_key_value_heads")?,
            q_lora_rank: usize_of(text, t, "q_lora_rank")?,
            kv_lora_rank: usize_of(text, t, "kv_lora_rank")?,
            qk_nope_head_dim: usize_of(text, t, "qk_nope_head_dim")?,
            qk_rope_head_dim: usize_of(text, t, "qk_rope_head_dim")?,
            v_head_dim: usize_of(text, t, "v_head_dim")?,
            mla_use_nope: bool_of(text, t, "mla_use_nope")?,
            mla_use_output_gate: bool_of(text, t, "mla_use_output_gate")?,
            kda_num_heads: usize_of(lin, l, "num_heads")?,
            kda_head_dim: usize_of(lin, l, "head_dim")?,
            kda_short_conv_kernel: usize_of(lin, l, "short_conv_kernel_size")?,
            kda_gate_lower_bound: f64_of(lin, l, "gate_lower_bound")?,
            kda_use_full_rank_gate: bool_of(lin, l, "use_full_rank_gate")?,
            moe_intermediate_size: usize_of(text, t, "moe_intermediate_size")?,
            routed_expert_hidden_size: usize_of(text, t, "routed_expert_hidden_size")?,
            num_experts: usize_of(text, t, "num_experts")?,
            experts_per_token: usize_of(text, t, "num_experts_per_token")?,
            num_shared_experts: usize_of(text, t, "num_shared_experts")?,
            first_k_dense_replace,
            moe_renormalize: bool_of(text, t, "moe_renormalize")?,
            routed_scaling_factor: f64_of(text, t, "routed_scaling_factor")?,
            router_activation: str_of(text, t, "moe_router_activation_func")?,
            topk_method: str_of(text, t, "topk_method")?,
            latent_moe_use_norm: bool_of(text, t, "latent_moe_use_norm")?,
            expert_quant_format: str_of(group0, qpath, "format")?,
            expert_quant_bits: usize_of(weights, wpath, "num_bits")?,
            expert_quant_group_size: usize_of(weights, wpath, "group_size")?,
            layers,
        })
    }

    pub fn from_file(path: &str) -> Result<K3Config, ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::Invalid(format!("cannot read {path}: {e}")))?;
        K3Config::from_json_str(&text)
    }

    pub fn attn_layers(&self, kind: AttnKind) -> usize {
        self.layers.iter().filter(|l| l.attn == kind).count()
    }

    pub fn moe_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.ffn == FfnKind::Moe).count()
    }

    /// routed expert 1個のパラメータ数。gate/up/down の3行列（latent次元 x moe中間次元）。
    pub fn params_per_routed_expert(&self) -> u64 {
        3 * self.routed_expert_hidden_size as u64 * self.moe_intermediate_size as u64
    }

    /// 全 routed expert のパラメータ総数（量子化前の論理的な個数）。
    pub fn routed_expert_params(&self) -> u64 {
        self.params_per_routed_expert() * self.num_experts as u64 * self.moe_layers() as u64
    }

    /// MXFP4 の本体(4bit詰め)バイト数 = params / 2。
    pub fn packed_bytes(params: u64) -> u64 {
        params / 2
    }

    /// MXFP4 のスケール(グループごとに1バイト)バイト数。
    pub fn scale_bytes(&self, params: u64) -> u64 {
        params / self.expert_quant_group_size as u64
    }

    /// 1トークンで実際に読まれる routed expert のパラメータ数。
    pub fn routed_active_params_per_token(&self) -> u64 {
        self.params_per_routed_expert() * self.experts_per_token as u64 * self.moe_layers() as u64
    }

    /// 1トークンで読む routed expert の実バイト数（本体+スケール）。
    pub fn routed_active_bytes_per_token(&self) -> u64 {
        let p = self.routed_active_params_per_token();
        Self::packed_bytes(p) + self.scale_bytes(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 実物の K3 config から必要な項目だけ抜いた JSON を組み立てる。
    fn sample_json() -> String {
        let full: Vec<usize> = (4..=92).step_by(4).chain([93]).collect();
        let kda: Vec<usize> = (1..=93).filter(|n| !full.contains(n)).collect();
        let list = |v: &[usize]| v.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        format!(
            r#"{{
  "model_type": "kimi_k3",
  "text_config": {{
    "hidden_size": 7168, "num_hidden_layers": 93, "vocab_size": 163840,
    "max_position_embeddings": 1048576, "rms_norm_eps": 1E-05, "hidden_act": "situ",
    "attn_res_block_size": 12, "num_nextn_predict_layers": 0,
    "intermediate_size": 33792,
    "num_attention_heads": 96, "num_key_value_heads": 96,
    "q_lora_rank": 1536, "kv_lora_rank": 512,
    "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
    "mla_use_nope": true, "mla_use_output_gate": true,
    "moe_intermediate_size": 3072, "routed_expert_hidden_size": 3584,
    "num_experts": 896, "num_experts_per_token": 16, "num_shared_experts": 2,
    "first_k_dense_replace": 1, "moe_layer_freq": 1, "moe_renormalize": true,
    "routed_scaling_factor": 1.0, "moe_router_activation_func": "sigmoid",
    "topk_method": "noaux_tc", "latent_moe_use_norm": true,
    "linear_attn_config": {{
      "full_attn_layers": [{}], "kda_layers": [{}],
      "gate_lower_bound": -5.0, "head_dim": 128, "num_heads": 96,
      "short_conv_kernel_size": 4, "use_full_rank_gate": true
    }},
    "quantization_config": {{
      "format": "mxfp4-pack-quantized",
      "config_groups": {{ "group_0": {{
        "format": "mxfp4-pack-quantized",
        "weights": {{ "num_bits": 4, "group_size": 32, "strategy": "group" }}
      }} }}
    }}
  }}
}}"#,
            list(&full),
            list(&kda)
        )
    }

    fn check_k3(c: &K3Config) {
        assert_eq!(c.num_layers, 93);
        assert_eq!(c.layers.len(), 93);
        assert_eq!(c.attn_layers(AttnKind::Kda), 69);
        assert_eq!(c.attn_layers(AttnKind::Mla), 24);
        assert_eq!(c.moe_layers(), 92);
        // config は1始まり: 1層目は KDA+dense、4層目は MLA+MoE、93層目は MLA+MoE
        assert_eq!(c.layers[0], LayerSpec { attn: AttnKind::Kda, ffn: FfnKind::Dense });
        assert_eq!(c.layers[3], LayerSpec { attn: AttnKind::Mla, ffn: FfnKind::Moe });
        assert_eq!(c.layers[91], LayerSpec { attn: AttnKind::Mla, ffn: FfnKind::Moe });
        assert_eq!(c.layers[92], LayerSpec { attn: AttnKind::Mla, ffn: FfnKind::Moe });
        assert_eq!(c.layers[91 - 1].attn, AttnKind::Kda); // 91層目

        assert_eq!(c.params_per_routed_expert(), 33_030_144);
        // Hugging Face が公開している moonshotai/Kimi-K3 の safetensors メタデータの
        // U8 要素数 (4bit詰めを2個と数えた値) と完全に一致する。
        assert_eq!(c.routed_expert_params(), 2_722_740_830_208);
        assert_eq!(K3Config::packed_bytes(c.routed_expert_params()), 1_361_370_415_104);
        assert_eq!(c.scale_bytes(c.routed_expert_params()), 85_085_650_944);

        assert_eq!(c.routed_active_params_per_token(), 48_620_371_968);
        assert_eq!(c.routed_active_bytes_per_token(), 25_829_572_608);
        assert_eq!(c.expert_quant_bits, 4);
        assert_eq!(c.expert_quant_group_size, 32);
        assert_eq!(c.expert_quant_format, "mxfp4-pack-quantized");
        assert!((c.rms_norm_eps - 1e-5).abs() < 1e-12);
        assert!((c.kda_gate_lower_bound + 5.0).abs() < 1e-12);
    }

    #[test]
    fn parses_k3_shape() {
        let c = K3Config::from_json_str(&sample_json()).unwrap();
        check_k3(&c);
    }

    #[test]
    fn parses_real_config_if_present() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/kimi-k3/config.json");
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: {path} not found");
            return;
        }
        let c = K3Config::from_file(path).unwrap();
        check_k3(&c);
    }

    #[test]
    fn rejects_bad_layer_tables() {
        let ok = sample_json();
        // 93層目を KDA 側にも書いて重複させる
        let dup = ok.replacen("\"kda_layers\": [1,", "\"kda_layers\": [93,1,", 1);
        assert!(matches!(K3Config::from_json_str(&dup), Err(ConfigError::Invalid(_))));
        // 範囲外
        let oob = ok.replacen("\"kda_layers\": [1,", "\"kda_layers\": [94,1,", 1);
        assert!(matches!(K3Config::from_json_str(&oob), Err(ConfigError::Invalid(_))));
        // 1層目が抜ける
        let hole = ok.replacen("\"kda_layers\": [1,", "\"kda_layers\": [", 1);
        assert!(matches!(K3Config::from_json_str(&hole), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn reports_missing_and_wrong_type() {
        let ok = sample_json();
        let missing = ok.replacen("\"hidden_size\": 7168,", "", 1);
        match K3Config::from_json_str(&missing) {
            Err(ConfigError::Missing(p)) => assert_eq!(p, "text_config.hidden_size"),
            other => panic!("{other:?}"),
        }
        let wrong = ok.replacen("\"hidden_size\": 7168", "\"hidden_size\": \"big\"", 1);
        assert!(matches!(K3Config::from_json_str(&wrong), Err(ConfigError::WrongType { .. })));
        assert!(matches!(K3Config::from_json_str("{"), Err(ConfigError::Json(_))));
    }
}
