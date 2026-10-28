use kp::config::{AttnKind, K3Config};
use std::{env, process};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("inspect") => match args.get(1) {
            Some(path) => inspect(path),
            None => {
                eprintln!("usage: kp inspect <config.json>");
                2
            }
        },
        _ => {
            eprintln!("KP - from-scratch Kimi K3 inference engine\n\nusage:\n  kp inspect <config.json>   print model structure and routed-expert sizes");
            2
        }
    };
    process::exit(code);
}

fn inspect(path: &str) -> i32 {
    let c = match K3Config::from_file(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let routed = c.routed_expert_params();
    let packed = K3Config::packed_bytes(routed);
    let scales = c.scale_bytes(routed);
    let per_tok = c.routed_active_bytes_per_token();

    println!("== Kimi K3 structure ==");
    println!("layers            : {} (KDA {} / MLA {})", c.num_layers, c.attn_layers(AttnKind::Kda), c.attn_layers(AttnKind::Mla));
    println!("FFN               : dense {} / MoE {}", c.num_layers - c.moe_layers(), c.moe_layers());
    println!("hidden / vocab    : {} / {}", c.hidden_size, c.vocab_size);
    println!("context           : {}", c.max_position_embeddings);
    println!("experts           : {} routed (top-{}) + {} shared, latent {} -> inter {}", c.num_experts, c.experts_per_token, c.num_shared_experts, c.routed_expert_hidden_size, c.moe_intermediate_size);
    println!("router            : {} / {} / renormalize={}", c.router_activation, c.topk_method, c.moe_renormalize);
    println!("activation        : {}", c.hidden_act);
    println!();
    println!("== routed experts ({} {}bit, group {}) ==", c.expert_quant_format, c.expert_quant_bits, c.expert_quant_group_size);
    println!("params / expert   : {}", c.params_per_routed_expert());
    println!("params (all)      : {}", routed);
    println!("packed weights    : {} bytes ({:.1} GiB)", packed, packed as f64 / GIB);
    println!("scales            : {} bytes ({:.1} GiB)", scales, scales as f64 / GIB);
    println!("read per token    : {} bytes ({:.2} GiB) for {} active routed params", per_tok, per_tok as f64 / GIB, c.routed_active_params_per_token());
    0
}
