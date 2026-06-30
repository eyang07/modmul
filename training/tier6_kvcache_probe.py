"""KV-cache gates P2 (parity) + P3 (budget), per the tier-6 plan.

The tier-6 scratchpad (~3000+ tokens) is infeasible under the current no-cache O(L^3)
decode within the 300s total budget. This probe validates the hand-rolled KV-cache
(decode_cached in tier5_modmul.py) on two axes BEFORE any long train:

  * PARITY (P2): decode_cached must produce the IDENTICAL token stream as the proven
    no-cache decode_nocache. Run on a trained ckpt under bf16 (the real risk is a
    bf16 argmax divergence between the full-forward and cached attention reductions);
    a random-init model also exercises the math in fp32. Gate: zero mismatches.
  * BUDGET (P3): time the cached decode to a target length on a random-init model at
    the planned tier-6 config. Gate: projected tiers 0-6 total < 280s. This decides
    base-256 vs base-64 vs skip-tier-6.

Usage:
  # P2 parity on the trained tier-5 ckpt (run on the pod, bf16):
  python training/tier6_kvcache_probe.py parity \
      --ckpt training/checkpoints/modmul_t5_b16_d512_best.pt --amp --n 500

  # P2 parity quick math check, random-init, CPU/fp32 (run anywhere):
  python training/tier6_kvcache_probe.py parity \
      --base 16 --d-model 128 --layers 4 --nhead 4 --dim-ff 256 \
      --max-len 1853 --abacus-max 19 --n 64

  # P3 budget microbench at a tier-6 config + length (run on the pod, bf16):
  python training/tier6_kvcache_probe.py bench \
      --base 256 --d-model 512 --layers 10 --nhead 8 --dim-ff 2048 \
      --max-len 3400 --abacus-max 20 --n 100 --amp
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time

import torch

from tier5_modmul import (
    AbacusDecoder, _is_prime, _sdpa_layer_step, build_prime_pool, decode_cached,
    decode_nocache, digits_base, make_vocab, pick_device,
)


def build_model(args, device):
    """From a ckpt (real weights + cfg) or a random-init config (timing/math check)."""
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ck["config"]
        V = make_vocab(cfg["base"])
        model = AbacusDecoder(V["VOCAB"], cfg["max_len"], cfg["abacus_max"],
                              d_model=cfg["d_model"], nhead=cfg["nhead"],
                              num_layers=cfg["layers"], dim_ff=cfg["dim_ff"]).to(device)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.ckpt} | step {ck.get('step','?')} best {ck.get('best',-1):.3f}")
        return model.eval(), V, cfg["base"], cfg["max_len"], cfg["abacus_max"]
    V = make_vocab(args.base)
    model = AbacusDecoder(V["VOCAB"], args.max_len, args.abacus_max, d_model=args.d_model,
                          nhead=args.nhead, num_layers=args.layers, dim_ff=args.dim_ff).to(device)
    print(f"random-init | base {args.base} d_model {args.d_model} layers {args.layers} "
          f"max_len {args.max_len}")
    return model.eval(), V, args.base, args.max_len, args.abacus_max


def make_prompts(n, base, V, p_min, p_max, rng):
    """n header prompts `BOS x MUL y MOD p EQ` (+ abacus), grouped by length so each
    decode chunk is uniform-width. Real primes when the range is enumerable-ish; else
    Miller-Rabin nextprime of a random draw (enough for parity/timing -- the model
    output only needs to be self-consistent across the two decoders)."""
    pool = build_prime_pool(p_min, p_max, min(2000, n * 4), rng)
    samples = []
    for _ in range(n):
        p = pool[rng.randrange(len(pool))]
        x, y = rng.randrange(p), rng.randrange(p)
        xd, yd, pd = digits_base(x, base), digits_base(y, base), digits_base(p, base)
        toks = [V["BOS"]] + xd + [V["MUL"]] + yd + [V["MOD"]] + pd + [V["EQ"]]
        abac = ([0] + list(range(len(xd))) + [0] + list(range(len(yd)))
                + [0] + list(range(len(pd))) + [0])
        samples.append((toks, abac, (x * y) % p))
    return samples


def chunked_by_len(samples, chunk):
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[len(s[0])].append(i)
    out = []
    for _, idxs in groups.items():
        for cs in range(0, len(idxs), chunk):
            out.append(idxs[cs:cs + chunk])
    return out


def run_parity(args, device, amp_ctx):
    model, V, base, max_len, abacus_max = build_model(args, device)
    specials_t = torch.tensor(sorted(V["SPECIALS"]), device=device)
    rng = random.Random(args.seed)
    samples = make_prompts(args.n, base, V, args.p_min, args.p_max, rng)
    chunks = chunked_by_len(samples, args.chunk)

    mism, ans_nc, ans_c, tot = 0, 0, 0, 0
    with amp_ctx:
        for idxs in chunks:
            toks = torch.tensor([samples[i][0] for i in idxs], dtype=torch.long, device=device)
            abac = torch.tensor([samples[i][1] for i in idxs], dtype=torch.long, device=device)
            g_nc = decode_nocache(model, toks, abac, max_len, abacus_max, specials_t, V)
            g_c = decode_cached(model, toks, abac, max_len, abacus_max, specials_t, V)
            for j, i in enumerate(idxs):
                tot += 1
                if g_nc[j] != g_c[j]:
                    mism += 1
                    if mism <= 3:
                        print(f"  MISMATCH sample {i}: lens {len(g_nc[j])} vs {len(g_c[j])}")
                if args.ckpt:                       # only a trained model has real answers
                    ans_nc += int(_answer(g_nc[j], base, V) == samples[i][2])
                    ans_c += int(_answer(g_c[j], base, V) == samples[i][2])
    print(f"\nPARITY: {tot - mism}/{tot} identical token streams "
          f"({'PASS -- zero mismatches' if mism == 0 else f'FAIL -- {mism} mismatch(es)'})")
    if args.ckpt:
        print(f"answer acc: no-cache {ans_nc/tot:.3f} | cached {ans_c/tot:.3f} "
              f"(should match)")
    return mism == 0


@torch.no_grad()
def run_logitparity(args, device, amp_ctx):
    """Teacher-forced LOGIT parity -- the weight-independent correctness test.

    Argmax token streams (run_parity) are fragile under random init: near-tied logits
    flip on tiny cache-vs-full-forward rounding, so a random model FAILS even when the
    cache math is exact. Here we instead run decode_cached while recording, per step,
    the raw logit row it emits and the (token, abacus) it commits; then re-run the
    full-forward model.forward over that SAME reconstructed (toks, abac) and compare
    logits position-by-position. The cache is correct iff max|Delta logit| is at the
    arithmetic floor (fp32 ~1e-3; bf16 ~O(0.1-1) precision, which a trained model's
    large margins absorb). argmax-agree is reported too but is the tie-sensitive metric.
    """
    model, V, base, max_len, abacus_max = build_model(args, device)
    specials_t = torch.tensor(sorted(V["SPECIALS"]), device=device)
    rng = random.Random(args.seed)
    samples = make_prompts(args.n, base, V, args.p_min, args.p_max, rng)
    chunks = chunked_by_len(samples, args.chunk)
    layers = model.transformer.layers

    max_abs, sum_abs, agree, tot = 0.0, 0.0, 0, 0
    margin_lt_delta = 0
    with amp_ctx:
        for idxs in chunks:
            toks0 = torch.tensor([samples[i][0] for i in idxs], dtype=torch.long, device=device)
            abac0 = torch.tensor([samples[i][1] for i in idxs], dtype=torch.long, device=device)
            B, T0 = toks0.shape
            # --- cached pass: record each emitted logit + the (tok, abac) committed ---
            h = (model.tok_emb(toks0) + model.pos_emb(model.pos_ids[:T0])
                 + model.abacus_emb(abac0))
            pk = [None] * len(layers); pv = [None] * len(layers)
            for li, layer in enumerate(layers):
                h, pk[li], pv[li] = _sdpa_layer_step(layer, h, None, None)
            logit = model.head(model.ln(h[:, -1]))     # predicts token at abs pos T0
            full_toks, full_abac = [toks0], [abac0]
            clog, cpos = [logit], [T0]
            nxt = logit.argmax(-1)
            seg = torch.zeros(B, dtype=torch.long, device=device)
            place, steps = T0, 0
            while place < max_len - 1 and steps < args.steps_cap:
                is_special = (nxt.unsqueeze(1) == specials_t).any(1)
                new_abac = torch.where(is_special, torch.zeros_like(seg),
                                       torch.clamp(seg, max=abacus_max - 1))
                seg = torch.where(is_special, torch.zeros_like(seg), seg + 1)
                full_toks.append(nxt.unsqueeze(1)); full_abac.append(new_abac.unsqueeze(1))
                h = (model.tok_emb(nxt) + model.pos_emb(model.pos_ids[place])
                     + model.abacus_emb(new_abac)).unsqueeze(1)
                for li, layer in enumerate(layers):
                    h, pk[li], pv[li] = _sdpa_layer_step(layer, h, pk[li], pv[li])
                logit = model.head(model.ln(h[:, -1]))  # predicts token at abs pos place+1
                clog.append(logit); cpos.append(place + 1)
                nxt = logit.argmax(-1)
                place += 1; steps += 1
            # --- reference: full-forward over the reconstructed sequence ---
            ft, fa = torch.cat(full_toks, dim=1), torch.cat(full_abac, dim=1)
            ref = model(ft, fa)                          # ref[:, i] predicts token i+1
            for logit, p in zip(clog, cpos):
                rl = ref[:, p - 1]
                d = (logit.float() - rl.float()).abs()
                max_abs = max(max_abs, d.max().item())
                sum_abs += d.mean().item()
                agree += (logit.argmax(-1) == rl.argmax(-1)).sum().item()
                # margin of the reference (gap between top-1 and top-2 logit)
                top2 = rl.float().topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1])
                margin_lt_delta += (margin < d.max(dim=-1).values).sum().item()
                tot += logit.shape[0]
    mean_abs = sum_abs / max(1, len(clog) * len(chunks))
    print(f"\nLOGIT-PARITY ({'bf16' if (args.amp and device.type=='cuda') else 'fp32'}): "
          f"max|Δlogit| = {max_abs:.3e} | mean|Δ| ≈ {mean_abs:.3e} over {tot} positions")
    print(f"  argmax-agree {agree}/{tot} ({agree/max(1,tot):.4f}) "
          f"| positions where ref-margin < Δ (i.e. a tie the cache 'could' flip): "
          f"{margin_lt_delta}/{tot}")
    ok = max_abs < args.tol
    print(f"  gate(max|Δ| < tol={args.tol:g}): "
          f"{'PASS -- cache numerically faithful' if ok else 'FAIL -- real divergence'}")
    return ok


def _answer(gen, base, V):
    if V["COLON"] not in gen:
        return None
    k = len(gen) - 1 - gen[::-1].index(V["COLON"])
    ans = [d for d in gen[k + 1:] if d < base]
    if not ans:
        return None
    v = 0
    for d in ans:
        v = v * base + d
    return v


@torch.no_grad()
def run_bench(args, device, amp_ctx):
    """Time a full-length cached decode (EOS ignored -> worst case) for n problems at
    the given max_len. Reports s/100 and the implied tier-6 budget headroom."""
    model, V, base, max_len, abacus_max = build_model(args, device)
    rng = random.Random(args.seed)
    samples = make_prompts(args.n, base, V, args.p_min, args.p_max, rng)
    # one uniform-width chunk for clean timing (pad-group the most common length)
    chunks = chunked_by_len(samples, args.chunk)

    def time_decode(use_cache):
        layers = model.transformer.layers
        t0 = time.monotonic()
        for idxs in chunks:
            toks = torch.tensor([samples[i][0] for i in idxs], dtype=torch.long, device=device)
            abac = torch.tensor([samples[i][1] for i in idxs], dtype=torch.long, device=device)
            B, T0 = toks.shape
            if use_cache:
                h = (model.tok_emb(toks) + model.pos_emb(model.pos_ids[:T0])
                     + model.abacus_emb(abac))
                pk = [None] * len(layers); pv = [None] * len(layers)
                for li, layer in enumerate(layers):
                    h, pk[li], pv[li] = _sdpa_layer_step(layer, h, None, None)
                nxt = model.head(model.ln(h[:, -1])).argmax(-1)
                seg = torch.zeros(B, dtype=torch.long, device=device)
                for place in range(T0, max_len - 1):       # full length, EOS ignored
                    new_abac = torch.clamp(seg, max=abacus_max - 1)
                    seg = seg + 1
                    h = (model.tok_emb(nxt) + model.pos_emb(model.pos_ids[place])
                         + model.abacus_emb(new_abac)).unsqueeze(1)
                    for li, layer in enumerate(layers):
                        h, pk[li], pv[li] = _sdpa_layer_step(layer, h, pk[li], pv[li])
                    nxt = model.head(model.ln(h[:, -1])).argmax(-1)
            else:
                while toks.shape[1] < max_len:
                    nxt = model(toks, abac)[:, -1].argmax(-1)
                    ab = torch.clamp(
                        torch.full((B,), toks.shape[1], device=device), max=abacus_max - 1)
                    toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
                    abac = torch.cat([abac, ab.unsqueeze(1)], dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.monotonic() - t0

    with amp_ctx:
        _ = time_decode(True)                              # warmup (CUDA graphs/alloc)
        t_cached = time_decode(True)
    per100 = t_cached / args.n * 100
    print(f"\nBUDGET: cached decode to L={max_len}, {args.n} problems "
          f"({len(chunks)} chunk(s)) = {t_cached:.1f}s -> {per100:.1f}s / 100")
    # tiers 0-4 ~50s + tier5 (cached, projected) + this tier6; gate is total < 280s.
    print(f"gate: with tiers 0-5 (~60-90s cached) leave ~210-240s for tier 6; "
          f"this tier-6 estimate {per100:.0f}s -> "
          f"{'FITS' if per100 < 200 else 'TIGHT/OVER -- shorten chain (lower base) or skip'}")
    if args.also_nocache:
        with amp_ctx:
            t_nc = time_decode(False)
        print(f"  (no-cache same config = {t_nc/args.n*100:.1f}s/100 -> "
              f"speedup {t_nc/max(t_cached,1e-9):.1f}x)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parity", "logitparity", "bench"])
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=1853)
    ap.add_argument("--abacus-max", type=int, default=19)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--p-min", type=int, default=2 ** 33)
    ap.add_argument("--p-max", type=int, default=2 ** 64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--also-nocache", action="store_true",
                    help="bench: also time the no-cache path to report the speedup")
    ap.add_argument("--steps-cap", type=int, default=64,
                    help="logitparity: max generated steps per chunk (keeps it fast)")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="logitparity: max|Δlogit| gate (fp32 floor ~1e-3; use larger for bf16)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if args.amp and device.type == "cuda" else contextlib.nullcontext())
    print(f"device {device} | amp {args.amp and device.type == 'cuda'}")

    if args.mode == "parity":
        ok = run_parity(args, device, amp_ctx)
    elif args.mode == "logitparity":
        ok = run_logitparity(args, device, amp_ctx)
    else:
        ok = run_bench(args, device, amp_ctx)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
