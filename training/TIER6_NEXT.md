# TIER 6 — RESUME HERE (command-first)

> Picking the tier-6 push back up. Read this top-to-bottom; run the commands in order on
> the CUDA pod. Full rationale: `~/.claude/plans/elegant-tinkering-toast.md` and
> `memory/tier6-progress.md`. Everything below is already built + pushed to branch `ebm-dev`.

## Where we are
- **Shipped:** htop90=5, `cire77/modmul @9e196608` (tier 5 is a ~0.90 coin flip; floor htop90=4).
- **Goal now:** reach **tier 6** (`p ∈ [2^64, 2^128)`); solidify tier 5 along the way.
- **Key fact:** tier 6 at **base-256** has the SAME chain length as tier-5 base-16 (L≈1853;
  2397 with `--self-check`). So no sub-scratchpad is needed for length — IF the base-256
  one-shot multiply `pp = x·d` groks (that is gate P1, the one real unknown).
- **Already built (this branch):** KV-cache decode (`decode_cached` in `training/tier5_modmul.py`,
  O(L^3)→O(L^2)); `--self-check` borrow field `bk=(p-1)-r`; gate probe
  `training/tier6_kvcache_probe.py`. Local parity validated (48/48).

---

## STEP 0 — sync
```bash
cd /workspace/modular-arithmetic-challenge && git checkout ebm-dev && git pull
```

## STEP 1 — run the 3 gates (do P2 + P3 first; they're minutes)

**P2 — KV-cache parity (HARD gate, must pass).**
```bash
python training/tier6_kvcache_probe.py parity \
    --ckpt training/checkpoints/modmul_t5_b16_d512_best.pt --amp --n 500
```
PASS = `500/500 identical` + the two answer-acc lines equal. If it FAILS → ping Claude
(needs an fp32-softmax fallback in `_sdpa_layer_step`); do not proceed past this.

**P3 — budget microbench.**
```bash
# base-256 ship candidate (L=2397, ~tier-5 length):
python training/tier6_kvcache_probe.py bench --base 256 --d-model 512 --layers 10 \
    --nhead 8 --dim-ff 2048 --max-len 2397 --abacus-max 19 --n 100 --amp --also-nocache
# base-16 fallback (L=6765, the slow case):
python training/tier6_kvcache_probe.py bench --base 16 --d-model 512 --layers 10 \
    --nhead 8 --dim-ff 2048 --max-len 6765 --abacus-max 35 --n 100 --amp --also-nocache
```
PASS = base-256 `s/100` small enough that tiers 0–6 total < 280s (it should be ~tier-5 cost).

**P1 — base-256 multiply learnability (~1–2h; background).**
```bash
python training/tier5_pp_probe.py --base 256 --bits-max 128 --amp \
    --ckpt training/checkpoints/pp_b256_t6.pt > pp_b256_t6.log 2>&1 &
tail -f pp_b256_t6.log
```
PASS = widest-x bucket → ~1.0 AND `tf_tok > 0.999` within a few thousand steps.
- If PASS → train tier 6 at **base 256** (STEP 2).
- If STALLS → re-run with `--base 128` then `--base 64` (shorter atom, longer chain); use the
  largest base that groks. If none grok → the explicit-multiply sub-scratchpad is needed
  (not yet built — ping Claude).

---

## STEP 2 — train (after gates green). 5090, ~12h each; recipe = proven tier-5 recipe.

**Tier 6** (base from P1; example shows base-256). p-min=2^64, p-max=2^128:
```bash
python training/tier5_modmul.py --base 256 --amp --steps 80000 --batch 64 --curriculum \
    --curr-frac 0.85 --prime-pow 1.0 --self-check \
    --p-min 18446744073709551616 --p-max 340282366920938463463374607431768211456 \
    --d-model 512 --layers 10 --dim-ff 2048 \
    --ckpt training/checkpoints/modmul_t6_b256_d512.pt > t6_b256.log 2>&1 &
tail -f t6_b256.log
```
Watch: `tf_tok` climbing past ~0.94; the hardest log-bucket (top, ≈99.9% of the scorer's
value-uniform draw) climbing; `skip` stays 0; no collapse when `cur_pmax` opens to full range
(~step 60–70k). Best ckpt auto-saved as `..._best.pt` (gated on hardest bucket).

**Tier 5 solidify** (base-16 + self-check, lifts the coin flip):
```bash
python training/tier5_modmul.py --base 16 --amp --steps 80000 --batch 64 --curriculum \
    --curr-frac 0.85 --prime-pow 1.0 --self-check \
    --d-model 512 --layers 10 --dim-ff 2048 \
    --ckpt training/checkpoints/modmul_t5_b16_d512_sc.pt > t5_sc.log 2>&1 &
```

**Verdict probe** (scorer-faithful; reads the prime range from the ckpt config, no extra args):
```bash
python training/tier5_accuracy_probe.py --ckpt training/checkpoints/modmul_t6_b256_d512_best.pt \
    --amp --primes 25 --per-prime 40
```
Read `overall mean` (expected tier score) and `P(tier >= 0.90)`. Even <0.90 on tier 6 still
raises the `overall_accuracy` tiebreaker if it COMPLETES in budget — so it's worth shipping.

---

## STEP 3 — integrate + submit (Claude writes the code after P2 passes; not yet built)
When you're back with gates green and ckpts trained, ping Claude to:
1. Port `decode_cached` into `submission/ebm_modmul/model.py` (cached `_modmul_decode_base`)
   and add the **tier6 route**: `p ≥ 2^64 & < 2^128` → tier6 decoder; keep `≥ 2^128 → [0]`.
2. Generalize `training/bundle_tier5.py` to write a `tier6` weights key (with parity check).
3. Bundle, then verify locally — **this is the real gate**:
   ```bash
   modchallenge evaluate submission/ebm_modmul --seed abcd1234 --timeout 300 -v
   ```
   Need: tiers 0–6 all `complete` within 300s, tier 5 ≥0.90, `deterministic: true`, static pass.
4. Upload to `cire77/modmul` (new commit), `modchallenge evaluate-hf cire77/modmul <sha>` to
   reproduce through the official path, then submit repo_id + SHA. Leaderboard keeps best →
   pure upside (can only raise htop90 from the banked 5).

## Constraints (unchanged — static AST scanner checks every submission .py)
No `%`//`//`/division/Barrett/Montgomery/CRT on the product; reduction must be learned; only
`int(a)%p`/`int(b)%p` allowed; deterministic; no RL. KV-cache + `bk` are compliant.
