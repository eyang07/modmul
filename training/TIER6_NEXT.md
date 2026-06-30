# TIER 6 — RESUME HERE (command-first)

> **VERDICT 2026-06-30: tier 6 NOT reached; banked htop90=5.** Gates P2/P3 passed. P1: base-256
> multiply won't grok, base-64 does. But the full base-64 modmul train (d512/L10, self-check, 80k
> steps, 9.8h on a 4090) PLATEAUED — tf_tok climbed to ~0.74 then fell to **0.639** when the
> curriculum opened to the full 2^128 range; all buckets ~0. The multiply ATOM groks; the L=4347
> end-to-end chain does not. To retry: the bottleneck is chain length, not the atom (try base-128 /
> bigger model / explicit-multiply sub-scratchpad). The 24GB-card fixes below are real and validated.



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

## DRIVING THE POD OVER SSH (agent ops — read first)

Claude drives the RunPod 4090 **directly via its Bash tool over SSH** (no copy-paste through
the Jupyter terminal). The Mac side is already set up:

- **Connect:** `ssh runpod-t6` — alias in `~/.ssh/config` on the Mac, using a dedicated key
  `~/.ssh/id_runpod` (its public key is on the RunPod account). Connection multiplexing
  (`ControlMaster`/`ControlPersist 10m`) is configured, so back-to-back commands reuse one tunnel.
- **Cold-start drops are normal.** The FIRST connection after the pod boots, or after the
  master idle-expires, often fails with `Connection closed by remote host` (exit 255). Just retry:
  ```bash
  for i in 1 2 3; do ssh -o ConnectTimeout=20 runpod-t6 'echo up' && break || sleep 3; done
  ```
- **New pod = new IP/port.** From RunPod console → Connect → **"SSH over exposed TCP"** (the
  `IP -p PORT` form, NOT the `ssh.runpod.io` proxy — the proxy can't do scp/rsync), update
  `HostName` and `Port` in the `runpod-t6` block of `~/.ssh/config`. The pubkey is already on
  the account, so there is no per-pod key step.

### Bring up a fresh pod (it starts EMPTY — repo not cloned, deps not installed)
```bash
ssh runpod-t6 'cd /workspace && git clone --branch ebm-dev \
  https://github.com/eyang07/modular-arithmetic-challenge.git && \
  cd modular-arithmetic-challenge && pip install -e . --break-system-packages && \
  mkdir -p training/checkpoints'
```
`--break-system-packages` is required (PEP 668 on the system Python; torch is preinstalled).
Verify: `ssh runpod-t6 'cd /workspace/modular-arithmetic-challenge && python -c "import torch;print(torch.cuda.is_available())"'`

### Run long experiments DETACHED (so they survive SSH drops)
Never run a multi-hour train inside a foreground SSH. Launch with `nohup ... > LOG 2>&1 &`.
**Always launch UNBUFFERED (`python -u` / `PYTHONUNBUFFERED=1`)** — otherwise stdout is
block-buffered to the file and the log only flushes at process EXIT, so a live monitor
sees nothing until the run ends (learned the hard way 2026-06-29).
```bash
ssh runpod-t6 'cd /workspace/modular-arithmetic-challenge && \
  PYTHONUNBUFFERED=1 nohup python -u training/<script>.py <args> > run.log 2>&1 & echo PID $!'
ssh runpod-t6 'cd /workspace/modular-arithmetic-challenge && tail -8 run.log; pgrep -af "[<]script>"'
```
**`pkill`/`pgrep` self-match trap:** `pkill -f "tier5_pp_probe.py --base 64"` ALSO matches
the shell running that very command (the pattern is in its argv) → it kills its own SSH
session (exit 255) and the target. Defeat it with a regex bracket so the literal pattern
can't match itself: `pkill -f 'tier5_pp_probe.py.*--base 6[4]'`, `pgrep -fc "[t]ier5_pp_probe.py"`.

### MONITOR ON A LOOP — every 30 minutes
While any train/probe is running, **monitor on a loop and check the log every 30 min**
(`/loop 30m`, or `ScheduleWakeup` with `delaySeconds≈1800`). Each tick: tail the log, report
the leading metric (`tf_tok`, hardest-bucket acc), and **stop the loop + alert the user** when
(a) the gate/grok signal is hit, (b) the run stalls or collapses (tf_tok flat, NaN, `skip` rising),
or (c) the process has died (`pgrep` empty). Don't poll faster than ~30 min — these runs grok slowly.

### Pull artifacts off the pod BEFORE stopping it
Network-volume data can be wiped, and a *stopped* pod hides `/workspace`. As soon as a best
ckpt exists, pull it to the Mac:
```bash
rsync -e ssh -az runpod-t6:/workspace/modular-arithmetic-challenge/training/checkpoints/<ckpt>.pt training/checkpoints/
```
(The tier-5 ship ckpt `modmul_t5_b16_d512_best.pt` was already lost this way once — it's now
only in HF `cire77/modmul @9e196608`.)

### Mac-side gotcha
`find`/`grep` in the agent's Bash sometimes hit a broken shell wrapper (error: "claude native
binary not installed"). Use absolute paths instead: `/usr/bin/find`, `/usr/bin/grep`.

---

## STEP 0 — sync (fresh pod: use the clone+install block above first)
```bash
cd /workspace/modular-arithmetic-challenge && git checkout ebm-dev && git pull
```

## STEP 1 — run the 3 gates (do P2 + P3 first; they're minutes)

**P2 — KV-cache parity (HARD gate). ✅ ALREADY PASSED (2026-06-29).**
The trained tier-5 ckpt is gone, and the argmax `parity` mode FALSELY fails on a random-init
model (near-tied random logits flip under bf16 rounding — not a cache bug). Use the
weight-independent **`logitparity`** mode instead (added to the probe): it teacher-forces the
cached vs full-forward paths over the same tokens and compares raw logits.
```bash
# fp32 — proves the cache algorithm is exact (expect max|Δlogit| ~1e-3):
python training/tier6_kvcache_probe.py logitparity --n 50 --max-len 384 \
    --d-model 512 --layers 10 --steps-cap 64
# bf16 — quantifies the precision gap (expect ~1e-2, far below any trained margin):
python training/tier6_kvcache_probe.py logitparity --amp --n 50 --max-len 384 \
    --d-model 512 --layers 10 --steps-cap 64 --tol 5.0
```
Got fp32 `max|Δ|=4.7e-4`, bf16 `1.2e-2` → cache is numerically faithful. **Deferred final check:**
once a trained tier-6 `_best.pt` exists, run the argmax `parity --ckpt <best> --amp` on it — with
real (large-margin) logits it should be `identical` token streams.

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
**RESULT (2026-06-29): ✅ base-256 = 47s/100 (8.3× speedup), fits easily. ❌ base-16 = 540s/100,
blows the whole 300s budget — so base-256 is the ONLY viable tier-6 path.** This makes P1
(below) make-or-break: if base-256 multiply does not grok, tier 6 needs the explicit-multiply
sub-scratchpad (not yet built), there is no base-16 fallback that fits.

**P1 — base-256 multiply learnability (~1–2h; background).**
```bash
python training/tier5_pp_probe.py --base 256 --bits-max 128 --amp \
    --ckpt training/checkpoints/pp_b256_t6.pt > pp_b256_t6.log 2>&1 &
tail -f pp_b256_t6.log
```
PASS = widest-x bucket → ~1.0 AND `tf_tok > 0.999` within a few thousand steps.
**RESULT (2026-06-29): ❌ base-256 FAILED to grok** (tf_tok dead-flat ~0.21 over a fully-annealed
40k steps, buckets ~0 — the base-256 multiply atom, carries up to 255×255, is too hard).
**✅ base-64 GROKKED** (d384/L8, 40k steps: hardest [88-129b) bucket 0.986, tf_tok 0.9992) **AND
fits budget** (base-64 tier-6 `max_len=4347 abacus_max=25`; bench @ d512/L10 = **132s/100**, fits).
**→ tier 6 ships at base-64** (STEP 2). Did not need base-128. base-64 best probe ckpt:
`training/checkpoints/pp_b64_t6_best.pt`.
- **BUDGET NOTE for integration (STEP 3):** tier-6 decode is 132s, so tier-5 MUST also use the
  KV-cache in the submission (the shipped tier-5 was 174.6s UNcached — 174.6+132 > 300s). Cached
  tier-5 is ~25s, so cached t5 + cached t6 + tiers0-4 ≈ 170-200s, fits.

---

## STEP 2 — train (after gates green). 5090, ~12h each; recipe = proven tier-5 recipe.

**Tier 6 — base-64** (chosen by P1). p-min=2^64, p-max=2^128. Launch UNBUFFERED + detached:
```bash
PYTHONUNBUFFERED=1 nohup python -u training/tier5_modmul.py --base 64 --amp --steps 80000 \
    --batch 64 --curriculum --curr-frac 0.85 --prime-pow 1.0 --self-check \
    --p-min 18446744073709551616 --p-max 340282366920938463463374607431768211456 \
    --d-model 512 --layers 10 --dim-ff 2048 \
    --ckpt training/checkpoints/modmul_t6_b64_d512.pt > t6_b64.log 2>&1 &
```
Watch: `tf_tok` climbing past ~0.94; the hardest log-bucket (top, ≈99.9% of the scorer's
value-uniform draw) climbing; `skip` stays 0; no collapse when `cur_pmax` opens to full range
(~step 60–70k). Best ckpt auto-saved as `..._best.pt` (gated on hardest bucket).

**24GB-card fixes baked in (2026-06-29; needed on a 4090, NOT a 5090):**
1. **`--batch 24`** (not 64). base-64 L=4347 forward+backward needs ~20.4GB at B=24 (B=16→13.8GB
   if tighter margin wanted). Memory is CONSTANT per step (all pad to max_len) → if step 1 fits,
   the whole run fits. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for fragmentation.
2. **`AbacusDecoder.forward` rewritten to the SDPA flash path** (`_sdpa_layer_step` loop). The old
   `nn.TransformerEncoder` + `triu(-inf)` float mask materialized O(B·H·L²) attention → OOM at
   L=4347. (Also: its `is_causal=True` hint is a NO-OP when `mask=None` → silently runs FULL
   non-causal attention; verified. So you can't just drop the mask.) New path matches the old to
   3e-4 (and P2 logit-parity) and is causal — validated before launch.
3. **`eval_answer` now uses `decode_cached`** (was inline nocache). At L=4347 nocache eval was
   ~40min/eval → the run would never finish. Cached + `--eval-every 4000 --eval-n 96` keeps eval
   to ~2h total over the run.

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
