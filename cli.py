#!/usr/bin/env python3
"""
cli.py — ChiralCheck + Idea Auditor  |  by William Ashioya / SHOAL

QUICK START:
  1. pip install -r requirements.txt
  2. cp .env.example .env  (add ANTHROPIC_API_KEY + VOYAGE_API_KEY)
  3. python cli.py --preflight
  4. python cli.py --quicktest
  5. python cli.py --input your_paper.txt --plot

COMMANDS:
  python cli.py --text "idea..."           score an idea
  python cli.py --input paper.txt --plot   score a file with 3D plot
  python cli.py --input paper.txt --json   machine-readable output
  python cli.py --chiral "thesis..."       ChiralCheck only (stability)
  python cli.py --agent "audit paper.txt"  autonomous Dispatch agent mode
  python cli.py --preflight                validate API keys
  python cli.py --quicktest                smoke test (Newton F=ma)
  python cli.py --providers                show configured keys
  python cli.py --calibrate                print raw log-det for tuning
  python cli.py --compare --text "..."     all providers side-by-side
"""

import argparse, json, sys
from auditor import audit, load, auto_config
from providers import available, preflight as pf_check

W = 68

def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _y(s): return f"\033[33m{s}\033[0m"
def _b(s): return f"\033[36m{s}\033[0m"
def _bold(s): return f"\033[1m{s}\033[0m"

def _bar(s, w=24):
    if s is None: return _y("░"*w + "  N/A")
    f = int(s/100*w)
    bar = "█"*f + "░"*(w-f) + f"  {s}/100"
    return (_g(bar) if s>=70 else _y(bar) if s>=45 else _r(bar))

def _sec(t):  print(f"\n  ┌─ {_b(t)} {'─'*(W-len(t)-6)}┐")
def _row(l,v=""): print(f"  │  {l:<26}{v}")
def _div():   print(f"  └{'─'*(W-3)}┘")
def _wrap(label, text, w=42):
    words,line,first = text.split(),[],True
    for word in words:
        if sum(len(x)+1 for x in line)+len(word)>w:
            print(f"  {'  '+label if first else ' '*28}{' '.join(line)}")
            first=False; line=[word]
        else: line.append(word)
    if line: print(f"  {'  '+label if first else ' '*28}{' '.join(line)}")

def _plot(u,o,s,st):
    W2,H2=64,26; canvas=[[" "]*W2 for _ in range(H2)]
    def draw(x,y,c):
        if 0<=y<H2 and 0<=x<W2: canvas[y][x]=c
    def ds(x,y,s2):
        for i,c in enumerate(s2): draw(x+i,y,c)
    ox,oy,sc=12,20,0.16
    def iso(a,b,c):
        return int(ox+(a*0.80+b*0.0+c*(-0.60))*sc), int(oy+(a*(-0.40)+b*(-0.80)+c*(-0.30))*sc)
    for t in range(0,101,20):
        for uv,ov,sv in [(t,0,0),(0,t,0),(0,0,t)]: draw(*iso(uv,ov,sv),"·")
    ds(*iso(105,0,0),"UTILITY →"); ds(*iso(0,108,0),"↑ ORIG"); ds(*iso(0,0,108),"SPEC ←")
    px,py=iso(u,o,s); _,by=iso(u,0,s)
    for dy in range(py+1,by+1): draw(px,dy,"┊")
    draw(px,py,"◉" if st and st>=75 else "◈" if st and st>=50 else "✖")
    ds(px+1,py-1,f"({u},{o},{s})")
    return "\n".join("".join(row) for row in canvas)

QUICKTEST = (
    "A net force applied to a body of mass m produces acceleration F=ma. "
    "Validated by centuries of experiment across all scales of classical mechanics."
)

def display(result: dict, plot: bool = False):
    a=result["axes"]; c=result["composite"]; p=result.get("providers",{})
    print(f"\n  {'═'*W}")
    print(f"  {_bold('CHIRALCHECK // IDEA AUDITOR')}  ·  SHOAL")
    print(f"  {_b(p.get('llm','?'))}  +  {_b(p.get('embed','?'))}")
    print(f"  \"{result['idea_preview'][:63]}\"")
    if result.get("quick_mode"): print(f"  {_y('[quick — 4 perturbations]')}")

    for label, key in [
        ("UTILITY — claim coherence / EC proxy",      "utility"),
        ("ORIGINALITY — distance from prior-art",      "originality"),
        ("SPECIFICITY — known / total subtasks",        "specificity"),
        ("STABILITY — ChiralCheck covariance geometry","stability"),
    ]:
        d=a.get(key,{}); sc=d.get("score")
        _sec(label); _row("Score:", _bar(sc))
        _wrap("", d.get("interpretation") or d.get("error","—"))

        if key=="utility":
            for lb in (d.get("load_bearing") or [])[:2]: _wrap("├ load-bearing:", lb[:52])
        if key=="specificity":
            for op in (d.get("open_problems") or [])[:3]: _wrap("│  ✖ open:", op.get("subtask","")[:52])
        if key=="originality" and d.get("closest_prior_art"):
            _wrap("├ closest:", d["closest_prior_art"][:55])
        if key=="stability":
            m=d.get("metrics",{})
            if m:
                _row("│ log-det:", str(m.get("log_det_normalised","?")))
                _row("│ entropy:", f"{m.get('semantic_entropy_nats','?')} nats")
                _row("│ clusters:", str(m.get("semantic_clusters","?")))
                _row("│ T1 IR (legacy):", str(m.get("legacy_ir_cosine","?")))
            if d.get("semantic_illusion_warning"): _row("│ ⚠ T1 IR said STABLE", "geometry disagrees")
        _div()

    print(f"\n  ┌─ COMPOSITE {'─'*(W-14)}┐")
    print(f"  │  {_bar(c.get('composite'),30)}")
    print(f"  │  utility×0.35 · originality×0.25 · specificity×0.25 · stability×0.15")
    if c.get("axes_failed"): print(f"  │  {_y('reweighted — failed: '+', '.join(c['axes_failed']))}")
    print(f"  └  {result.get('elapsed_s','?')}s total\n")

    if plot:
        u=(a.get("utility") or {}).get("score")
        o=(a.get("originality") or {}).get("score")
        s=(a.get("specificity") or {}).get("score")
        st=(a.get("stability") or {}).get("score")
        if all(v is not None for v in [u,o,s]):
            print(f"  3D POSITION  (Utility, Originality, Specificity)")
            for line in _plot(u,o,s,st or 50).splitlines(): print("  "+line)
            print()

def _calibrate(cfg):
    import chiral
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cases = [
        ("Newton F=ma (expect: stable, log-det << -6.5)", QUICKTEST),
        ("Quantum consciousness (expect: fragile, > -5.5)",
         "Quantum effects in microtubules bridge physics and subjective experience, explaining consciousness and free will."),
    ]
    print(f"\n  Calibration  ({cfg.describe()})\n  {'─'*56}")
    for label, thesis in cases:
        print(f"\n  {label}")
        texts=[thesis]
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs=[ex.submit(chiral._perturb, thesis, a, t, cfg.llm) for a,t in chiral._ANGLES[:6]]
            for f in as_completed(futs):
                try:
                    r=f.result()
                    if r: texts.append(r)
                except Exception: pass
        vecs=cfg.embed.embed(texts, input_type="document")
        c2=chiral._clip(vecs[1:])
        print(f"  log-det: {round(chiral._log_det(c2),4)}")
        print(f"  entropy: {round(chiral._entropy(c2),4)}")
        print(f"  perturbations: {len(texts)-1}")
    print(f"\n  Adjust _to_score() in chiral.py if these are off.\n")

def main():
    p=argparse.ArgumentParser(description="ChiralCheck + Idea Auditor | SHOAL")
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text"); g.add_argument("--input")
    g.add_argument("--chiral", metavar="THESIS")
    g.add_argument("--agent",  metavar="TASK", help="Dispatch autonomous agent mode")
    g.add_argument("--preflight",  action="store_true")
    g.add_argument("--quicktest",  action="store_true")
    g.add_argument("--providers",  action="store_true")
    g.add_argument("--calibrate",  action="store_true")
    p.add_argument("--llm");  p.add_argument("--embed")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--json",  action="store_true")
    p.add_argument("--plot",  action="store_true")
    args=p.parse_args()

    if args.providers:
        av=available()
        print("\n  Configured providers:")
        for k,v in av.items(): print(f"  {'  '+_g('✓') if v else '  '+_r('✗')}  {k}")
        return

    if args.agent:
        from agent import run_dispatch_task
        report = run_dispatch_task(args.agent, verbose=True)
        print("\n" + "═"*W + "\n  DISPATCH AGENT REPORT\n" + "═"*W)
        print(report)
        print("═"*W + "\n")
        return

    try:
        cfg = load(args.llm, args.embed) if (args.llm or args.embed) else auto_config()
    except ValueError as e:
        print(f"\n{_r('Config error:')}\n{e}\n"); sys.exit(1)

    if args.preflight:
        print(f"\n  Preflight  ({cfg.describe()})...")
        ok, msg = pf_check(cfg.llm_name, cfg.embed_name)
        print(msg)
        if ok: print(f"\n  {_g('Ready.')}  Run --quicktest next.\n")
        else: sys.exit(1)
        return

    if args.calibrate:
        _calibrate(cfg); return

    if args.chiral:
        import chiral
        r=chiral.audit(args.chiral, cfg.llm, cfg.embed)
        if args.json: print(json.dumps(r,indent=2))
        else:
            sc=r.get("score"); vd=r.get("verdict","?")
            print(f"\n  Score:   {_bar(sc)}")
            print(f"  Verdict: {(_g if sc and sc>=75 else _y if sc and sc>=50 else _r)(vd)}")
            print(f"  {r.get('interpretation','')}\n")
        return

    if args.quicktest:
        print(f"\n  {_b('Quick test')}  (Newton F=ma — expect: stability ≥ 60, utility ≥ 50)\n")
        r=audit(QUICKTEST, cfg, verbose=True, quick=True)
        if not args.json:
            display(r, plot=False)
            st=(r["axes"].get("stability") or {}).get("score")
            ut=(r["axes"].get("utility")   or {}).get("score")
            ok=(st is not None and st>=60) and (ut is not None and ut>=50)
            print(f"  Quick test: {_g('PASS') if ok else _r('FAIL')}")
            if ok: print(f"  {_g('Tool working.')}  Run --input your_paper.txt --plot next.\n")
            else:  print(f"  {_r('Check keys and run --calibrate.')}\n")
        else: print(json.dumps(r,indent=2))
        return

    if args.compare:
        av=available()
        pairs=[]
        if av.get("anthropic") and av.get("voyage"): pairs.append(("anthropic","voyage"))
        if av.get("openai"):                         pairs.append(("openai","openai"))
        if av.get("gemini"):                         pairs.append(("gemini","gemini"))
        if not pairs: print("No providers."); sys.exit(1)
        text=open(args.input).read() if args.input else (args.text or "")
        if not text.strip(): print("No input."); sys.exit(1)
        all_r=[]
        for lp,ep in pairs:
            try:
                c2=load(lp,ep); r=audit(text,c2,verbose=False); r["pl"]=f"{lp}+{ep}"; all_r.append(r)
                if not args.json: display(r,plot=False)
            except Exception as e: print(f"  {_r('ERROR')} {lp}+{ep}: {e}")
        if len(all_r)>1 and not args.json:
            print(f"\n  {'Provider':<22}{'Util':>5}{'Orig':>5}{'Spec':>5}{'Stab':>5}{'Comp':>6}")
            print("  "+"─"*48)
            for r in all_r:
                a=r["axes"]
                print(f"  {r['pl']:<22}"
                      +"".join(f"{str((a.get(k) or {}).get('score','?')):>5}"
                                for k in ["utility","originality","specificity","stability"])
                      +f"{str(r['composite'].get('composite','?')):>6}")
        return

    text=open(args.input).read() if args.input else (args.text or "")
    if not text.strip(): print("Empty input."); sys.exit(1)
    r=audit(text, cfg, verbose=not args.json)
    if args.json: print(json.dumps(r,indent=2))
    else: display(r, plot=args.plot)

if __name__=="__main__": main()
