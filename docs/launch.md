# Launch artefacts

## The post

One number, one link, one honest caveat. No thread.

> Your eval dashboard flags a regression. You bisect for a week. The system never changed —
> the provider rotated the judge snapshot underneath you.
>
> I measured how often the standard approach gets this wrong. A t-test re-run on every CI
> commit raises a false alarm on **20.6% of perfectly healthy pipelines**, and when only the
> judge moved it says "regression" **100% of the time**.
>
> Benchlock holds a frozen anchor set your system never touches, so if those scores move,
> only the judge can have moved them. 0.0% false alarms, correct attribution 100% of the
> time.
>
> The honest caveat: it detects a real regression about **8× slower** than the invalid test.
> That is the price of a guarantee that survives daily peeking, and it is in the results
> table, not a footnote.
>
> Six of eight attacks still beat it. Those are published too.
>
> github.com/…/benchlock

## The terminal recording

30 seconds, Demo 1. Record with `asciinema rec`, keep it to these five commands:

```sh
benchlock plan --target-shift 0.05      # 200 anchors, $0.07/run
benchlock baseline --anchors suite.jsonl
benchlock observe results.jsonl --rescore-anchors
benchlock verdict                        # verdict=judge
benchlock gate; echo $?                  # 0 — a judge change does not fail your build
```

The beat to land: `verdict=judge` followed by exit code `0`. The tool telling you *not* to
act is the point.

## The diagram

[`docs/assets/mechanism.svg`](assets/mechanism.svg) — two streams, one judge, four answers,
with `indeterminate` outlined because it is the one that makes the other three mean
anything. Renders in both light and dark GitHub themes.

## What not to do

- No "enterprise-grade". No "revolutionise". The tool tells you which of two things moved;
  it does not make your evals correct.
- Do not lead with the false-alarm number alone. Lead with `indeterminate`, or with the
  8× delay. The interesting claim is that the tool refuses to answer when it cannot know,
  and that its honest cost is published.
- Credit the preprint (arXiv 2606.15474) in the first paragraph of any technical write-up.
  It got here first.
