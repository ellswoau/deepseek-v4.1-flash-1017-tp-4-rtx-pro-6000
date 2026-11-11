import json, sys
for l in sys.stdin:
    try: r = json.loads(l)
    except ValueError: print(l, end=""); continue
    print("  C=%2d  per-stream %6s (min %s)  total %7s tok/s  TTFT %ss  accept len %s" % (
        r["C"], r["decode_per_stream_median"], r["decode_per_stream_min"], r["total_out_tok_s"],
        r["ttft_median_s"], r["accept_len_reqweighted"]), flush=True)
