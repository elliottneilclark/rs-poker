# CFR exploration diagnostics analyzer.
#
# Input: JSONL produced by the rsp binary with
#   RSP_DIAG_LOG=cfr_diag=trace ... 2> stats.jsonl
# Each line is a tracing-subscriber JSON event with shape:
#   { "timestamp": ..., "level": "TRACE",
#     "fields": { "depth": N, "stop_cause": "...", "final_iterations": N,
#                 "final_elapsed_us": N, "timer_armed": bool,
#                 "actions_considered": N, "regret_series": "[...]" },
#     "target": "cfr_diag" }
#
# Args:
#   --arg deadline_ms "250"  (default; the configured Deadline budget in ms)
#
# Output: three plain-text sections to stdout.

# Read all events into an array, ignoring non-JSON lines (e.g. human logs
# that may have slipped onto stderr) by skipping parse errors.
# -R (raw input) feeds each line as a string; fromjson? silently drops
# lines that are not valid JSON (e.g. cargo compile messages on stderr).
[inputs | fromjson? | select(.target == "cfr_diag")] as $events |

# ── Helpers ─────────────────────────────────────────────────────────

# Quantiles over a sorted numeric array.
def quantile(p):
    if length == 0 then null
    else
        (length - 1) as $n
        | ($n * p | floor) as $lo
        | ($n * p | ceil)  as $hi
        | (.[$lo] + .[$hi]) / 2
    end;

# Format a percentage as a fixed-width column.
def pct: . * 100 | . * 10 | round / 10 | tostring + "%";

# ── Section A: stop_cause × depth cross-tab ─────────────────────────
"stop_cause × depth (n=\($events | length) events)",
(
    ($events | group_by(.fields.depth) | map(.[0].fields.depth)) as $depths |
    ($events | group_by(.fields.stop_cause) | map(.[0].fields.stop_cause)) as $causes |
    (
        "  " + ($depths | map("depth=\(.)") | join("  "))
    ),
    (
        $causes[] as $cause |
        ($cause + "  " + (
            $depths | map(
                . as $d |
                ($events | map(select(.fields.depth == $d and .fields.stop_cause == $cause)) | length) as $n |
                ($events | map(select(.fields.depth == $d)) | length) as $total |
                if $total == 0 then "    -" else (($n / $total) | pct) end
            ) | join("  ")
        ))
    )
),
"",

# ── Section B: deadline utilization at depth=0 ──────────────────────
($deadline_ms | tonumber * 1000) as $deadline_us |
($events | map(select(.fields.depth == 0))) as $d0 |
"deadline utilization (depth=0, deadline=\($deadline_ms)ms, n=\($d0 | length))",
(
    ($d0 | map(.fields.final_elapsed_us / $deadline_us)) as $ratios |
    [
        ["[0-25%) ",  ($ratios | map(select(. < 0.25))                     | length)],
        ["[25-50%)",  ($ratios | map(select(. >= 0.25 and . < 0.5))        | length)],
        ["[50-75%)",  ($ratios | map(select(. >= 0.5  and . < 0.75))       | length)],
        ["[75-95%)",  ($ratios | map(select(. >= 0.75 and . < 0.95))       | length)],
        ["[95-100%)", ($ratios | map(select(. >= 0.95 and . < 1.0))        | length)],
        [">=100%  ",  ($ratios | map(select(. >= 1.0))                     | length)]
    ] as $buckets |
    ($d0 | length) as $total |
    $buckets[] | "\(.[0])  \(.[1])  " + (if $total == 0 then "-" else ((.[1] / $total) | pct) end)
),
"",

# ── Section C: final regret quantiles by depth ──────────────────────
"final regret last(regret_series) quantiles by depth",
"depth  n      p10        p50        p90        p99",
(
    ($events | group_by(.fields.depth)) as $by_depth |
    $by_depth[] as $group |
    ($group[0].fields.depth) as $d |
    (
        $group
        | map(.fields.regret_series | fromjson)
        | map(select(length > 0) | .[length-1])
        | sort
    ) as $finals |
    if ($finals | length) == 0 then
        "\($d)      0      -          -          -          -"
    else
        "\($d)      \($finals | length)      " +
        ($finals | quantile(0.10) | tostring) + "    " +
        ($finals | quantile(0.50) | tostring) + "    " +
        ($finals | quantile(0.90) | tostring) + "    " +
        ($finals | quantile(0.99) | tostring)
    end
),
"",

"convergence ratio last/first by depth (lower = converging more)",
"depth  n      p10        p50        p90",
(
    ($events | group_by(.fields.depth)) as $by_depth |
    $by_depth[] as $group |
    ($group[0].fields.depth) as $d |
    (
        $group
        | map(.fields.regret_series | fromjson)
        | map(select(length > 1))
        | map(.[length-1] / .[0])
        | sort
    ) as $ratios |
    if ($ratios | length) == 0 then
        "\($d)      0      -          -          -"
    else
        "\($d)      \($ratios | length)      " +
        ($ratios | quantile(0.10) | tostring) + "    " +
        ($ratios | quantile(0.50) | tostring) + "    " +
        ($ratios | quantile(0.90) | tostring)
    end
)
