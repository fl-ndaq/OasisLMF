#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summarysummarycalc_P3
mkfifo fifo/il_S1_summarycalc_P3

mkfifo il_S1_summary_P3
mkfifo il_S1_summarysummarycalc_P3
mkfifo il_S1_summarycalc_P3



# --- Do insured loss computes ---
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid1=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_summarysummarycalc_P3 > /dev/null & pid2=$!
summarycalc -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &

eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P3 -a1 -i - | fmcalc -a2 > fifo/il_P3  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P3 > il_P3 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
summarycalctocsv -s < il_S1_summarysummarycalc_P3 > work/full_correlation/kat/il_S1_summarycalc_P3 & pid1=$!
tee < il_S1_summary_P3 il_S1_summarysummarycalc_P3 > /dev/null & pid2=$!
summarycalc -f  -1 il_S1_summary_P3 < il_P3 &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_summarycalc_P3 > output/il_S1_summarycalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_summarycalc_P3 > output/full_correlation/il_S1_summarycalc.csv & kpid2=$!
wait $kpid1 $kpid2
