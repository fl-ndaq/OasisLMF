#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryaalcalc

mkfifo fifo/gul_P12

mkfifo fifo/gul_S1_summary_P12



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P12 work/gul_S1_summaryaalcalc/P12.bin > /dev/null & pid1=$!
summarycalc -g  -1 fifo/gul_S1_summary_P12 < fifo/gul_P12 &

eve 12 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P12  &

wait $pid1


# --- Do ground up loss kats ---
