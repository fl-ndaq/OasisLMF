#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

error_handler(){
   echo 'Run Error - terminating'
   exit_code=$?
   set +x
   group_pid=$(ps -p $$ -o pgid --no-headers)
   sess_pid=$(ps -p $$ -o sess --no-headers)
   printf "Script PID:%d, GPID:%s, SPID:%d" $$ $group_pid $sess_pid >> log/killout.txt

   if hash pstree 2>/dev/null; then
       pstree -pn $$ >> log/killout.txt
       PIDS_KILL=$(pstree -pn $$ | grep -o "([[:digit:]]*)" | grep -o "[[:digit:]]*")
       kill -9 $(echo "$PIDS_KILL" | grep -v $group_pid | grep -v $$) 2>/dev/null
   else
       ps f -g $sess_pid > log/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | grep -v celery | grep -v $group_pid | grep -v $$)
       echo "$PIDS_KILL" >> log/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   fi
   exit $(( 1 > $exit_code ? 1 : $exit_code ))
}
trap error_handler QUIT HUP INT KILL TERM ERR

touch log/stderror.err
ktools_monitor.sh $$ & pid0=$!

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc

mkfifo fifo/gul_P21

mkfifo fifo/il_P21

mkfifo fifo/gul_S1_summary_P21
mkfifo fifo/gul_S1_summaryeltcalc_P21
mkfifo fifo/gul_S1_eltcalc_P21
mkfifo fifo/gul_S1_summarysummarycalc_P21
mkfifo fifo/gul_S1_summarycalc_P21
mkfifo fifo/gul_S1_summarypltcalc_P21
mkfifo fifo/gul_S1_pltcalc_P21

mkfifo fifo/il_S1_summary_P21
mkfifo fifo/il_S1_summaryeltcalc_P21
mkfifo fifo/il_S1_eltcalc_P21
mkfifo fifo/il_S1_summarysummarycalc_P21
mkfifo fifo/il_S1_summarycalc_P21
mkfifo fifo/il_S1_summarypltcalc_P21
mkfifo fifo/il_S1_pltcalc_P21



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_summaryeltcalc_P21 > work/kat/il_S1_eltcalc_P21 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P21 > work/kat/il_S1_summarycalc_P21 & pid2=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P21 > work/kat/il_S1_pltcalc_P21 & pid3=$!
tee < fifo/il_S1_summary_P21 fifo/il_S1_summaryeltcalc_P21 fifo/il_S1_summarypltcalc_P21 fifo/il_S1_summarysummarycalc_P21 work/il_S1_summaryaalcalc/P21.bin work/il_S1_summaryleccalc/P21.bin > /dev/null & pid4=$!
( summarycalc -f  -1 fifo/il_S1_summary_P21 < fifo/il_P21 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_summaryeltcalc_P21 > work/kat/gul_S1_eltcalc_P21 & pid5=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P21 > work/kat/gul_S1_summarycalc_P21 & pid6=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P21 > work/kat/gul_S1_pltcalc_P21 & pid7=$!
tee < fifo/gul_S1_summary_P21 fifo/gul_S1_summaryeltcalc_P21 fifo/gul_S1_summarypltcalc_P21 fifo/gul_S1_summarysummarycalc_P21 work/gul_S1_summaryaalcalc/P21.bin work/gul_S1_summaryleccalc/P21.bin > /dev/null & pid8=$!
( summarycalc -i  -1 fifo/gul_S1_summary_P21 < fifo/gul_P21 ) 2>> log/stderror.err  &

( eve 21 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P21 | fmcalc -a2 > fifo/il_P21  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P21 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P21 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P21 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P21 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P21 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P21 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


# Stop ktools watcher
kill -9 $pid0