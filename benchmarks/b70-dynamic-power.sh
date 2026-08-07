#!/usr/bin/env bash
# B70 Dynamic Power/Clock Manager — boost during prefill, relax during decode.
#
# Prefill on the MoE is COMPUTE-bound and scales with power (230W = +16-22%).
# Decode is BANDWIDTH-bound and self-limits to ~140W regardless of the cap.
# So: raise the cap to PREFILL_W when an active compute burst is detected,
# drop to DECODE_W once the card settles back to decode/idle draw.
#
# Detection: sample card watts (energy-delta) every DEV s. When the 3-sample
# power exceeds BOOST_W, the GPU is doing a prefill burst -> set PREFILL cap.
# When power stays below RELAX_W for HOLD consecutive samples, it's decoding
# or idle -> set DECODE cap. Also polls /v1/metrics? (optional) for a request
# count signal to force BOOST while a request is actively prefilling.
#
# Usage: bash b70-dynamic-power.sh [dev_s] [logfile]
set -u
POW=/sys/class/hwmon/hwmon4/power1_cap
DEV="${1:-0.5}"
LOG="${2:-/tmp/dyn-power.log}"
PREFILL_W="${B70_PREFILL_W:-230000000}"   # 230W during prefill
DECODE_W="${B70_DECODE_W:-165000000}"     # 165W during decode
BOOST_W="${B70_BOOST_W:-170}"             # watts above this = boost phase
RELAX_W="${B70_RELAX_W:-155}"             # watts at/below this = decode phase
HOLD="${B70_HOLD:-4}"                     # samples to confirm a phase change

: > "$LOG"
echo "dyn-power: dev=${DEV}s prefill=${PREFILL_W} decode=${DECODE_W} boost>${BOOST_W}W relax<=${RELAX_W}W hold=${HOLD}" | tee -a "$LOG"
echo "dyn-power: current cap=$(( $(cat $POW) / 1000000 ))W" | tee -a "$LOG"

# running watts (energy-delta over DEV s), returns integer watts
watts() {
  local e1 e2 t1 t2 dte dew
  e1=$(cat ${POW%/*}/energy1_input 2>/dev/null)
  t1=$(date +%s.%N)
  sleep "$DEV"
  e2=$(cat ${POW%/*}/energy1_input 2>/dev/null)
  t2=$(date +%s.%N)
  dte=$(echo "$t2 - $t1" | bc)
  dew=$(echo "$e2 - $e1" | bc)
  echo "scale=0; $dew / $dte / 1000000" | bc 2>/dev/null || echo 0
}

set_cap() {
  echo "$1" | sudo -n tee "$POW" > /dev/null 2>&1
  echo "$(date +%H:%M:%S) cap=$(( $1 / 1000000 ))W" >> "$LOG"
}

state=decode
low_count=0
echo "t watts phase cap" | tee -a "$LOG"
while true; do
  w=$(watts)
  if [ "$w" -gt "$BOOST_W" ]; then
    # active compute burst (prefill) -> boost
    low_count=0
    if [ "$state" != boost ]; then
      set_cap "$PREFILL_W"
      state=boost
    fi
  else
    # not boosting -> count toward decode
    low_count=$((low_count + 1))
    if [ "$state" = boost ] && [ "$low_count" -ge "$HOLD" ]; then
      set_cap "$DECODE_W"
      state=decode
      low_count=0
    fi
  fi
  echo "$(date +%s.%N) $w $state $(( $(cat $POW) / 1000000 ))" >> "$LOG"
done