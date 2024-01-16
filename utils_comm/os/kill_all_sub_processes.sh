#!/bin/bash
# This function will kill all sub jobs.
function KillJobs() {
  [[ -z "$(jobs -p)" ]] && return # no jobs to kill
  local SIG="INT" # default to a gentle goodbye
  [[ ! -z "$1" ]] && SIG="$1" # optionally send a different signal
  # my version of 'kill' doesn't seem to understand `kill -- -${PID}`
  #jobs -p | xargs -I%% kill -s "$SIG" -- -%% # kill each job's processes group
  jobs -p | xargs kill -s "$SIG" # kill each job's processes group
  
  ## give the processes a moment to die, before forcing them to.
  [[ "$SIG" != "KILL" ]] && {
    sleep 0.2
    KillJobs "KILL"
  }
}