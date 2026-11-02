"""Owned Docker child guard: GNU deadline plus identity-checked group cleanup."""
from pathlib import Path
import argparse,json,os,signal,time

ROOT=Path(__file__).resolve().parent

def start_ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def stop(state_path):
    state=json.loads(state_path.read_text());pid=state['pid']
    ticks=start_ticks(pid)
    if ticks is None or ticks!=state['start_ticks']:
        return {'status':'already-exited-or-reused','pid':pid,'group_signal_sent':False}
    try:actual=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return {'status':'already-exited-or-reused','pid':pid,'group_signal_sent':False}
    actual=[x.decode()for x in actual if x]
    if actual!=state['argv']:
        # The state is published just before exec; at that point no CUDA
        # child exists and only this exact owned guard PID may be stopped.
        if str(Path(__file__).resolve())not in actual or str(state_path)not in actual:
            raise RuntimeError('Owned remote PID command identity changed; no signal sent')
        os.kill(pid,signal.SIGTERM)
        return {'status':'guard-stopped-before-exec','pid':pid,'group_signal_sent':False}
    if os.getpgid(pid)!=pid:raise RuntimeError('Owned timeout process-group identity differs')
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:pass
    deadline=time.monotonic()+2.
    while alive(pid)and time.monotonic()<deadline:time.sleep(.02)
    # Signal the same owned group even if the leader is now a zombie; its
    # same-start-time identity still protects against PID/group reuse.
    if start_ticks(pid)==state['start_ticks']:
        try:os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    return {'status':'owned-group-terminated','pid':pid,'group_signal_sent':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--stop',action='store_true');p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args()
    state=a.state.resolve()
    if not state.is_relative_to(ROOT)or state==ROOT:raise RuntimeError('State must remain in the owned controller directory')
    if a.stop:
        print(json.dumps(stop(state))if state.exists()else json.dumps({'status':'not-started','group_signal_sent':False}));return
    argv=a.argv[1:]if a.argv[:1]==['--']else a.argv
    if state.exists()or not state.parent.is_dir()or not 1<=a.seconds<=3600:raise RuntimeError('Fresh state and bounded positive deadline required')
    if not argv or argv[0]!='taskset':raise RuntimeError('Only the explicit pinned native command is supported')
    command=['/usr/bin/timeout','--signal=TERM','--kill-after=15s',f'{a.seconds}s']+argv
    pid=os.getpid();ticks=start_ticks(pid)
    if ticks is None:raise RuntimeError('Owned guard identity is not observable')
    state.write_text(json.dumps({'pid':pid,'start_ticks':ticks,'argv':command},indent=2)+'\n')
    os.execv(command[0],command)

if __name__=='__main__':main()
