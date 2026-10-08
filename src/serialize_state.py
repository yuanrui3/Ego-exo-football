import json,math,sys
from pathlib import Path

def fmt(v): return 'NA' if v is None else (f'{v:.2f}' if isinstance(v,float) else str(v))
def serialize_state(r,include_answer=False):
    actor=str(r.get('actor_id','NA')); team=r.get('actor_team') or 'NA'; b=r.get('ball') or {}
    lines=[f"MATCH {r.get('match_id','NA')}",f"PERIOD {r.get('period','NA')}",f"TIME_MS {r.get('timestamp_ms','NA')}",f"POSSESSION {r.get('possession_team') or 'NA'}",f"ACTOR {actor}",f"ACTOR_TEAM {team}",f"BALL x={fmt(b.get('x'))} y={fmt(b.get('y'))} z={fmt(b.get('z'))} vx={fmt(b.get('vx'))} vy={fmt(b.get('vy'))} vz={fmt(b.get('vz'))}"]
    players=list(r.get('players') or [])
    def rank(p):
        pid=str(p.get('player_id')); g=0 if pid==actor else 1 if p.get('team')==team else 2 if p.get('team') in ('left','right') else 3
        return (g,pid)
    for p in sorted(players,key=rank): lines.append(f"PLAYER id={p.get('player_id','NA')} track={p.get('track_id','NA')} team={p.get('team') or 'NA'} role={p.get('role') or 'NA'} jersey={p.get('jersey','NA')} x={fmt(p.get('x'))} y={fmt(p.get('y'))} vx={fmt(p.get('vx'))} vy={fmt(p.get('vy'))}")
    if include_answer: lines.append(f"ACTION {r['action']['label']}")
    return '\n'.join(lines)
if __name__=='__main__':
    for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
        if line.strip(): print('='*72); print(serialize_state(json.loads(line),True))
