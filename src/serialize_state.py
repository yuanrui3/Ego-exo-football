import json,math
from pathlib import Path

def fmt(v): return 'NA' if v is None else (f'{v:.2f}' if isinstance(v,float) else str(v))

def serialize_identity_state(r,include_answer=False):
    actor=str(r.get('actor_id','NA')); team=r.get('actor_team') or 'NA'; b=r.get('ball') or {}
    lines=[f"MATCH {r.get('match_id','NA')}",f"PERIOD {r.get('period','NA')}",f"TIME_MS {r.get('timestamp_ms','NA')}",f"POSSESSION {r.get('possession_team') or 'NA'}",f"ACTOR {actor}",f"ACTOR_TEAM {team}",f"BALL x={fmt(b.get('x'))} y={fmt(b.get('y'))} z={fmt(b.get('z'))} vx={fmt(b.get('vx'))} vy={fmt(b.get('vy'))} vz={fmt(b.get('vz'))}"]
    players=list(r.get('players') or [])
    def rank(p):
        pid=str(p.get('player_id')); g=0 if pid==actor else 1 if p.get('team')==team else 2 if p.get('team') in ('left','right') else 3
        return (g,pid)
    for p in sorted(players,key=rank): lines.append(f"PLAYER id={p.get('player_id','NA')} track={p.get('track_id','NA')} team={p.get('team') or 'NA'} role={p.get('role') or 'NA'} jersey={p.get('jersey','NA')} x={fmt(p.get('x'))} y={fmt(p.get('y'))} vx={fmt(p.get('vx'))} vy={fmt(p.get('vy'))}")
    if include_answer: lines.append(f"ACTION {r['action']['label']}")
    return '\n'.join(lines)

def player_distance(player,actor):
    try:
        return math.hypot(float(player['x'])-float(actor['x']),float(player['y'])-float(actor['y']))
    except (KeyError,TypeError,ValueError):
        return math.inf

def serialize_anonymous_state(r,include_answer=False):
    players=list(r.get('players') or [])
    actor_id=r.get('actor_id')
    actor_index=next((i for i,p in enumerate(players) if actor_id is not None and str(p.get('player_id'))==str(actor_id)),None)
    actor=players[actor_index] if actor_index is not None else {}
    actor_team=r.get('actor_team') or actor.get('team')
    lines=[]
    if r.get('period') is not None: lines.append(f"PERIOD {r['period']}")
    lines.append(f"ACTOR team={actor_team or 'NA'} role={actor.get('role') or 'NA'} x={fmt(actor.get('x'))} y={fmt(actor.get('y'))} vx={fmt(actor.get('vx'))} vy={fmt(actor.get('vy'))}")

    others=[(i,p) for i,p in enumerate(players) if i!=actor_index]
    for teammate,team_label in ((True,'TEAMMATE'),(False,'OPPONENT')):
        group=[(i,p) for i,p in others if bool(actor_team) and p.get('team')==actor_team] if teammate else [(i,p) for i,p in others if not (bool(actor_team) and p.get('team')==actor_team)]
        group.sort(key=lambda item:(player_distance(item[1],actor),item[0]))
        for number,(_,player) in enumerate(group,1):
            lines.append(f"{team_label}_{number} team={player.get('team') or 'NA'} role={player.get('role') or 'NA'} x={fmt(player.get('x'))} y={fmt(player.get('y'))} vx={fmt(player.get('vx'))} vy={fmt(player.get('vy'))}")
    if include_answer: lines.append(f"ACTION {r['action']['label']}")
    return '\n'.join(lines)

def serialize_state(r,include_answer=False,mode='identity'):
    if mode=='identity': return serialize_identity_state(r,include_answer)
    if mode=='anonymous': return serialize_anonymous_state(r,include_answer)
    raise ValueError(f"Unknown serializer mode: {mode}")

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--mode',choices=['identity','anonymous'],default='identity'); a=ap.parse_args()
    for line in Path(a.input).read_text(encoding='utf-8').splitlines():
        if line.strip(): print('='*72); print(serialize_state(json.loads(line),True,a.mode))
