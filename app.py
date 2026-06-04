import json
import os
import random
from flask import Flask, jsonify, render_template

app = Flask(__name__)

GLOBAL_PATIENTS = []

def load_initial_data():
    global GLOBAL_PATIENTS
    try:
        with open('patients.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            GLOBAL_PATIENTS = data.get('patients', [])
        print(f"✅ {len(GLOBAL_PATIENTS)}명 환자 데이터 로드 완료")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")

load_initial_data()

POSTURE_SPOT_WEIGHTS = {
    "Supine": {"occiput":0.8,"scapula_l":0.5,"scapula_r":0.5,"sacrum":1.0,"heel_l":0.6,"heel_r":0.6},
    "Left":   {"occiput":0.2,"scapula_l":0.9,"scapula_r":0.0,"sacrum":0.4,"heel_l":0.9,"heel_r":0.1},
    "Right":  {"occiput":0.2,"scapula_l":0.0,"scapula_r":0.9,"sacrum":0.4,"heel_l":0.1,"heel_r":0.9},
}

def simulate_logical_changes():
    global GLOBAL_PATIENTS
    postures = ["Left", "Right", "Supine"]
    action_labels = {"Left":"좌측위","Right":"우측위","Supine":"앙와위"}

    for p in GLOBAL_PATIENTS:
        max_time = p.get('maxAllowedMinutes', 120)
        elapsed  = p.get('elapsedMinutes', 0)

        # ── 1. 체위 변경 판단 ──────────────────────────────────
        needs_change = elapsed >= max_time or p.get('status') == 'immediate'

        if needs_change:
            recommended = p.get('recommendedPosition')
            if not recommended or recommended == p.get('currentPosition'):
                candidates = [pos for pos in postures if pos != p.get('currentPosition')]
                recommended = random.choice(candidates)
            p['currentPosition'] = recommended
            p['elapsedMinutes']  = 0
        else:
            p['elapsedMinutes'] = elapsed + 15

        # ── 2. 히트맵 업데이트 ────────────────────────────────
        if not p.get('heatmap'):
            p['heatmap'] = [{"p": p['currentPosition'], "m": 15}]
        else:
            last = p['heatmap'][-1]
            if last['p'] == p['currentPosition']:
                last['m'] += 15
            else:
                p['heatmap'].append({"p": p['currentPosition'], "m": 15})
                if len(p['heatmap']) > 6:
                    p['heatmap'].pop(0)

        # ── 3. 위험 점수 계산 (버그 수정: 키 이름 통일) ────────
        if 'riskFactors' in p:
            time_ratio = p['elapsedMinutes'] / max_time
            p['riskFactors']['pressureDuration'] = min(100, int(time_ratio * 100))

            f = p['riskFactors']
            # 기존 코드의 키 이름 불일치 버그 수정
            calculated = (
                f.get('pressureDuration', 0) * 0.40 +
                f.get('frictionShear',    0) * 0.20 +  # 'friction' → 'frictionShear'
                f.get('moisture',         0) * 0.15 +
                f.get('mobility',         0) * 0.15 +
                f.get('nutrition',        0) * 0.10
            )
            p['riskScore'] = round(min(100, max(0, calculated)))

        # ── 4. 신체 부위 위험도 갱신 ──────────────────────────
        if 'spotRisks' not in p:
            p['spotRisks'] = {k:0 for k in POSTURE_SPOT_WEIGHTS['Supine']}

        cur_pos  = p['currentPosition']
        weights  = POSTURE_SPOT_WEIGHTS.get(cur_pos, POSTURE_SPOT_WEIGHTS['Supine'])
        pressure = p.get('riskFactors', {}).get('pressureDuration', 0)
        braden   = p.get('bradenScore', 15)
        vuln     = max(0, (23 - braden) * 1.5)

        for spot, w in weights.items():
            target  = min(100, pressure * w + vuln)
            current = p['spotRisks'].get(spot, 0)
            p['spotRisks'][spot] = round(current + (target - current) * 0.3)

        # ── 5. 상태 갱신 ──────────────────────────────────────
        score = p.get('riskScore', 0)
        p['status'] = 'immediate' if score >= 70 else 'caution' if score >= 40 else 'stable'

        # ── 6. 권장 체위 도출 ─────────────────────────────────
        usage = {"Left":0,"Right":0,"Supine":0}
        for seg in p.get('heatmap', []):
            usage[seg.get('p','Supine')] += seg.get('m', 1)
        cands = [pos for pos in postures if pos != p['currentPosition']]
        p['recommendedPosition'] = min(cands, key=lambda pos: usage.get(pos, 0))

        # ── 7. 행동 지침 ──────────────────────────────────────
        label = action_labels.get(p['recommendedPosition'], p['recommendedPosition'])
        if p['status'] == 'immediate':
            p['recommendedAction'] = f"{label}로 즉시 변경"
        elif p['status'] == 'caution':
            p['recommendedAction'] = f"{label}로 변경 준비"
        else:
            p['recommendedAction'] = "상태 재평가"

    # ── 8. 우선순위 정렬 ──────────────────────────────────────
    GLOBAL_PATIENTS.sort(key=lambda x: x.get('riskScore', 0), reverse=True)
    for i, p in enumerate(GLOBAL_PATIENTS):
        p['priority'] = i + 1


# ── API ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/patients', methods=['GET'])
def get_patients():
    simulate_logical_changes()
    return jsonify(GLOBAL_PATIENTS)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)