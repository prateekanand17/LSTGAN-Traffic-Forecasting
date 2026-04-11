def speed_to_color(speed):
    if speed >= 55: return '#00b894'
    elif speed >= 45: return '#55efc4'
    elif speed >= 35: return '#fdcb6e'
    elif speed >= 25: return '#e17055'
    elif speed >= 15: return '#d63031'
    else: return '#6c5ce7'

def speed_to_level(speed):
    if speed >= 50: return '🟢 Free Flow'
    elif speed >= 35: return '🟡 Moderate'
    elif speed >= 20: return '🔴 Heavy'
    else: return '⛔ Severe'

def speed_to_badge_color(speed):
    if speed >= 50: return '#00b894'
    elif speed >= 35: return '#fdcb6e'
    elif speed >= 20: return '#e17055'
    else: return '#d63031'
