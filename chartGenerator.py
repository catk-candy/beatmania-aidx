import librosa
import numpy as np
import json
import random
import sys
import os
from scipy.signal import butter, lfilter

# --- 定数 ---
PLEASANT_CHORDS_2 = [[1,3], [3,5], [5,7], [2,4], [4,6], [1,7]]
PLEASANT_CHORDS_3 = [[1,3,5], [3,5,7], [1,2,3], [5,6,7], [2,4,6], [1,4,7]]

def analyze_audio_structure(y, sr, bpm, offset_sec, total_bars):
    """ 音量バランスからセクションを判定 """
    print("Analyzing song structure...")
    nyquist = 0.5 * sr
    b_low, a_low = butter(2, 200 / nyquist, btype='low')
    y_low = lfilter(b_low, a_low, y)
    
    seconds_per_bar = (60.0 / bpm) * 4
    samples_per_bar = int(seconds_per_bar * sr)
    start_sample = int(offset_sec * sr)
    
    bar_stats = []
    for i in range(total_bars):
        start = start_sample + i * samples_per_bar
        end = start + samples_per_bar
        if start >= len(y): break
        end = min(end, len(y))
        
        c_full = y[start:end]
        c_low = y_low[start:end]
        if len(c_full) == 0:
            bar_stats.append({'full': 0, 'low': 0})
            continue
        
        rms_full = np.sqrt(np.mean(c_full**2))
        rms_low = np.sqrt(np.mean(c_low**2))
        bar_stats.append({'full': rms_full, 'low': rms_low})

    if not bar_stats: return ['verse'] * total_bars

    max_full = max((s['full'] for s in bar_stats), default=1) or 1
    max_low = max((s['low'] for s in bar_stats), default=1) or 1
    
    raw_sections = []
    for s in bar_stats:
        nf = s['full'] / max_full
        nl = s['low'] / max_low
        if nf > 0.7 and nl > 0.6: sec = 'drop'
        elif nf > 0.4 and nl < 0.5: sec = 'buildup'
        elif nf < 0.3: sec = 'break'
        else: sec = 'verse'
        raw_sections.append(sec)

    refined = raw_sections.copy()
    for i in range(min(4, len(refined))):
        if refined[i] != 'drop': refined[i] = 'intro'
    outro_len = min(4, len(refined))
    for i in range(len(refined)-outro_len, len(refined)):
        refined[i] = 'outro'
    
    for i in range(1, len(refined)-1):
        if refined[i-1] == refined[i+1] and refined[i] != refined[i-1]:
            if refined[i-1] == 'drop': refined[i] = 'drop'
            elif refined[i] != 'break': refined[i] = refined[i-1]
            
    return refined

def get_band_onset_strengths(y, sr):
    """ 3帯域（Low, Mid, High）ごとのオンセット強度 """
    nyquist = 0.5 * sr
    b_l, a_l = butter(2, 200/nyquist, btype='low')
    b_m, a_m = butter(2, [200/nyquist, 2000/nyquist], btype='band')
    b_h, a_h = butter(2, 2000/nyquist, btype='high')
    
    y_low = lfilter(b_l, a_l, y)
    y_mid = lfilter(b_m, a_m, y)
    y_high = lfilter(b_h, a_h, y)
    
    onset_l = librosa.onset.onset_strength(y=y_low, sr=sr)
    onset_m = librosa.onset.onset_strength(y=y_mid, sr=sr)
    onset_h = librosa.onset.onset_strength(y=y_high, sr=sr)
    
    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) + 1e-9)
        
    return normalize(onset_l), normalize(onset_m), normalize(onset_h)

def generate_music_game_chart(audio_path, target_notes_count):
    print(f"Loading {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error: {e}")
        return None

    # --- 1. BPM/Offset ---
    print("Detecting BPM...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo[0]) if np.ndim(tempo) > 0 else float(tempo)
    except:
        bpm = 140.0
        
    if bpm < 60: bpm *= 2
    if bpm > 300: bpm /= 2
    bpm = round(bpm)
    print(f"BPM: {bpm}")

    print("Detecting Onset Frames...")
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    offset_sec = onset_times[0] if len(onset_times) > 0 else 0.0

    # --- 2. 解析 ---
    print("Analyzing frequency bands...")
    o_low, o_mid, o_high = get_band_onset_strengths(y, sr)
    times_map = librosa.times_like(o_low, sr=sr)

    seconds_per_beat = 60.0 / bpm
    duration = librosa.get_duration(y=y, sr=sr)
    total_beats_float = (duration - offset_sec) / seconds_per_beat
    total_bars = int(np.ceil(total_beats_float / 4)) + 1
    
    section_map = analyze_audio_structure(y, sr, bpm, offset_sec, total_bars)
    
    # --- 3. グリッド生成 ---
    max_grid = int(total_beats_float * 4)
    all_grids = [i * 0.25 for i in range(max_grid)]
    
    grid_features = []
    grid_scores = []
    
    print("Mapping audio features...")
    for beat in all_grids:
        t = beat * seconds_per_beat + offset_sec
        idx = np.searchsorted(times_map, t)
        if idx >= len(times_map): idx = len(times_map) - 1
        
        w = 2
        s_idx = max(0, idx - w)
        e_idx = min(len(times_map), idx + w + 1)
        
        l_val = np.max(o_low[s_idx:e_idx])
        m_val = np.max(o_mid[s_idx:e_idx])
        h_val = np.max(o_high[s_idx:e_idx])
        env_val = np.max(onset_env[s_idx:e_idx])
        
        bar_idx = int(beat // 4)
        sec = section_map[bar_idx] if bar_idx < len(section_map) else 'verse'
        
        score = env_val
        if sec == 'drop': score *= 1.5
        elif sec == 'break': score *= 0.2
        elif sec == 'buildup': score *= (1.0 + (beat%4)/4.0)
        
        if beat % 1.0 == 0: score *= 1.2
        
        grid_features.append({'beat': beat, 'l': l_val, 'm': m_val, 'h': h_val, 'sec': sec})
        grid_scores.append(score)

    # --- 4. ノーツ選択 ---
    grid_scores = np.array(grid_scores) + 1e-9
    probs = grid_scores / grid_scores.sum()
    
    num_select = min(len(all_grids), target_notes_count)
    try:
        selected_indices = np.random.choice(len(all_grids), size=num_select, replace=False, p=probs)
    except:
        selected_indices = np.random.choice(len(all_grids), size=num_select, replace=False)
        
    selected_indices = sorted(selected_indices)
    print(f"Generating notes for {len(selected_indices)} beats...")

    # --- 5. 配置ロジック (階段抑制版) ---
    notes = []
    max_shift_notes = min(30, target_notes_count // 25)
    shift_count = 0
    
    last_lanes = set()
    all_keys = [1, 2, 3, 4, 5, 6, 7]
    
    flow_state = {'type': 'random', 'dir': 1, 'idx': 3} 
    pattern_duration = 0

    global_max_low = np.max(np.abs(l_val)) if l_val > 0 else 1.0 # 簡易
    # NOTE: ループ内で正規化済み値を使っているのでここではスキップ

    for idx in selected_indices:
        feat = grid_features[idx]
        beat = feat['beat']
        l_str = feat['l']
        m_str = feat['m']
        h_str = feat['h']
        sec = feat['sec']
        
        assigned_lanes = []
        
        # --- A. Kick (Lane 0) ---
        is_kick_heavy = (l_str > 0.6)
        is_bar_head = (beat % 4 == 0)
        use_shift = False
        
        if shift_count < max_shift_notes:
            if is_kick_heavy and (is_bar_head or sec=='drop'):
                if random.random() < 0.7: use_shift = True
            elif is_bar_head and sec != 'break':
                if random.random() < 0.3: use_shift = True
        
        if use_shift:
            assigned_lanes.append(0)
            shift_count += 1
            
        # --- B. 鍵盤 ---
        is_snare_cymbal = (h_str > m_str and h_str > 0.5)
        is_melody = (m_str > h_str)
        
        use_chord = False
        chord_size = 1
        
        if sec == 'drop':
            if is_snare_cymbal: chord_size = 3 if random.random() < 0.2 else 2
            elif is_kick_heavy: chord_size = 2
        elif sec == 'buildup':
            if is_snare_cymbal: chord_size = 2
        
        if chord_size > 1: use_chord = True
            
        if use_chord:
            c_list = PLEASANT_CHORDS_3 if chord_size == 3 else PLEASANT_CHORDS_2
            valid = [c for c in c_list if not set(c).intersection(last_lanes)]
            if not valid: valid = c_list
            keys_to_add = random.choice(valid)
            
            # 同時押し後はパターンをリセット
            flow_state['type'] = 'random'
            pattern_duration = 0
            
        else:
            # --- 単音 (階段抑制ロジック) ---
            next_lane = -1
            
            # パターン更新 (duration切れ または 新しいフレーズ感)
            if pattern_duration <= 0:
                r = random.random()
                
                # デフォルトはランダム（乱打）
                flow_state['type'] = 'random'
                pattern_duration = random.randint(4, 16)

                # メロディが強くても階段にする確率は低くする (0.6 -> 0.25)
                if is_melody:
                    if r < 0.25: 
                        flow_state['type'] = 'stairs'
                        flow_state['dir'] = 1 if random.random() > 0.5 else -1
                        # 階段は短く終わらせる (max 6)
                        pattern_duration = random.randint(3, 6)
                
                # トリルも低確率 (0.15)
                elif is_snare_cymbal:
                    if r < 0.15:
                        flow_state['type'] = 'trill'
                        base = random.randint(1,6)
                        flow_state['trill_pair'] = [base, base+1]
                        pattern_duration = random.randint(4, 8)

            # --- レーン決定 ---
            if flow_state['type'] == 'stairs':
                flow_state['idx'] = (flow_state['idx'] + flow_state['dir'])
                # 折り返し
                if flow_state['idx'] > 6: 
                    flow_state['idx'] = 5
                    flow_state['dir'] = -1
                elif flow_state['idx'] < 0:
                    flow_state['idx'] = 1
                    flow_state['dir'] = 1
                next_lane = all_keys[flow_state['idx']]
                
            elif flow_state['type'] == 'trill':
                pair = flow_state['trill_pair']
                if list(last_lanes) and list(last_lanes)[0] == pair[0]:
                    next_lane = pair[1]
                else:
                    next_lane = pair[0]
            else:
                # --- Improved Random (Wide Spread) ---
                # 直前のレーンと近いところばかり選ぶと「偶発的階段」になるので
                # 離れたレーンを選びやすくする重み付け
                
                weights = []
                last_l = list(last_lanes)[0] if last_lanes else 4
                
                for k in all_keys:
                    if k in last_lanes:
                        weights.append(0) # 縦連禁止
                    else:
                        dist = abs(k - last_l)
                        # 距離が遠いほど重みを大きく (距離1=1, 距離6=6)
                        # ただし極端になりすぎないように +1
                        weights.append(dist + 1)
                
                # 重みに基づいて選択
                total_w = sum(weights)
                if total_w > 0:
                    probs = [w/total_w for w in weights]
                    next_lane = np.random.choice(all_keys, p=probs)
                else:
                    next_lane = random.choice([k for k in all_keys if k not in last_lanes])
                
                flow_state['idx'] = next_lane - 1

            # 念のための縦連防止
            while next_lane in last_lanes:
                next_lane = random.choice(all_keys)
                
            keys_to_add = [next_lane]

        assigned_lanes.extend(keys_to_add)
        
        for l in assigned_lanes:
            beat_val = int(beat) if beat.is_integer() else beat
            notes.append({"lane": int(l), "beat": beat_val})
            
        current_keys = set([l for l in assigned_lanes if l != 0])
        if current_keys:
            last_lanes = current_keys
            pattern_duration -= 1

    notes.sort(key=lambda x: x['beat'])

    chart_data = {
        "title": "Generated Chart (Less Stairs)",
        "artist": "Auto Generator",
        "audio": os.path.basename(audio_path),
        "bpm": int(bpm),
        "timeSignature": [4, 4],
        "bars": total_bars,
        "offsetSec": round(offset_sec, 3),
        "lanes": [
            {"index": 0, "key": "Shift"},
            {"index": 1, "key": "KeyZ"},
            {"index": 2, "key": "KeyS"},
            {"index": 3, "key": "KeyX"},
            {"index": 4, "key": "KeyD"},
            {"index": 5, "key": "KeyC"},
            {"index": 6, "key": "KeyF"},
            {"index": 7, "key": "KeyV"}
        ],
        "notes": notes
    }
    return chart_data

if __name__ == "__main__":
    input_arg = "song"
    target_notes = 800
    if len(sys.argv) > 1: input_arg = sys.argv[1]
    if len(sys.argv) > 2:
        try: target_notes = int(sys.argv[2])
        except: pass

    if input_arg == "target":
        base_name = "target"
        target_wav = "target.wav"
    elif input_arg.endswith(".wav"):
        base_name = input_arg[:-4]
        target_wav = input_arg
    else:
        base_name = input_arg
        target_wav = f"{input_arg}.wav"
        
    output_filename = f"{base_name}.chart.json"
    
    print(f"Target: {target_wav} -> {output_filename}")
    print(f"Target Notes: {target_notes}")
    
    if os.path.exists(target_wav):
        chart_json = generate_music_game_chart(target_wav, target_notes)
        if chart_json:
            try:
                with open(output_filename, "w", encoding='utf-8') as f:
                    json.dump(chart_json, f, indent=2, ensure_ascii=False)
                print(f"Successfully generated chart: {output_filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print("Failed to generate chart data.")
    else:
        print(f"Error: File '{target_wav}' not found.")