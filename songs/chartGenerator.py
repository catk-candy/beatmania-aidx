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
    """ librosaを使って音量・周波数帯域からEDM特有のセクションを判定 """
    print("Analyzing EDM structure using librosa...")
    
    # STFTを用いてスペクトログラムを計算
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Low帯域(キックやベース)のみのマスクを作成 (< 250 Hz)
    low_mask = freqs < 250
    S_low = S[low_mask, :]
    
    # librosaで全体のRMS計算し、Low帯域は手動で計算 (ParameterError回避)
    rms_full = librosa.feature.rms(S=S)[0]
    rms_low = np.sqrt(np.mean(S_low**2, axis=0)) if S_low.shape[0] > 0 else np.zeros_like(rms_full)
    
    times = librosa.times_like(rms_full, sr=sr)
    seconds_per_bar = (60.0 / bpm) * 4
    
    bar_stats = []
    for i in range(total_bars):
        start_t = offset_sec + i * seconds_per_bar
        end_t = start_t + seconds_per_bar
        
        idx = np.where((times >= start_t) & (times < end_t))[0]
        if len(idx) == 0:
            bar_stats.append({'full': 0, 'low': 0})
            continue
            
        bar_rms_full = np.mean(rms_full[idx])
        bar_rms_low = np.mean(rms_low[idx])
        bar_stats.append({'full': bar_rms_full, 'low': bar_rms_low})

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

def generate_music_game_chart(audio_path, target_notes_count, analysis_file=None, level=1):
    print(f"Loading {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error: {e}")
        return None

    # --- 1. メロディとパーカッションのリズム推定 (Harmonic & Percussive components) ---
    print("Estimating melody and percussive rhythm...")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    melody_onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr)
    perc_onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

    if np.max(melody_onset_env) > 0:
        melody_onset_env = melody_onset_env / np.max(melody_onset_env)
    if np.max(perc_onset_env) > 0:
        perc_onset_env = perc_onset_env / np.max(perc_onset_env)

    # --- 1.5. BPM/Offset ---
    print("Detecting BPM...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    bpm = None
    offset_sec = None
    
    if analysis_file and os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                if 'bpm' in analysis_data:
                    bpm = float(analysis_data['bpm'])
                    print(f"Loaded BPM from analysis file: {bpm}")
                if 'offsetSec' in analysis_data:
                    offset_sec = float(analysis_data['offsetSec'])
                    print(f"Loaded Offset from analysis file: {offset_sec}")
        except Exception as e:
            print(f"Failed to read analysis file: {e}")

    if bpm is None:
        try:
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            bpm = float(tempo[0]) if np.ndim(tempo) > 0 else float(tempo)
        except:
            bpm = 140.0
            
        if bpm < 60: bpm *= 2
        if bpm > 300: bpm /= 2
        bpm = round(bpm)
        print(f"Auto-detected BPM: {bpm}")

    if offset_sec is None:
        print("Detecting Onset Frames...")
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        offset_sec = onset_times[0] if len(onset_times) > 0 else 0.0
        print(f"Auto-detected Offset: {offset_sec}")

    # --- 2. 解析 ---
    print("Analyzing frequency bands...")
    o_low, o_mid, o_high = get_band_onset_strengths(y, sr)
    times_map = librosa.times_like(o_low, sr=sr)

    seconds_per_beat = 60.0 / bpm
    duration = librosa.get_duration(y=y, sr=sr)
    total_beats_float = (duration - offset_sec) / seconds_per_beat
    total_bars = int(np.ceil(total_beats_float / 4)) + 1
    
    section_map = analyze_audio_structure(y, sr, bpm, offset_sec, total_bars)
    
    # 構成の切り替わり地点 (Structural boundaries) を記録
    # 前の小節とセクションが変わったタイミングのビートを記録します
    structural_boundaries = set()
    for bar_idx in range(1, len(section_map)):
        if section_map[bar_idx] != section_map[bar_idx-1]:
            # 小節の頭のビートを切り替わり地点とする (bar_idx * 4)
            structural_boundaries.add(bar_idx * 4.0)

    # --- 3. グリッド生成 (解析された生のリズムを使用) ---
    print("Extracting raw onsets from melody and percussion...")
    melody_frames = librosa.onset.onset_detect(onset_envelope=melody_onset_env, sr=sr, backtrack=True)
    perc_frames = librosa.onset.onset_detect(onset_envelope=perc_onset_env, sr=sr, backtrack=True)
    env_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    
    all_onset_frames = np.concatenate([melody_frames, perc_frames, env_frames])
    all_onset_frames = np.unique(all_onset_frames)
    all_onset_times = librosa.frames_to_time(all_onset_frames, sr=sr)
    all_onset_times.sort()
    
    # 重複や近すぎるオンセット（50ms未満）を除外
    # --- クオンタイズ処理 (16分音符スナップ) ---
    filtered_grids = {}
    for t in all_onset_times:
        if t < offset_sec:
            continue
            
        beat_float = (t - offset_sec) / seconds_per_beat
        # 16分音符 (1/4拍) 単位で丸める
        quantized_beat = round(beat_float * 4.0) / 4.0
        
        # すでに同じビートにスナップされたオンセットがあればスキップ
        if quantized_beat not in filtered_grids:
            quantized_t = offset_sec + (quantized_beat * seconds_per_beat)
            filtered_grids[quantized_beat] = quantized_t
            
    # 時間をbeatと秒(t)の両方で保持するように変更
    all_grids = [{'beat': beat, 't': t} for beat, t in sorted(filtered_grids.items())]
    
    grid_features = []
    grid_scores = []
    
    print("Mapping audio features...")
    for grid_info in all_grids:
        beat = grid_info['beat']
        t = grid_info['t']
        
        idx = np.searchsorted(times_map, t)
        if idx >= len(times_map): idx = len(times_map) - 1
        
        w = 2
        s_idx = max(0, idx - w)
        e_idx = min(len(times_map), idx + w + 1)
        
        l_val = np.max(o_low[s_idx:e_idx])
        m_val = np.max(o_mid[s_idx:e_idx])
        h_val = np.max(o_high[s_idx:e_idx])
        env_val = np.max(onset_env[s_idx:e_idx])
        melody_val = np.max(melody_onset_env[s_idx:e_idx])
        perc_val = np.max(perc_onset_env[s_idx:e_idx])
        
        bar_idx = int(beat // 4)
        sec = section_map[bar_idx] if bar_idx < len(section_map) else 'verse'
        
        # 曲が静か（全体オンセットが弱い）で、メロディかパーカッションが際立っている場合は、そのリズムを特別に強くする
        is_quiet = env_val < 0.3
        if is_quiet and (melody_val > 0.4 or perc_val > 0.4):
            # メロディかパーカッション、強い方のリズムに合わせる
            dominant_val = max(melody_val, perc_val)
            score = dominant_val * 4.0
        else:
            # メロディのリズムとパーカッシブなリズムの両方をベースにスコアを算出
            score = melody_val * 1.5 + perc_val * 1.5 + env_val * 0.5
            
        progress = (beat / total_beats_float) if total_beats_float > 0 else 0
        
        # 低難易度（Lv 1-5）の場合の特別処理
        is_low_level = (level <= 5)
            
        if sec == 'drop': 
            score *= 4.0 if is_low_level else 8.0    # 盛り上がる箇所 (低難易度は倍率を抑える)
        elif sec == 'outro': score *= 0.05           # アウトロはノーツを極限まで減らす
        elif sec == 'break': score *= 0.1            # 落ち着いた箇所もノーツをかなり減らす
        elif sec == 'intro': score *= 0.3            # イントロは少なめ
        elif sec == 'buildup': score *= (2.0 + (beat%4)/2.0) # ビルドアップは徐々に増やす
        else: score *= 1.0                           # verse等は通常
        
        # 拍の頭に近い場合は少しスコアを上げる
        if abs((beat % 1.0) - round(beat % 1.0)) < 0.1: score *= 1.2
        
        # [NEW] アンチスパイク (ラス殺し防止) ロジック：Lv 1-5 のみ適用
        # 曲の後半 (70%以降) で徐々にノーツ採用スコアにペナルティをかける
        if is_low_level and progress > 0.7:
            # 0.7 〜 1.0 の間で、ペナルティ倍率を 1.0 から 0.4 に下げる (最大 60% 減)
            penalty_multiplier = max(0.4, 1.0 - (progress - 0.7) * 1.5)
            score *= penalty_multiplier
        
        grid_features.append({'beat': beat, 't': t, 'l': l_val, 'm': m_val, 'h': h_val, 'melody': melody_val, 'perc': perc_val, 'sec': sec})
        grid_scores.append(score)

    # --- 4. ノーツ選択 (TOP-N選択へ変更) ---
    grid_scores = np.array(grid_scores)
    
    num_select = min(len(all_grids), target_notes_count)
    
    # スコアが高いものから順番に TOP-N 個を取得
    top_indices = np.argsort(grid_scores)[::-1][:num_select]
    
    selected_indices = sorted(top_indices)
    print(f"Generating notes for {len(selected_indices)} beats...")

    # --- 5. 配置ロジック (階段抑制版) ---
    notes = []
    max_shift_notes = min(30, target_notes_count // 25)
    shift_count = 0
    
    last_lanes = set()
    if level <= 3:
        all_keys = [1, 3, 5, 7] # 橙レーン(2,4,6)を一切出現させない
    else:
        all_keys = [1, 2, 3, 4, 5, 6, 7]
    
    flow_state = {'type': 'random', 'dir': 1, 'idx': len(all_keys)//2} 
    pattern_duration = 0

    global_max_low = np.max(np.abs(l_val)) if l_val > 0 else 1.0 # 簡易
    # NOTE: ループ内で正規化済み値を使っているのでここではスキップ

    for idx in selected_indices:
        feat = grid_features[idx]
        beat = feat['beat']
        t_sec = feat['t']
        l_str = feat['l']
        m_str = feat['m']
        h_str = feat['h']
        sec = feat['sec']
        
        assigned_lanes = []
        
        # 楽曲構成の切り替わり地点か判定（誤差0.1beat内）
        is_structure_boundary = False
        for bnd_beat in structural_boundaries:
            if abs(beat - bnd_beat) < 0.1:
                is_structure_boundary = True
                break

        # --- A. Kick (Lane 0) ---
        is_kick_heavy = (l_str > 0.6)
        use_shift = False
        
        if is_structure_boundary:
            # 切り替わり地点では高い確率でShift(Lane 0)ノーツを降らせる
            if random.random() < 0.9: 
                use_shift = True
        elif shift_count < max_shift_notes:
            if is_kick_heavy and sec=='drop':
                if random.random() < 0.7: use_shift = True
            elif abs((beat % 4.0) - round(beat % 4.0)) < 0.1 and sec != 'break':
                if random.random() < 0.3: use_shift = True
        
        if use_shift:
            assigned_lanes.append(0)
            shift_count += 1
            
        # --- B. 鍵盤 ---
        is_snare_cymbal = (h_str > m_str and h_str > 0.5)
        melody_str = feat.get('melody', 0.0)
        perc_str = feat.get('perc', 0.0)
        
        # メロディ成分とパーカッシブ成分の比較
        is_melody = (melody_str > perc_str or m_str > h_str)
        
        # 同時押しを廃止し単音のみの評価とする
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
                    idx1 = random.randint(0, len(all_keys)-2)
                    flow_state['trill_pair'] = [all_keys[idx1], all_keys[idx1+1]]
                    pattern_duration = random.randint(4, 8)

        # --- レーン決定 ---
        if flow_state['type'] == 'stairs':
            flow_state['idx'] = (flow_state['idx'] + flow_state['dir'])
            # 折り返し
            max_idx = len(all_keys) - 1
            if flow_state['idx'] > max_idx - 1: 
                flow_state['idx'] = max_idx - 1
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
            
            flow_state['idx'] = all_keys.index(next_lane)

        # 念のための縦連防止
        while next_lane in last_lanes:
            next_lane = random.choice(all_keys)
            
        keys_to_add = [next_lane]

        assigned_lanes.extend(keys_to_add)
        
        for l in assigned_lanes:
            ms_val = int(round(t_sec * 1000))
            notes.append({"lane": int(l), "ms": ms_val})
            
        current_keys = set([l for l in assigned_lanes if l != 0])
        if current_keys:
            last_lanes = current_keys
            pattern_duration -= 1

    notes.sort(key=lambda x: x['ms'])

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
    difficulty_arg = "01"
    
    if len(sys.argv) > 1: input_arg = sys.argv[1]
    if len(sys.argv) > 2: difficulty_arg = sys.argv[2]

    try:
        level = int(difficulty_arg)
        level = max(1, min(12, level))
    except ValueError:
        level = 1
        
    target_notes = level * 100
    chart_type = f"{level:02d}"

    if input_arg == "target":
        base_name = "target"
        target_wav = "target.wav"
        target_analysis = "target.analysis.json"
    elif input_arg.endswith(".wav"):
        base_name = input_arg[:-4]
        target_wav = input_arg
        target_analysis = f"{base_name}.analysis.json"
    else:
        base_name = input_arg
        target_wav = f"{input_arg}.wav"
        target_analysis = f"{input_arg}.analysis.json"
        
    output_filename = f"{base_name}_{chart_type}.chart.json"
    
    print(f"Target: {target_wav} -> {output_filename}")
    print(f"Difficulty: {level} (Target Notes: {target_notes})")
    
    if os.path.exists(target_wav):
        analysis_file = target_analysis if os.path.exists(target_analysis) else None
        chart_json = generate_music_game_chart(target_wav, target_notes, analysis_file, level)
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