import pandas as pd
import numpy as np
import re

# ==============================================================================
# 1. CONFIGURATION: MARKERS
# ==============================================================================
MARKER_START = "experimental ai feature"
MARKER_ELIGIBLE_PRIOR_CHRONIC = "since you're not currently being treated"
MARKER_INELIGIBLE = "limited set of situations right now"

MARKER_S3_START = "can you tell me about your symptoms" 
MARKER_UNHANDLED = "conditions i'm not trained to handle"
MARKER_ER = "call 911" 
MARKER_DDX = "few more questions" 
MARKER_REPORT = "## what to do"

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def parse_chat_to_list(chat_history_str):
    chat_text = str(chat_history_str)
    messages = []
    
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2}[^|]*)\|\|\|\[([^\]]+)\]:')
    matches = list(pattern.finditer(chat_text))
    
    for i, match in enumerate(matches):
        timestamp_str = match.group(1)
        sender = match.group(2).strip().lower()
        
        start_text_idx = match.end()
        end_text_idx = matches[i+1].start() if i + 1 < len(matches) else len(chat_text)
            
        text = chat_text[start_text_idx:end_text_idx].strip()
        
        try:
            current_time = pd.to_datetime(timestamp_str, utc=True)
            messages.append({'time': current_time, 'sender': sender, 'text': text})
        except:
            continue
            
    return messages

# ==============================================================================
# 3. 7-STAGE ANALYSIS LOGIC
# ==============================================================================
def analyze_session_stages(chat_history_str):
    messages = parse_chat_to_list(chat_history_str)
    if not messages:
        return pd.Series(dtype=float)

    def find_idx(marker, start_search=0):
        if start_search is None: return None
        for i in range(start_search, len(messages)):
            if marker in messages[i]['text'].lower():
                return i
        return None

    def calc_stats(start_idx, end_idx):
        if start_idx is None or end_idx is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        if start_idx > end_idx: start_idx = end_idx
            
        sys_msgs, user_msgs = 0, 0
        sys_time, user_time = 0.0, 0.0
        
        if start_idx == end_idx:
            if messages[start_idx]['sender'] == 'patient': user_msgs = 1
            else: sys_msgs = 1
            return user_msgs, sys_msgs, user_msgs + sys_msgs, 0.0, 0.0, 0.0
            
        for i in range(start_idx, end_idx):
            msg = messages[i]
            next_msg = messages[i+1]
            duration = max(0, (next_msg['time'] - msg['time']).total_seconds())
            
            if msg['sender'] == 'patient':
                user_msgs += 1
                sys_time += duration 
            else:
                sys_msgs += 1
                user_time += duration 
                
        if messages[end_idx]['sender'] == 'patient': user_msgs += 1
        else: sys_msgs += 1
            
        overall_msgs = user_msgs + sys_msgs
        overall_time = user_time + sys_time

        return user_msgs, sys_msgs, overall_msgs, user_time, sys_time, overall_time

    # --- LOCATE MARKERS ---
    idx_intro = find_idx(MARKER_START)
    if idx_intro is None: idx_intro = 0 
    
    idx_elig_chron = find_idx(MARKER_ELIGIBLE_PRIOR_CHRONIC, idx_intro)
    idx_ineligible = find_idx(MARKER_INELIGIBLE, idx_intro)
    
    idx_symp_start = find_idx(MARKER_S3_START, idx_intro)
    
    idx_unhandled = find_idx(MARKER_UNHANDLED, idx_symp_start if idx_symp_start else 0)
    idx_er = find_idx(MARKER_ER, idx_symp_start if idx_symp_start else 0)
    idx_ddx = find_idx(MARKER_DDX, idx_symp_start if idx_symp_start else 0)
    idx_report = find_idx(MARKER_REPORT, idx_symp_start if idx_symp_start else 0)

    idx_first_user = None
    for i in range(idx_intro, len(messages)):
        if messages[i]['sender'] == 'patient':
            idx_first_user = i
            break

    # S2 Eval logic
    elig_candidates = [i for i in [idx_elig_chron, idx_ineligible] if i is not None and (idx_first_user is None or i > idx_first_user)]
    idx_elig_eval = min(elig_candidates) if elig_candidates else None

    # S4-S7 Eval logic
    eval_candidates = [i for i in [idx_unhandled, idx_er, idx_ddx, idx_report] if i is not None and (idx_symp_start is None or i > idx_symp_start)]
    idx_eval = min(eval_candidates) if eval_candidates else None

    s1_bounds = s2_bounds = s3_bounds = s4_bounds = s5_bounds = s6_bounds = s7_bounds = (None, None, False)

    # --- 7-STAGE FUNNEL LOGIC ---
    if idx_first_user is None:
        s1_bounds = (idx_intro, len(messages) - 1, False)
    else:
        s1_bounds = (idx_intro, idx_first_user, True)
        
        s2_end_marker = idx_elig_eval if idx_elig_eval is not None else idx_symp_start
        
        if s2_end_marker is None:
            s2_bounds = (idx_first_user, len(messages) - 1, False)
        else:
            s2_bounds = (idx_first_user, s2_end_marker, True) 
            
            if s2_end_marker == idx_ineligible:
                s2_bounds = (idx_first_user, s2_end_marker, False) # Failed/Blocked
            else:
                if idx_symp_start is None:
                    s3_bounds = (s2_end_marker, len(messages) - 1, False)
                else:
                    if idx_eval is None:
                        s3_bounds = (idx_symp_start, len(messages) - 1, False)
                    else:
                        safe_s3_end = max(idx_symp_start, idx_eval - 1)
                        s3_bounds = (idx_symp_start, safe_s3_end, True) 
                        
                        if idx_eval == idx_unhandled:
                            # Terminated due to unhandled. S4 (ER) never starts.
                            s4_bounds = s5_bounds = s6_bounds = s7_bounds = (None, None, False)
                        else:
                            if idx_eval == idx_er:
                                s4_bounds = (safe_s3_end, idx_er, False) # ER detected
                                s5_bounds = s6_bounds = s7_bounds = (None, None, False)
                            else:
                                s4_bounds = (safe_s3_end, idx_eval, True) # ER Check passed
                                if idx_eval == idx_ddx:
                                    if idx_report is None:
                                        s5_bounds = (idx_eval, len(messages) - 1, False)
                                    else:
                                        safe_s5_end = max(idx_eval, idx_report - 1)
                                        s5_bounds = (idx_eval, safe_s5_end, True)
                                        s6_bounds = (safe_s5_end, idx_report, True)
                                        
                                        has_user_followup = any(msg['sender'] == 'patient' for msg in messages[idx_report+1:])
                                        s7_bounds = (idx_report, len(messages) - 1, True) if has_user_followup else (None, None, False)
                                elif idx_eval == idx_report:
                                    # Skipped DDX
                                    s5_bounds = (idx_eval, idx_eval, True) 
                                    s6_bounds = (idx_eval, idx_report, True)
                                    
                                    has_user_followup = any(msg['sender'] == 'patient' for msg in messages[idx_report+1:])
                                    s7_bounds = (idx_report, len(messages) - 1, True) if has_user_followup else (None, None, False)

    stages = [
        ('S1', s1_bounds[0], s1_bounds[1], s1_bounds[2]),
        ('S2', s2_bounds[0], s2_bounds[1], s2_bounds[2]),
        ('S3', s3_bounds[0], s3_bounds[1], s3_bounds[2]),
        ('S4', s4_bounds[0], s4_bounds[1], s4_bounds[2]),
        ('S5', s5_bounds[0], s5_bounds[1], s5_bounds[2]),
        ('S6', s6_bounds[0], s6_bounds[1], s6_bounds[2]),
        ('S7', s7_bounds[0], s7_bounds[1], s7_bounds[2])
    ]
    
    stats = {}
    
    # --- FLAGS ---
    stats['S2_Eligible_Prior_Chronic'] = 1 if (idx_elig_chron is not None and s2_end_marker == idx_elig_chron) else 0
    stats['S2_Ineligible_Blocked'] = 1 if (idx_ineligible is not None and s2_end_marker == idx_ineligible) else 0
    stats['S2_Eligible_No_Chronic'] = 1 if (s2_bounds[2] == True and stats['S2_Eligible_Prior_Chronic'] == 0 and stats['S2_Ineligible_Blocked'] == 0) else 0

    stats['S3_Unhandled_Blocked'] = 1 if (idx_unhandled is not None and idx_eval == idx_unhandled) else 0
    stats['S4_ER_Blocked'] = 1 if (idx_er is not None and idx_eval == idx_er) else 0

    for name, start, end, is_completed in stages:
        stats[f'{name}_Started'] = 1 if start is not None else 0
        stats[f'{name}_Completed'] = 1 if is_completed else 0
        
        u_msgs, s_msgs, o_msgs, u_time, s_time, o_time = calc_stats(start, end)
        
        if name in ['S4', 'S6'] and start is not None:
            u_msgs = 0; s_msgs = 1; o_msgs = 1
            
        if start is not None:
            stats[f'{name}_User_Msgs'] = u_msgs
            stats[f'{name}_PHA_Msgs'] = s_msgs
            stats[f'{name}_Overall_Msgs'] = o_msgs
            stats[f'{name}_User_Time'] = u_time
            stats[f'{name}_PHA_Time'] = s_time
            stats[f'{name}_Overall_Time'] = o_time
        else:
            stats[f'{name}_User_Msgs'] = np.nan
            stats[f'{name}_PHA_Msgs'] = np.nan
            stats[f'{name}_Overall_Msgs'] = np.nan
            stats[f'{name}_User_Time'] = np.nan
            stats[f'{name}_PHA_Time'] = np.nan
            stats[f'{name}_Overall_Time'] = np.nan
            
    return pd.Series(stats)

