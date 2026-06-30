import pandas as pd
import numpy as np


def parse_session_metrics(chat_history_str):
    chat_text = str(chat_history_str)
    pha_time_sec = 0.0
    user_time_sec = 0.0
    patient_messages = 0
    prev_time = None
    
    pha_durations = []
    user_durations = []
    
    lines = chat_text.split('\n')
    for line in lines:
        if '|||[' not in line: continue
        parts = line.split('|||[', 1)
        timestamp_str = parts[0]
        sender = parts[1].split(']:', 1)[0].strip().lower()
        
        try:
            current_time = pd.to_datetime(timestamp_str, utc=True)
        except:
            continue
            
        if prev_time is not None:
            duration = max(0, (current_time - prev_time).total_seconds())
            if sender == 'patient': 
                user_time_sec += duration
                user_durations.append(duration) 
            else: 
                pha_time_sec += duration
                pha_durations.append(duration) 
                
        if sender == 'patient': patient_messages += 1
        prev_time = current_time

    pha_avg_per_turn = np.mean(pha_durations) if len(pha_durations) > 0 else 0.0
    pha_med_per_turn = np.median(pha_durations) if len(pha_durations) > 0 else 0.0
    
    user_avg_per_turn = np.mean(user_durations) if len(user_durations) > 0 else 0.0
    user_med_per_turn = np.median(user_durations) if len(user_durations) > 0 else 0.0

    return pd.Series({
        'turns': patient_messages, 
        'pha_duration_sec': pha_time_sec, 
        'user_duration_sec': user_time_sec,
        'pha_avg_per_turn_sec': pha_avg_per_turn,
        'pha_med_per_turn_sec': pha_med_per_turn,
        'user_avg_per_turn_sec': user_avg_per_turn,
        'user_med_per_turn_sec': user_med_per_turn
    })

def get_patient_text_only(chat_text):
    parts = str(chat_text).split('|||[')
    patient_texts = []
    for p in parts:
        if p.lower().startswith('patient]:'):
            msg = p[9:].strip()
            if msg.lower() != 'hi violet.':
                patient_texts.append(msg)
    return " ".join(patient_texts).lower()

# --- 2. CATEGORIZATIONS ---
def categorize_interaction(row):
    user_text = get_patient_text_only(row['chat_history'])
    if not user_text or user_text.strip() == "": 
        return "Abandoned (No User Response)"
    
    test_keywords = ['test', 'testing', '123', 'asdf']
    if user_text in test_keywords:
        return "Greeting / Testing Only"
        
    off_topic = ['ignore previous', 'system prompt', 'bypass', 'jailbreak', 'joke', 'poem', 'recipe', 'python', 'code']
    if any(word in user_text for word in off_topic): 
        return "Jailbreak / Off-Topic"
    
    return "Valid Symptom Check"

def get_final_output_type(chat_text):
    chat_text = str(chat_text).lower()
    is_regular = "## what to do" in chat_text
    
    # Catches the ineligible sessions
    if "i can only offer guidance for a limited set of situations right now" in chat_text: 
        return "Terminated: Ineligible"
        
    if "conditions i'm not trained to handle" in chat_text: return "Terminated: Unhandled Symptoms"
    if "don't show any healthcare provider visits" in chat_text: return "Terminated: Ineligible (No EHR)"
        
    if not is_regular:
        if "urgent or life-threatening" in chat_text: return "Output: Early Cutoff - ER"
        if "within the next 2 days" in chat_text: return "Output: Early Cutoff - Urgent Referral"
        return "Terminated: Abandoned Mid-Conversation"
    else:
        if "urgent or life-threatening" in chat_text: return "Output: Regular Complete - ER"
        if "make an appointment to see your healthcare provider" in chat_text: return "Output: Regular Complete - PCP Follow-up"
        if "if your symptoms don't improve" in chat_text: return "Output: Regular Complete - Self Care"
        return "Output: Regular Complete - Unknown"

def get_major_group(outcome):
    if "Regular Complete" in outcome: return "Regular Completed Sessions"
    elif "Early Cutoff" in outcome: return "Early Cutoff Sessions"
    elif "Abandoned Mid" in outcome: return "Terminated: Abandoned Mid-Conversation"
    elif "Unhandled" in outcome: return "Terminated: Unhandled Symptoms"
    elif "Ineligible" in outcome: return "Terminated: Ineligible" 
    else: return "Other Terminations"

# --- 3. POST-OUTPUT ENGAGEMENT ---
def get_post_output_engagement(chat_text):
    chat_text = str(chat_text)
    if "## What to do" not in chat_text: return 0 
    return chat_text[chat_text.rfind("## What to do"):].lower().count('|||[patient]:')

# --- 4. EXPLICIT LATENCY FRUSTRATION ---
def extract_latency_frustration(chat_text):
    user_text = get_patient_text_only(chat_text)
    
    latency_words = [
        'taking too long', 'taking forever', 'hello?', 
        'are you there', 'is this working', 'is the app working', 'you are frozen'
    ]
    
    has_frustration = any(word in user_text for word in latency_words)
    frustrated_quote = None
    if has_frustration:
        parts = str(chat_text).split('|||[')
        for p in parts:
            if p.lower().startswith('patient]:') and any(word in p.lower() for word in latency_words):
                frustrated_quote = p[9:].strip()
                break
                
    return pd.Series({'latency_frustration_flag': has_frustration, 'frustrated_quote': frustrated_quote})

# --- 5. MASTER PROCESSING FUNCTION ---
def process_triage_data(df):
    df_clean = df.copy()
    df_clean['interaction_type'] = df_clean.apply(categorize_interaction, axis=1)
    df_clean['final_outcome'] = df_clean['chat_history'].apply(get_final_output_type)
    df_clean['major_group'] = df_clean['final_outcome'].apply(get_major_group)
    df_clean['post_output_user_messages'] = df_clean['chat_history'].apply(get_post_output_engagement)
    
    frustration_df = df_clean['chat_history'].apply(extract_latency_frustration)
    metrics_df = df_clean['chat_history'].apply(parse_session_metrics)
    
    return pd.concat([df_clean, frustration_df, metrics_df], axis=1)
