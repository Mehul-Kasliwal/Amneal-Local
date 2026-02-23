import json
import re

FREQ_REGEX = re.compile(
    r'(?:[Ee]very|[Aa]t\s+least|every\s+time|once\s+in(?:\s+every)?)\s*'
    r'(\d+)\s*(hr|hrs|hour|hours|min|mins|minutes?)'
    r'\s*(?:±|\\pm|\+\/\-|\+\/-|\+-)\s*'
    r'(\d+)\s*(min|mins|minutes?)'
)

with open('all_result_76_20feb.json') as f:
    data = json.load(f)

fmj = data['steps']['filled_master_json']
for p in fmj:
    if p.get('page') == 86:
        rules = []
        for c in p.get('page_content', []):
            if c.get('type') == 'kv_text_block':
                kv = c.get('extracted_kv_text_block', {})
                if isinstance(kv, dict):
                    rules.extend(kv.get('rules_or_instructions', []))
                elif isinstance(kv, list):
                    for item in kv:
                        if isinstance(item, dict):
                            rules.extend(item.get('rules_or_instructions', []))
        
        markdown = ' '.join(rules).lower()
        print('Markdown length:', len(markdown))
        # print first match
        for m in FREQ_REGEX.finditer(markdown):
            start = max(0, m.start() - 200)
            end = min(len(markdown), m.end() + 200)
            context = markdown[start:end].lower()
            print('MATCH:', m.group(0))
            
            qa_idx = -1
            sm_idx = -1
            for pat in ['(qa)', ' qa ', '-qa-', 'for qa']:
                idx = context.find(pat)
                if idx != -1: qa_idx = idx if qa_idx == -1 else min(qa_idx, idx)
            for pat in ['(sm)', ' sm ', '-sm-', 'for sm', 'by (sm)']:
                idx = context.find(pat)
                if idx != -1: sm_idx = idx if sm_idx == -1 else min(sm_idx, idx)
            print(f'  QA idx: {qa_idx}, SM idx: {sm_idx}')
            
            match_pos = m.start() - start
            if qa_idx != -1 and sm_idx != -1:
                d_qa = abs(qa_idx - match_pos)
                d_sm = abs(sm_idx - match_pos)
                role = 'QA' if d_qa < d_sm else 'SM'
            elif qa_idx != -1: role = 'QA'
            elif sm_idx != -1: role = 'SM'
            else: role = 'GENERIC'
            print(f'  Assigned role: {role}')
