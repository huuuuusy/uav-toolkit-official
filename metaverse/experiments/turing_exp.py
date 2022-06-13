def get_exp_info(index):
    if index == 'All': 
        subjects = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5','H1','H2','H3','H4','H5','I1','I2','I4','I5','J2','J3','J4','J5','K1','K2','K3','K4','K5','L1','L2','L3','L4','L5','M1','M2','M3','M4','M5','N1','N2','N3','N4','N5','O1','O2','O3','O4','O5','P1','P2','P3','P4','P5','Q1','Q2','Q3','Q4','Q5','T1','T2']
        description = 'tracking result in all sequences'
    # single subject
    if index == 'A': 
        subjects = ['A1','A2','A3','A4','A5']
        description = 'group A (short-term tracking in 500-1000 frames)'
    if index == 'B': 
        subjects = ['B1','B2','B3','B4','B5']
        description = 'group B (short-term tracking in 1000-2000 frames)'
    if index == 'C': 
        subjects = ['C1','C2','C3','C4','C5']
        description = 'group C (long-term tracking in 1000-2000 frames)'
    if index == 'D': 
        subjects = ['D1','D2','D3','D4','D5']
        description = 'group D (long-term tracking in 5000-10000 frames)'
    if index == 'E': 
        subjects = ['E1','E2','E3','E4','E5']
        description = 'group E (global instance tracking in 1000-2000 frames)'
    if index == 'F': 
        subjects = ['F1','F2','F3','F4','F5']
        description = 'group F (global instance tracking in 5000-10000 frames)'
    if index == 'G': 
        subjects = ['G1','G2','G3','G4','G5']
        description = 'group G (global instance tracking in 15000-30000 frames)'
    if index == 'H': 
        subjects = ['H1','H2','H3','H4','H5']
        description = 'group H (tracking the target with abnormal ratio in 500-1000 frames)'
    if index == 'I': 
        subjects = ['I1','I2','I4','I5']
        description = 'group I (tracking the target with abnormal scale in 500-1000 frames)'
    if index == 'J': 
        subjects = ['J2','J3','J4','J5']
        description = 'group J (tracking the target with abnormal illumination in 500-1000 frames)'
    if index == 'K': 
        subjects = ['K1','K2','K3','K4','K5']
        description = 'group K (tracking the target with blur bounding-box in 500-1000 frames)'
    if index == 'L': 
        subjects = ['L1','L2','L3','L4','L5']
        description = 'group L (tracking the target with drastic ratio variation in 500-1000 frames)'
    if index == 'M': 
        subjects = ['M1','M2','M3','M4','M5']
        description = 'group M (tracking the target with drastic scale variation in 500-1000 frames)'
    if index == 'N': 
        subjects = ['N1','N2','N3','N4','N5']
        description = 'group N (tracking the target with drastic illumination variation in 500-1000 frames)'
    if index == 'O': 
        subjects = ['O1','O2','O3','O4','O5']
        description = 'group O (tracking the target with drastic clarity variation in 500-1000 frames)'
    if index == 'P': 
        subjects = ['P1','P2','P3','P4','P5']
        description = 'group P (tracking the target with drastic fast motion in 500-1000 frames)'
    if index == 'Q': 
        subjects = ['Q1','Q2','Q3','Q4','Q5']
        description = 'group Q (tracking the target with low correlation coefficient in 500-1000 frames)'
    if index == 'T': 
        subjects = ['T1','T2']
        description = 'group T (TEST videos)'
    # task
    if index == 'STT': 
        subjects = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5']
        description = 'short-term tracking task'
    if index == 'LTT': 
        subjects = ['C1','C2','C3','C4','C5','D1','D2','D3','D4','D5']
        description = 'long-term tracking task'
    if index == 'GIT': 
        subjects = ['E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']
        description = 'global instance tracking task'
    # challenge number
    if index == 'Challenge_1': 
        subjects = ['A1','A3','B1','H2','I5','P4']
        description = 'one challenging factor'
    if index == 'Challenge_2': 
        subjects = ['A2','B2','B4','B5','C1','C5','E1','E3','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5','H4','I2','I4','K5','L3','M4','M5','N3','O2','O3','O4']
        description = 'two challenging factors'
    if index == 'Challenge_3': 
        subjects = ['T1','T2','A4','B3','D2','E2','J2','L4','L5','M2','O1','P1','P2','P3','P5','Q1','Q3']
        description = 'three challenging factors'
    if index == 'Challenge_4': 
        subjects = ['A5','C4','D3','E4','H1','H5','J3','J4','K1','K2','K3','K4','L1','L2','N2','N4','N5','O5','Q4','Q5']
        description = 'four challenging factors'
    if index == 'Challenge_5': 
        subjects = ['C3','D1','D5','M1','N1']
        description = 'five challenging factors'
    if index == 'Challenge_6': 
        subjects = ['C2','D4','H3','I1','J5','M3']
        description = 'six challenging factors'
    if index == 'Challenge_7': 
        subjects = ['Q2']
        description = 'seven challenging factors'
    # attribute 
    if index == 'blur_bbox': 
        subjects = ['A1','A2','C1','I2','I4','K5','T1','T2','P3','D3','K1','K2','K3','K4','N2','D1','D5','M1','C2','D4','H3','I1','M3','Q2']
        description = 'tracking the target with blur bounding-box'
    if index == 'illumination': 
        subjects = ['K5','N2','D1','D5','M1','C2','D4','H3','I1','M3','Q2','B2','B4','M4','M5','A4','B3','D2','J2','P5','H5','J3','J4','L2','N5','O5','C3','J5']
        description = 'tracking the target with abnormal illumination'
    if index == 'relative_scale': 
        subjects = ['M1','C2','D4','I1','M3','J2','A2','I2','I4','D3','I5','Q3','E4','Q4']
        description = 'tracking the target with abnormal scale'
    if index == 'ratio': 
        subjects = ['C2','D4','I1','M3','D3','Q4','H3','Q2','B2','B4','A4','B3','H5','J3','J4','L2','C3','J5','T2','K2','K3','K4','H2','C5','H4','L3','E2','L4','L5','Q1','A5','C4','H1','L1','Q5','N1']
        description = 'tracking the target with abnormal ratio'
    if index == 'delta_blur_bbox': 
        subjects = ['D4','A5','M5','O5','B1','B5','O2','O3','O4','O1','P1','P2']
        description = 'tracking the target with drastic clarity variation'
    if index == 'delta_illumination': 
        subjects = ['B5','P1','C2','I1','M3','Q4','H3','Q2','C3','J5','C4','N1','M1','N2','D1','N5','T1','N3','N4']
        description = 'tracking the target with drastic illumination variation'
    if index == 'delta_relative_scale': 
        subjects = ['I1','M3','H3','N1','M1','A5','H5','J3','J4','L2','K2','K3','K4','L4','L5','Q1','H1','L1','Q5','Q3','M2']
        description = 'tracking the target with drastic scale variation'
    if index == 'delta_ratio': 
        subjects = ['H3','N1','A5','H5','J3','J4','L2','K2','K3','K4','L4','L5','H1','L1','Q5','M2','Q2','J5','N4','O5','O1','P2','A4','B3','H4','L3','P5','K1']
        description = 'tracking the target with drastic ratio variation'
    if index == 'fast_motion': 
        subjects = ['N1','H1','L1','M2','Q2','J5','N4','O5','O1','P2','P5','K1','P1','C3','N2','N5','N3','O2','O3','O4','J2','E4','D5','M4','D2','P3','A3','P4']
        description = 'tracking the target with fast motion'
    if index == 'corrcoef': 
        subjects = ['Q2','J5','N4','K1','N5','D5','P3','Q5','Q1','Q3','Q4','C4','D1','T1']
        description = 'tracking the target with low correlation coefficient'
    if index == 'absent': 
        subjects = ['C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5','T2']
        description = 'tracking the target with disappearance'

    return subjects, description