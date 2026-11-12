import requests
import json

repo = 'MesserMMP/coronary-angiography-syntax'
r_s = requests.get(f'https://huggingface.co/datasets/{repo}/raw/main/test_data/siemens_20.json').json()
print("Siemens cases:")
for i, s in enumerate(r_s):
    print(f"[{i}] UID: {s['study_uid']} | mean_syntax: {s.get('mean_syntax')} | num_videos: {len(s['videos'])}")

r_p = requests.get(f'https://huggingface.co/datasets/{repo}/raw/main/test_data/philips_97.json').json()
print("\nPhilips cases (sample 0-10):")
for i, s in enumerate(r_p[:10]):
    print(f"[{i}] UID: {s['study_uid']} | mean_syntax: {s.get('mean_syntax')} | num_videos: {len(s['videos'])}")
