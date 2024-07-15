import json

caption_file_path = f"./conversation_bddx_eval.json"

with open(caption_file_path,"r") as fs:
    sig_data = json.load(fs)

gt_signal = {}
for item in sig_data:
    idx, signal = item['id'], item['conversations'][-1]['value']
    # print(signal.split(": "))
    speed, course = float(signal.split(": ")[1].split(" ")[0]),  float(signal.split(": ")[-1])
    gt_signal.update({idx:{"speed":speed,"course":course}})
    # break
    # print(gt_signal)
    # break
    
with open("./BDDX_Test_control_signal_v2.json","w") as ff:
    json.dump(gt_signal,ff,indent=4)