import sys,json,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from experiment import _prepare_action
from scenarios import normalize_task_config
torch.set_num_threads(2)
c=json.loads(Path('src/config.json').read_text())['main'];c.update(TASK='action',P=127,ORDER=3,RULES=[[13,7,14],[39,108,27]],D_MODEL=32,N_HEAD=4,N_LAYER=2,NUM_TRAIN_SAMPLES=8,NUM_TEST_SAMPLES=8,TRAIN_LEN=4,OOD_LEN=4,MISSING_PROB=0)
def build(std):
 torch.manual_seed(17996);random.seed(17996);cfg=dict(c,ACTION_RULE_TOKEN_INIT_STD=std)
 m=_prepare_action({'main':cfg},normalize_task_config('action',cfg))['model'];return m,torch.get_rng_state()
a,ra=build(1);b,rb=build(.02)
assert torch.equal(ra,rb)
for k,v in a.state_dict().items():
 w=b.state_dict()[k]
 if k in ['transformer.wte.weight','lm_head.weight']:
  assert torch.equal(v[:128],w[:128]);assert torch.equal(v[128:]*.02,w[128:])
 else:assert torch.equal(v,w),k
assert b.lm_head.weight is b.transformer.wte.weight
print('PASS: only rule rows scaled; all other weights and RNG identical; weight tying preserved')
