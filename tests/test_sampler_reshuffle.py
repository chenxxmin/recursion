"""Epoch reshuffling covers all samples and survives checkpoint resume exactly."""
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from datasets import BucketBatchSampler
from experiment import run_experiment


def data():
    return [torch.zeros(n,dtype=torch.long) for n in [6]*41+[9]*29]


def test_coverage_and_regrouping():
    samples=data()
    random.seed(17)
    fixed=BucketBatchSampler(samples,7,shuffle=True)
    after_fixed=random.getstate()
    random.seed(17)
    shuffled=BucketBatchSampler(samples,7,shuffle=True,reshuffle_each_epoch=True)
    assert fixed.batches==shuffled.batches and after_fixed==random.getstate()
    assert list(fixed)==list(fixed)
    first,second=list(shuffled),list(shuffled)
    for batches in [first,second]:
        assert sorted(i for batch in batches for i in batch)==list(range(len(samples)))
        assert len(batches)==len(shuffled)
        for batch in batches:
            assert 1<=len(batch)<=7
            assert len({len(samples[i]) for i in batch})==1
    assert first!=second
    assert {tuple(sorted(x)) for x in first}!={tuple(sorted(x)) for x in second}
    saved=random.getstate()
    expected=list(shuffled)
    restored=BucketBatchSampler(samples,7,shuffle=True,reshuffle_each_epoch=True)
    random.setstate(saved)
    assert list(restored)==expected
    evaluation=BucketBatchSampler(samples,7,shuffle=False,reshuffle_each_epoch=True)
    saved=random.getstate()
    assert list(evaluation)==list(evaluation)
    assert random.getstate()==saved


def run(directory,**overrides):
    main=dict(TASK='addition',P=7,A=1,B=1,D_MODEL=16,N_HEAD=1,N_LAYER=1,
              BATCH_SIZE=7,EPOCHS=3,LR=0.001,RANDOM_SEED=42,TRAIN_LEN=8,
              OOD_LEN=10,DROPOUT=0.,MAX_UNIQUE_RATIO=.7,WEIGHT_DECAY=.1,
              ENTROPY_PENALTY_WEIGHT=0.,FIRST_TASK_WEIGHT=1.,EVAL_INTERVAL=1,
              EARLY_STOP_ACCURACY=2.,EARLY_STOP_NO_IMPROVE=3000,
              USE_LEARNABLE_PE=False,MLP_RATIO=2,NUM_MASK=None,USE_AMP=False,
              RESHUFFLE_EACH_EPOCH=True,SKIP_FINAL_GENERATION_TEST=True,
              SAVE_PATH=str(directory/'model.pth'))
    main.update(overrides)
    path=directory/'config.json'
    path.write_text(json.dumps({'main':main,'_BATCH_RUN_MERGED':True}))
    code=0
    with contextlib.redirect_stdout(io.StringIO()):
        try: run_experiment(str(path))
        except SystemExit as exc: code=exc.code
    return code


def test_full_resume_matches_uninterrupted_training():
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        whole,split=folder/'whole',folder/'split'
        whole.mkdir(); split.mkdir()
        assert run(whole)==0
        assert run(split,MAX_TRAIN_HOURS=0)==42
        resume=split/'model_resume.pth'
        assert run(split,RESUME_FROM=str(resume))==0
        a=torch.load(whole/'model.pth',map_location='cpu',weights_only=False)
        b=torch.load(split/'model.pth',map_location='cpu',weights_only=False)
        assert a['final_epoch']==b['final_epoch']==2
        for key in a['model_state_dict']:
            assert torch.equal(a['model_state_dict'][key],b['model_state_dict'][key]),key
        assert a['config']['reshuffle_each_epoch'] is True


if __name__=='__main__':
    test_coverage_and_regrouping()
    test_full_resume_matches_uninterrupted_training()
    print('ALL TESTS PASSED: test_sampler_reshuffle.py')
