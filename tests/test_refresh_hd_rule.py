from phase_agent.analysis.feedback.model_refresh_plan import build_model_refresh_plan
from phase_agent.tools.workflows.model_refresh_state import _hd_reference
from phase_agent.analysis.phase.phase_quality import snapshot_evidence_digest


def test_high_hd_is_always_selected_using_original_operation():
    rows = [{'structure_id': str(i), 'structure_sha256': str(i), 'source_version':'old',
             'phase':'O3', 'x_Na_per_O2':0.5, 'ehull':0.05, 'hd_required':i == 0}
            for i in range(20)]
    plan = build_model_refresh_plan(rows, old_version='old', new_version='new')
    chosen = next(r for r in plan['candidates'] if r['structure_id'] == '0')
    assert chosen['selection_reason'] == 'high_hd' and chosen['operation'] == 'predict'
    assert len(plan['candidates']) == 3


def test_hd_reference_uses_existing_preview_and_keeps_formal_snapshot(monkeypatch):
    import copy
    entries = [{'record_id':'a', 'x_Na_per_O2':0., 'energy_per_O2':0., 'composition':{'O':2,'Co':1}},
               {'record_id':'bad', 'x_Na_per_O2':0.5, 'energy_per_O2':-2., 'composition':{'O':2,'Co':1,'Na':0.5}},
               {'record_id':'b', 'x_Na_per_O2':1., 'energy_per_O2':0., 'composition':{'O':2,'Co':1,'Na':1}},
               {'record_id':'c', 'x_Na_per_O2':0.5, 'energy_per_O2':0.2, 'composition':{'O':2,'Co':1,'Na':0.5}}]
    snapshot={'version':'v','model_version':'m','entries':entries}
    saved=copy.deepcopy(snapshot)
    report={'hull_version':'v','model_version':'m','snapshot_evidence_digest':snapshot_evidence_digest(snapshot),
            'screening_policy':{'hull_shift_mev_per_O2':50.,'outlier_gap_mev_per_O2':50.},
            'preview_excluded_geometry_keys':['g'],
            'structures':[{'geometry_key':'g','review_id':'r','record_ids':['bad'], 'hull_lift_mev_per_O2':100.}]}
    called=[]
    def preview(s, r, *, review_ids):
        called.append(review_ids)
        return {'excluded_record_ids':['bad']}
    monkeypatch.setattr('phase_agent.analysis.phase.phase_quality_preview.build_quality_preview',preview)
    high, ehull = _hd_reference(snapshot, report)
    assert called == [['r']] and high == {'bad'}
    assert ehull['c'] == 0.2/3.5 and snapshot == saved
