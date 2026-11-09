from ase.calculators.emt import EMT
from pymatgen.core import Structure,Lattice
from phase_agent.tools.remote.predict_dft_final_frame import batch_predictor


def test_predictor_loads_one_calculator_and_reuses_it():
    calls=[]
    def factory(options,*,model_path):
        calls.append((options,model_path))
        return EMT()
    predict=batch_predictor({'model_path':'unused','mace_head':'omat_pbe'},calculator_factory=factory)
    a=Structure(Lattice.cubic(4.0),['Cu'],[[0,0,0]])
    b=Structure(Lattice.cubic(4.1),['Cu'],[[0,0,0]])
    results=[predict(a),predict(b)]
    assert len(calls)==1
    assert all(r['status']=='completed' and len(r['forces'])==1 for r in results)
    assert results[0]['energy']!=results[1]['energy']
