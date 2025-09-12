from ddnnife import Ddnnf

ddnnf = Ddnnf.from_file("../../example_input/busybox-1.18.0_c2d.nnf", None)
features = 854

def test_sat():
    ddnnf_mut = ddnnf.as_mut()
    assert ddnnf_mut.is_sat([])
    assert ddnnf_mut.is_sat([-2])
    assert not ddnnf_mut.is_sat([1])


def test_count():
    count = ddnnf.rc()
    expected = 2061138519356781760670618805653750167349287991336595876373542198990734653489713239449032049664199494301454199336000050382457451123894821886472278234849758979132037884598159833615564800000000000000000000
    assert count == expected

    ddnnf_mut = ddnnf.as_mut()
    assert ddnnf_mut.count([]) == expected
    assert ddnnf_mut.count([1]) == 0


def test_core_and_dead():
    both = ddnnf.get_core()
    assert len(both) == 41

    ddnnf_mut = ddnnf.as_mut()
    core = ddnnf_mut.core([])
    assert len(core) == 23

    core = ddnnf_mut.dead([])
    assert len(core) == 18

def test_core():
    core = ddnnf.get_core()
    assert len(core) == 41


def test_enumerate():
    configs = ddnnf.as_mut().enumerate([], 1)
    assert len(configs) == 1
    assert len(configs[0]) == features


def test_random():
    configs = ddnnf.as_mut().random([], 2, 42)
    assert len(configs) == 2
    assert len(configs[0]) == features


def test_atomic_sets():
    atomic_sets = ddnnf.as_mut().atomic_sets(None, [1], True)
    assert len(atomic_sets[0]) == features


def test_t_wise():
    sample = ddnnf.sample_t_wise(1)
    assert sample.is_RESULT_WITH_SAMPLE()
    assert len(sample[0].vars) == features


def test_cnf():
    cnf = ddnnf.to_cnf()
    serialized = cnf.serialize()
    assert cnf.num_variables() == 2830
    assert len(cnf.clauses()) == 7481
    assert len(serialized) == 110642
