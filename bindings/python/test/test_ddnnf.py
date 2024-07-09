from ddnnife import Ddnnf

ddnnf = Ddnnf.from_file("../../example_input/busybox-1.18.0_c2d.nnf", None)


def test_count():
    count = ddnnf.rc()
    assert count == 2061138519356781760670618805653750167349287991336595876373542198990734653489713239449032049664199494301454199336000050382457451123894821886472278234849758979132037884598159833615564800000000000000000000


def test_core():
    core = ddnnf.get_core()
    assert len(core) == 41


def test_atomic_sets():
    atomic_sets = ddnnf.as_mut().atomic_sets(None, [1], True)
    assert len(atomic_sets[0]) == 854


def test_t_wise():
    sample = ddnnf.sample_t_wise(1)
