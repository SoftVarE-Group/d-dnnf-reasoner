use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
use ddnnife::Ddnnf;
use ddnnife::ddnnf::anomalies::t_wise_sampling::Sample;
use std::path::Path;

static BENCHMARKS: [(&str, &str, usize); 4] = [
    ("axTLS_d4.nnf", "axTLS (d4)", 2),
    ("busybox_c2d.nnf", "BusyBox (c2d)", 2),
    ("VP9_d4.nnf", "VP9 (d4)", 2),
    ("X264_c2d.nnf", "X264 (c2d)", 2),
];

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("t-wise");
    group.sampling_mode(SamplingMode::Flat);
    group.significance_level(0.01);
    group.noise_threshold(0.05);

    let data_dir = Path::new("tests/data");

    for (path, name, t) in BENCHMARKS {
        let ddnnf = Ddnnf::from_file(&data_dir.join(path), None);
        group.bench_with_input(BenchmarkId::new(name, t), &t, |bencher, t| {
            bencher.iter(|| ddnnf.sample_t_wise(*t, &Sample::default(), None))
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
