import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eispy2d.api import testset_api as ts
from eispy2d.api import benchmark_api as bmk

mytestset = ts.TestSet(name="Teste", wavelength=1.0,
                      image_size=(.8, .8), observation_radius=1., resolution=(60, 60))

# Generate tests
print('Creating tests...')
mytestset.randomize_tests()

# Build benchmark object
mybenchmark = bmk.Benchmark("mybenchmark", testset=mytestset)

# Run benchmark experiment
mybenchmark.run(parallelization=bmk.PARALLELIZE_TESTS)

# Save results
mybenchmark.save(save_testset=True)
