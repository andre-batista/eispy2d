

def make_testset(name, params, repetitions=N_REPETITIONS):
    samples = []
    configuration = None

    for repetition in range(repetitions):
        config, inputdata, _, _ = api.prepare_problem(
            params=params,
            forward_kwargs=FORWARD_KWARGS,
        )

        if configuration is None:
            configuration = config

        inputdata.name = f"{name}_{repetition:03d}"
        samples.append(inputdata)

    testset = tst.TestSet(
        name=name,
        configuration=configuration,
        contrast=params["contrast"],
        object_size=params["object_size"],
        resolution=params["resolution"],
        density=None,
        map_pattern=exp.REGULAR_POLYGONS_PATTERN,
        sample_size=repetitions,
        noise=params["noise_level"],
        indicators=[
            rst.REL_PERMITTIVITY_PAD_ERROR,
            rst.RESIDUAL_NORM_ERROR,
            rst.EXECUTION_TIME,
        ],
        contrast_mode=exp.FIXED_CONTRAST,
        object_size_mode=exp.FIXED_SIZE,
        density_mode=exp.SINGLE_OBJECT,
        allow_rotation=False,
        random_position=False,
    )

    testset.test = samples
    return testset


def build_testsets(repetitions=N_REPETITIONS):
    testsets = []

    # Phase 1: vary only the object shape.
    for shape in SHAPES:
        params = BASE_PARAMS.copy()
        params["shape"] = shape
        name = f"shape_{shape}"
        testsets.append(make_testset(name, params, repetitions))

    # Phase 2: vary only the number of sources.
    for number_sources in SOURCE_VALUES:
        params = BASE_PARAMS.copy()
        params["number_sources"] = number_sources
        name = f"sources_{number_sources}"
        testsets.append(make_testset(name, params, repetitions))

    # Phase 3: vary only the number of measurements.
    for number_measurements in MEASUREMENT_VALUES:
        params = BASE_PARAMS.copy()
        params["number_measurements"] = number_measurements
        name = f"measurements_{number_measurements}"
        testsets.append(make_testset(name, params, repetitions))

    # Phase 4: vary source/measurement pairs for every shape.
    for shape in SHAPES:
        for number_sources, number_measurements in SOURCE_MEASUREMENT_PAIRS:
            params = BASE_PARAMS.copy()
            params["shape"] = shape
            params["number_sources"] = number_sources
            params["number_measurements"] = number_measurements

            name = (
                f"pair_{shape}_sources_{number_sources}"
                f"_measurements_{number_measurements}"
            )
            testsets.append(make_testset(name, params, repetitions))

    return testsets


def run_benchmark(repetitions=N_REPETITIONS):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Generating API evaluate test sets...")
    testsets = build_testsets(repetitions)

    if ALGORITHM is None:
        raise ValueError("Set ALGORITHM to the inverse algorithm function.")

    evaluate_method = api.EvaluateS(
        algorithm=ALGORITHM,
        forward_kwargs=FORWARD_KWARGS,
    )

    benchmark = bmk.Benchmark(
        name=BENCHMARK_NAME,
        method=[evaluate_method],
        discretization=None,
        testset=testsets,
    )

    print(f"Test sets: {len(testsets)}")
    print(f"Repetitions per test set: {repetitions}")
    print(f"Total evaluations: {len(testsets) * repetitions}")
    print("Running benchmark...")

    benchmark.run(parallelization=None)

    save_path = str(OUTPUT_DIRECTORY) + os.sep
    benchmark.save(file_path=save_path, save_testset=True)

    print(f"Benchmark saved to: {save_path}{BENCHMARK_NAME}")
    return benchmark


if __name__ == "__main__":
    run_benchmark()