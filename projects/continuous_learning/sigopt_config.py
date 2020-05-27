sigopt_config = dict(
    name="GSC_duty_cycle_freezing",
    project="continuous_learning",
    observation_budget=200,
    parallel_bandwidth=1,
    parameters=[
        dict(
            name="cnn1_size",
            type="int",
            bounds=dict(min=64, max=256)
        ),
        dict(
            name="cnn2_size",
            type="int",
            bounds=dict(min=64, max=256)
        ),
        dict(
            name="cnn1_percent_on",
            type="double",
            bounds=dict(min=0.02, max=0.8)
        ),
        dict(
            name="cnn1_wt_sparsity",
            type="double",
            bounds=dict(min=0.05, max=0.8)
        ),
        dict(
            name="cnn2_percent_on",
            type="double",
            bounds=dict(min=0.02, max=0.8)
        ),
        dict(
            name="cnn2_wt_sparsity",
            type="double",
            bounds=dict(min=0.05, max=0.8)
        ),
        dict(
            name="dendrites_per_neuron",
            type="int",
            bounds=dict(min=1, max=6)
        ),
        dict(
            name="learning_rate",
            type="double",
            bounds=dict(min=0.001, max=0.2)
        ),
        dict(
            name="learning_rate_factor",
            type="double",
            bounds=dict(min=0, max=1)
        ),
        dict(
            name="use_batch_norm",
            type="categorical",
            categorical_values=["True", "False"]
        ),
        dict(
            name="log2_batch_size",
            type="int",
            bounds=dict(min=3, max=7)
        ),
        dict(
            name="linear1_n",
            type="int",
            bounds=dict(min=100,max=2500)
        ),
        dict(
            name="linear1_percent_on",
            type="double",
            bounds=dict(min=0.02, max=0.8)
        ),
        dict(
            name="linear1_weight_sparsity",
            type="double",
            bounds=dict(min=0.01, max=0.8)
        ),
        dict(
            name="linear2_percent_on",
            type="double",
            bounds=dict(min=0.02, max=0.8)
        ),
        dict(
            name="linear2_weight_sparsity",
            type="double",
            bounds=dict(min=0.01, max=0.8)
        ),
        dict(
            name="duty_cycle_period",
            type="int",
            bounds=dict(min=100, max=15000)
        ),
        dict(
            name="boost_strength",
            type="double",
            bounds=dict(min=0.0, max=2.0)
        ),
        dict(
            name="boost_strength_factor",
            type="double",
            bounds=dict(min=0.0,max=1.0)
        ),
    ],
    metrics=[
        dict(
            name="area_under_curve",
            objective="maximize"
        )
    ]
)
