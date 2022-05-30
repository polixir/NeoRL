# Results of NeoRL benchmark
Here we share the raw results of the benchmark for the community to analyze with their own interest. The raw results are stored in `raw_results.json`, with the structure as follow:
```
{
    "task_name": { // e.g. HalfCheetah-v3-low-100
        "algo_name" : [ // e.g. bc
            {
                "parameter" : {
                    "parameter_name" : "parameter_value",
                },
                "results": {
                    "algo_seed" : { // e.g. 7, 42, 210
                        "online" :  "accumulate reward",
                        "ope_method" : { // e.g. fqe, is
                            "ope_seed" : "ope_result", // e.g. 7, 42, 210
                        },
                    }
                }
            }
        ]
    }
}
```