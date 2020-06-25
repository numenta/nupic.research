# Examples for using Ax

To use Ax flexibly, at scale, for our use cases, it is effective to use:

1. An alternate AxClient (the `CoreAxClient`)
2. A strategy for wrapping an AxClient in a Ray actor (the `AxService`)
3. An alternate Ray AxSearch class (the `NonblockingAxSearch`) that knows how to talk to an AxClient actor.
4. A machine with plenty of memory and perhaps a GPU. (For example, an EC2 p2.xlarge instance.)

Details and rationale below.

## The CoreAxClient

The ax.core API isn't really designed around using Ax as a service. So, to handle use cases like ours, the ax.service API is a wrapper of the core API, and its main class is the `AxClient`.

With its current design, the AxClient is limiting. It is designed for multiple goals:

1. Provide a service-like interface to Ax
2. Eliminate the need for boilerplate code
3. Provide lots of other conveniences

The problem with Goal 2 is that it reduces flexibility. You can only run the scenarios that they have chosen to support in the AxClient. For example, the class currently assumes you have a single objective. It doesn't give you a way to say "use the MultiObjective". Goal 3 makes the class even more rigid / inflexible; for example, the AxClient has a set of charting methods that only make sense when there is a single objective.

We want an AxClient that focuses on Goal 1, without compromise. So we have a "CoreAxClient", a minimal AxClient that requires you to use boilerplate Core API code.

- Multi-objective example: [ax_multi_objective_example.py](ax_multi_objective_example.py)


## Using Ax with Ray

Ray Tune supports using Ax via `tune.run(search_alg=AxSearch)`, but this approach becomes unwieldy with large models. It involves calling Ax's APIs in the same thread that is running `tune.run` and monitoring the workers. These Ax API calls can take multiple minutes. During this time, all of the Ray workers will keep logging, and the logging will get backed up, freezing the workers' progress. Eventually, the majority of time is spent with the head node running Ax and the worker nodes frozen, waiting for the head note to tell them whether they should continue.

Fortunately, Ray Tune uses a polling loop on the SearchAlgorithm. This means we can run Ax in a separate process (a Ray actor) and use an alternate AxSearch to poll it for results.

To create an Ax actor, you will create a subclass of AxClient or CoreAxClient and pass it into our `AxService` class. (This is similar in spirit to how you create a Trainable subclass to use Ray Tune; it's a common theme with the Actor model.) This subclass's job is primarily to be a constructor, customizing your AxClient. When you subclass the CoreAxClient, your `__init__` method will consist mostly of boilerplate Core API that you'll tweak to suit your purposes.

The AxService creates two actors:

- a frontend, whose job is to queue parameters and results, while always responding quickly
- a backend, whose job is to instantiate and call your AxClient

The backend is often busy for long periods of time, while the frontend is quickly responding to GETs and PUTs while sending notifications to the backend.

Rather than using the `AxSearch` search algorithm, use our alternate `NonblockingAxSearch` search algorithm class. It talks to the AxService's frontend.

- Ray example: [ray_ax_example.py](ray_ax_example.py)
