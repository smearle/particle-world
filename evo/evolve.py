import os
from pdb import set_trace as TT
import random
from qdpy.algorithms.deap import DEAPQDAlgorithm
import ray

import deap
from timeit import default_timer as timer

from rl.eval_worlds import evaluate_worlds
from utils import update_individuals


class WorldEvolver(DEAPQDAlgorithm):
    def __init__(self, trainer, idx_counter, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.cfg = cfg
        self.idx_counter = idx_counter
        self.logbook = None
        self.show_warnings = True
        self.gen_itr = 0

    def run_setup(self, init_batch = None, **kwargs):
        """TODO"""
        self._update_params(**kwargs)
        # If needed, generate the initial batch
        if init_batch == None:
            if not hasattr(self, "init_batch") or self.init_batch == None:
                self.gen_init_batch(**kwargs)
        else:
            self.init_batch = init_batch
        # The co-learning loop will always start here, with at least a single round of world-generation
        idx_counter = ray.get_actor("idx_counter")

        if self.logbook is None:
            # assert net_itr == gen_itr == 0
            self.logbook = deap.tools.Logbook()

        if self.start_time == None:
            self.start_time = timer()

        self.trainer.workers.local_worker().set_policies_to_train([])
        # TODO: use "chapters" to hierarchicalize generator fitness, agent reward, and path length stats?
        self.logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (self.stats.fields if self.stats else []) \
            + ["meanRew", "meanEvalRew", "meanPath", "maxPath"] + ["elapsed"]

        if len(self.init_batch) == 0:
            raise ValueError("``init_batch`` must not be empty.")

        assert self.mutpb == 1.0

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]

    def evolve(self, net_itr, gen_itr, play_itr):
        """Simple QD algorithm using DEAP, modified to evaluate generated worlds inside an RLlib trainer object.

        Args:
            rllib_trainer: RLlib trainer object.
            rllib_eval: #TODO
            param quality_diversity: If this is False, we are reducing this to a vanilla evolutionary strategy.
            init_batch: Sequence of individuals used as initial batch.
            toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
            batch_size: The number of individuals in a batch.
        :param niter: The number of iterations.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                        contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :param show_warnings: Whether or not to show warnings and errors. Useful to check if some individuals were out-of-bounds.
        :param start_time: Starting time of the illumination process, or None to take the current time.
        :param iteration_callback: Optional callback funtion called when a new batch is generated. The callback function parameters are (iteration, batch, container, logbook).
        :param gen_itr: The current generator iteration if reloading.
        :param net_itr: The current net iteration if reloading (sum of generator and player iterations).
        :returns: The final batch
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                evolution
        """
        start_time = timer()

        if self.gen_itr == 0:
            batch = self.init_batch 

        else:
            batch = self.toolbox.select(self.container, self.batch_size)

            ## Vary the pool of individuals
            # offspring = deap.algorithms.varAnd(batch, toolbox, cxpb, mutpb)
            # offspring = [toolbox.clone(ind) for ind in batch]
            offspring = [None for ind in batch]
            for i in range(len(offspring)):
                if random.random() < 0.00:
                    # Create a new random individual.
                    offspring[i] = self.toolbox.individual()
                    del offspring[i].fitness.values
                else:
                    # Create a mutated copy of an existing individual.
                    offspring[i] = self.toolbox.clone(batch[i])
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            batch = [ind for ind in offspring if not ind.fitness.valid]

        print(f"{len(batch)} new offspring up for evaluation.")
        rllib_stats, world_stats, logbook_stats = evaluate_worlds(
            trainer=self.trainer, worlds={i: ind for i, ind in enumerate(batch)}, cfg=self.cfg, 
            idx_counter=self.idx_counter,
            net_itr=net_itr, start_time=self.start_time)
        # assert len(rllib_stats) == 1

        update_individuals(batch, world_stats)

        if len(batch) == 0:
            raise ValueError("No valid individual found !")

        # Update halloffame
        if self.halloffame is not None:
            self.halloffame.update(self.init_batch)

        # Store batch in container
        n_updated = self.container.update(batch, issue_warning=self.show_warnings)
        # if nb_updated == 0:
        #     raise ValueError("No individual could be added to the container !")

        # Compile stats and update logs
        record = self.stats.compile(self.container) if self.stats else {}
        logbook_stats.update({
            'iteration': net_itr, 'containerSize': self.container.size_str(), 'evals': len(batch), 
            'nbUpdated': n_updated, 'elapsed': timer() - start_time, **record,
        })
        self.logbook.record(**logbook_stats)
        if self.verbose:
            print(self.logbook.stream)
        net_itr += 1
        # Call callback function
        old_net_itr = net_itr
        net_itr = [net_itr]
        play_itr = [play_itr]
        if self.iteration_callback_fn != None:
            self.iteration_callback_fn(net_itr=net_itr, iteration=gen_itr, play_itr=play_itr, batch=self.init_batch, 
                container=self.container, logbook=self.logbook, stats=self.stats)
        new_net_itr = net_itr[0]
        play_itr = play_itr[0] + (new_net_itr - old_net_itr)
        net_itr = new_net_itr
        self.gen_itr = gen_itr = gen_itr + 1

        # TODO: move this to the outer loop
        done = play_itr >= self.niter

        return done
