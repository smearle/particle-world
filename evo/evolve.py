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
#       self.optimal_generators = False
        self.stale_generators = False
        self.time_until_stale = 10
        self.staleness = 0
        self.curr_itr = 0

    def reset_staleness(self) -> None:
        self.stale_generators = False
        self.staleness = 0

    def run_setup(self, init_batch = None, **kwargs):
        """TODO"""
        self._update_params(**kwargs)
        if init_batch == None:
            if not hasattr(self, "init_batch") or self.init_batch == None:
                self.gen_init_batch(**kwargs)
        else:
            self.init_batch = init_batch
        # The co-learning loop will always start here, with at least a single round of world-generation
        idx_counter = ray.get_actor("idx_counter")

#       if self.logbook is None:
#           self.logbook = deap.tools.Logbook()

        if self.start_time == None:
            self.start_time = timer()

        self.trainer.workers.local_worker().set_policies_to_train([])

        if len(self.init_batch) == 0:
            raise ValueError("``init_batch`` must not be empty.")

        assert self.mutpb == 1.0

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]

    def reevaluate_elites(self) -> dict:
        """Assuming our evaluation function has changed, re-evaluate some elites in the archive. "The reckoning."
        
        This may apply if, e.g., our evaluation function depends on a co-learning player policy. We first remove elites
        from the archive, then re-evaluate them, updating their objective and measure values, then attempt to re-insert
        them in the archive. When doing QD, this may result in some "collisions," which see the archive left less 
        populated than before.
        """
        # Pop individuals from the container for re-evaluation.
        # Randomly select individuals to pop, without replacement
        invalid_inds = random.sample(self.container, k=min(self.cfg.world_batch_size, len(self.container)))
        # invalid_ind = [ind for ind in container]
        # print(f"{len(invalid_inds)} up for re-evaluation.")

        # TODO: we have an n-time lookup in discard. We should be getting the indices of the individuals directly,
        #  then discarding by index using ``discard_by_index``.
        [self.container.discard(ind) for ind in invalid_inds]
        # container.clear_all()

        return self.evolve(batch=invalid_inds)

        # NOTE: Here we are assuming that we've only evaluated each world once. If we have duplicate stats for a given world, we will overwrite all but one instance of statistics 
        #  resulting from playthrough in this world.

    def generate_offspring(self) -> dict:
        # On the first iteration, randomly generate the initial batch if necessary.
        if self.curr_itr == 0:
            offspring = self.init_batch 

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

        offspring = {i: ind for i, ind in enumerate(offspring)}
        # print(f"{len(batch)} new offspring generated.")

        return offspring

    def evolve(self, batch=None) -> dict:
        """One step of a simple QD algorithm using DEAP, modified to evaluate generated worlds inside an RLlib trainer.

        Args:
            batch (Iterable): Sequence of individuals used as the current batch.
        """
        start_time = timer()

        # If the batch has been supplied externally, use it.
        if batch is not None:
            pass

        # Otherwise, generate a new batch by applying genetic operators to a subset of the individuals in the archive.
        else:
            batch = self.generate_offspring()

        rllib_stats, world_stats, logbook_stats = evaluate_worlds(
            trainer=self.trainer, worlds=batch, cfg=self.cfg, 
            idx_counter=self.idx_counter,
            start_time=self.start_time)
        # assert len(rllib_stats) == 1

        self.tell(batch, world_stats)

    def tell(self, batch, world_stats):

        update_individuals(batch, world_stats)

        if len(batch) == 0:
            raise ValueError("No valid individual found !")

        # Update halloffame
        if self.halloffame is not None:
            self.halloffame.update(self.init_batch)

        # Store batch in container
        n_updated = self.container.update(batch, issue_warning=self.show_warnings)

        self.staleness += n_updated == 0
        self.stale_generators = self.staleness > self.time_until_stale

    #   if cfg.objective_function == "min_solvable":
    #       max_possible_generator_fitness = n_sim_steps - 1 / n_pop
    #       self.optimal_generators = logbook.select("avg")[-1] >= max_possible_generator_fitness - 1e-3

        # if nb_updated == 0:
        #     raise ValueError("No individual could be added to the container !")

        # Compile stats and update logs
        logbook_stats = {}
        logbook_stats.update(self.stats.compile(self.container) if self.stats else {})
        logbook_stats.update({
            'containerSize': self.container.size_str(), 'evals': len(batch), 
            'nbUpdated': n_updated, 'elapsed': timer() - start_time})
        # self.logbook.record(**logbook_stats)
        self.curr_itr += 1

        return logbook_stats
