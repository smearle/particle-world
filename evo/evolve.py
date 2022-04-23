import os
from pdb import set_trace as TT
import random
from typing import Optional
from qdpy.algorithms.deap import DEAPQDAlgorithm
import ray

import deap
from timeit import default_timer as timer

from utils import update_world_individuals


class Evolver(DEAPQDAlgorithm):
    def __init__(self, trainer, idx_counter, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FIXME: This is circular, lose this reference!
        self.trainer = trainer

        self.cfg = cfg
        self.idx_counter = idx_counter
        self.logbook = None
        self.show_warnings = True
#       self.optimal_generators = False
        self.stale_generators = False
        self.time_until_stale = 10
        self.staleness = 0
        self.curr_itr = trainer.gen_itr
        self.stale_individuals = []

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
#       idx_counter = ray.get_actor("idx_counter")

    def tell(self, batch, world_stats):

        # TODO: Refer to a dict mapping fitness functions to their minimum values?
        self._update_individuals(batch, world_stats, min_fitness=-10)

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]

    def generate_batch(self, batch_size: int) -> list:
        """Generate a batch of individuals to evaluate.

        Where possible, we return an even mix of new mutants and "stale" elites.

        Args:
            batch_size (int): Number of individuals to generate.
        """
        contested_inds = self._disturb_elites(batch_size // 2)
        # print(f"{len(contested_inds)} contested individuals.")
        offspring = self._generate_offspring(batch_size - len(contested_inds))
        [self.container.discard(ind) for ind in contested_inds]
        batch = contested_inds + offspring
        batch = {i: ind for i, ind in enumerate(batch)}

        return batch

    def _disturb_elites(self, batch_size: int) -> list:
        """Collect maximally stale individuals from the archive so that they may be re-evaluated.
        
        Note that we may return a list of individuals smaller than the requested batch size.

        Args:
            batch_size (int): Number of individuals to return (ideally).
        """
        invalid_inds = []
        i = 0
        while len(invalid_inds) < batch_size and i < len(self.stale_individuals):
            ind = self.stale_individuals[i]
            if ind in self.container:
                invalid_inds.append(ind)
                # self.container.discard(ind)
            i += 1
        self.stale_individuals = self.stale_individuals[i:]

#       # FIXME: Having to check the container again implies we have duplicates in invalid_inds, in stale_individuals, 
#       # and thus in the container... How?
#       [self.container.discard(ind) for ind in invalid_inds if ind in self.container]

        # Randomly select individuals to pop, without replacement
        # invalid_inds = random.sample(self.stale_individuals, k=min(batch_size, len(self.stale_individuals)))
        # invalid_ind = [ind for ind in container]
        # print(f"{len(invalid_inds)} up for re-evaluation.")

        # TODO: we have an n-time lookup in discard. We should be getting the indices of the individuals directly,
        #  then discarding by index using ``discard_by_index``.
        # container.clear_all()
        self.reset_ages(invalid_inds)

        return invalid_inds

    def reevaluate_elites(self) -> dict:
        """Assuming our evaluation function has changed, re-evaluate some elites in the archive. "The reckoning."
        
        This may apply if, e.g., our evaluation function depends on a co-learning player policy. We first remove elites
        from the archive, then re-evaluate them, updating their objective and measure values, then attempt to re-insert
        them in the archive. When doing QD, this may result in some "collisions," which see the archive left less 
        populated than before.
        """
        invalid_inds = {i: ind for i, ind in enumerate(self._disturb_elites())}

        return {i: ind for i, ind in enumerate(invalid_inds)}

    def reevaluate_elites(self) -> dict:
        """Assuming our evaluation function has changed, re-evaluate some elites in the archive. "The reckoning."
        
        This may apply if, e.g., our evaluation function depends on a co-learning player policy. We first remove elites
        from the archive, then re-evaluate them, updating their objective and measure values, then attempt to re-insert
        them in the archive. When doing QD, this may result in some "collisions," which see the archive left less 
        populated than before.
        """
        invalid_inds = self.disturb_elites()

        return self.evolve(batch=invalid_inds)

        # NOTE: Here we are assuming that we've only evaluated each world once. If we have duplicate stats for a given world, we will overwrite all but one instance of statistics 
        #  resulting from playthrough in this world.

    def _generate_offspring(self, batch_size: Optional[int] = None) -> dict:
        batch_size = self.batch_size if batch_size is None else batch_size
        # On the first iteration, randomly generate the initial batch if necessary.
        if len(self.container) == 0:
            assert len(self.init_batch) >= batch_size
            offspring = self.init_batch[:batch_size]

        # Otherwise, generate a new batch by applying genetic operators to a subset of the individuals in the archive.
        else:
            batch = self.toolbox.select(self.container, batch_size)

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

        # offspring = {i: ind for i, ind in enumerate(offspring)}
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
            batch = {i: ind for i, ind in enumerate(self._generate_offspring())}

        # FIXME: do we want to revive this using the WorldEvoTrainer?
        # rllib_stats, world_stats, logbook_stats = evaluate_worlds(
        #     trainer=self.trainer, worlds=batch, cfg=self.cfg, 
        #     idx_counter=self.idx_counter,
        #     start_time=self.start_time)
        # # assert len(rllib_stats) == 1
        world_stats = {}

        self.tell(batch, world_stats)

    def tell(self, batch, world_stats):

        self._update_individuals(batch, world_stats)

        if len(batch) == 0:
            raise ValueError("No valid individual found !")

        # Update halloffame
        if self.halloffame is not None:
            self.halloffame.update(self.init_batch)

        # Store batch in container
        n_updated = self.container.update([ind for ind in batch.values()], issue_warning=self.show_warnings)

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
            'nbUpdated': n_updated, \
                # 'elapsed': timer() - start_time
            })
        # self.logbook.record(**logbook_stats)
        self.curr_itr += 1

        return logbook_stats

    def increment_ages(self):
        self.stale_individuals = []
        for ind in self.container:
            ind.fitness.age += 1
            # Note that `stale_individuals` is ordered debcreasing by age.
            self.stale_individuals.append(ind)
            # FIXME: we can do better than O(n * log n) here. Why can't we hash individuals? Find a way.
            self.stale_individuals = sorted(self.stale_individuals, key=lambda ind: ind.fitness.age, reverse=True)

    def reset_ages(self, individuals):
        for ind in individuals:
            ind.fitness.age = 0

    def _update_individuals(self, batch, world_stats):
        raise NotImplementedError

class WorldEvolver(Evolver):
    def _update_individuals(individuals, qd_stats, min_fitness=-10):
        """Update worlds with stats resulting from simulation.
        
        If worlds are unplayable, then we don't need to have simulated on them, and we penalize them so that they will be
        dominated by any playable world."""
        playable_individuals = {wk: ind for wk, ind in individuals.items() if ind.playability_penalty == 0}
        assert len(playable_individuals) == len(qd_stats), f"Number of individuals, {len(playable_individuals)}, != {len(qd_stats)},"\
            " number of qd_stats."
        
        for k in individuals:
            ind = individuals[k]
            if ind.playability_penalty > 0:
                # This is totally ad hoc, let's just exclude the individual entirely.
                ind.fitness.values = (min_fitness - ind.playability_penalty, )
                ind.features = [0, 0]
                continue
            assert k in qd_stats, f"Individual {k} not in qd_stats."
            ind.fitness.values = qd_stats[k][0]
            ind.features = qd_stats[k][1]


class PlayerEvolver(Evolver):
    def tell(self, batch, world_stats):
        logbook_stats = super().tell(batch, world_stats)
        logbook_stats = {f"play_{k}": v for k, v in logbook_stats.items()}
        return logbook_stats

    def _update_individuals(individuals, objs):
        for k, ind in individuals.items():
            ind.fitness.values = (objs[k], )
            ind.features = [0, 0]



