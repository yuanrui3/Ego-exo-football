import math
import unittest
from dataclasses import replace

from src.goalkeeper_counterfactual import (
    Ball, Config, Draws, Outcome, Player, Sample, State, compare_actions,
    evaluate, sample_draws, simulate,
)


class GoalkeeperCounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.state = State(
            Ball((25.0, 0.0), (-5.0, 0.0)),
            Player((20.0, 5.0), 4.0),
            Player((5.0, 0.0), 5.0),
        )
        self.config = Config(dt=0.1, ball_velocity_noise=0.5,
                             attacker_speed_noise=0.2)

    def test_identical_actions_and_order_share_the_same_draws(self):
        first = compare_actions(self.state, self.config, 17,
                                ("forward", "hold", "forward", "retreat"))
        reversed_branches = compare_actions(self.state, self.config, 17,
                                            ("retreat", "forward", "hold"))
        self.assertEqual(first[0].trajectory, first[2].trajectory)
        self.assertEqual(first[0].outcome, first[2].outcome)
        by_action = {b.action: b for b in first}
        self.assertEqual(by_action, {b.action: b for b in reversed_branches})
        draws = sample_draws(self.state, self.config, 17)
        for branch in first:
            self.assertEqual(branch.trajectory, simulate(self.state, self.config,
                                                         branch.action, draws))
            self.assertEqual(branch.outcome, evaluate(branch.trajectory, self.config))
        self.assertNotEqual(draws, sample_draws(self.state, self.config, 18))

    def test_controlled_step_duration_and_common_continuation(self):
        config = replace(self.config, dt=0.13, first_interval=0.5,
                         ball_velocity_noise=0.0, attacker_speed_noise=0.0)
        for action, expected in (("hold", 5.0), ("forward", 6.0), ("retreat", 4.0)):
            path = simulate(self.state, config, action)
            sample = next(s for s in path if s.time == config.first_interval)
            self.assertAlmostEqual(sample.keeper[0], expected)
            self.assertAlmostEqual(sample.keeper[1], 0)
            after = next(s for s in path if s.time > config.first_interval)
            # The same ball-chase feedback applies in every branch.
            self.assertGreater(after.keeper[0], sample.keeper[0])
            self.assertLessEqual(after.keeper[0] - sample.keeper[0],
                                 self.state.keeper.speed * (after.time - sample.time) + 1e-12)
        long_step = replace(config, first_interval=2.0)
        path = simulate(self.state, long_step, "forward")
        self.assertAlmostEqual(next(s for s in path if s.time == 2).keeper[0], 6)

    def test_illegal_actions_and_infeasible_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate(self.state, self.config, "jump")
        edge = replace(self.state, keeper=Player((0.2, 0), 5))
        with self.assertRaises(ValueError):
            simulate(edge, self.config, "retreat")
        self.assertEqual(simulate(edge, self.config, "hold")[0].keeper, (0.2, 0))
        exact_edge = replace(self.state, keeper=Player((1, 0), 5))
        boundary_path = simulate(exact_edge, self.config, "retreat")
        self.assertEqual(next(s for s in boundary_path if s.time == 0.5).keeper[0], 0)
        evaluate(boundary_path, self.config)
        for state in (
            replace(self.state, keeper=Player((-0.1, 0), 5)),
            replace(self.state, attacker=Player((0, 0), 12.1)),
            replace(self.state, ball=Ball((25, 0), (51, 0))),
            replace(self.state, ball=Ball((25, math.nan), (-5, 0))),
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                simulate(state, self.config, "hold")
        for config in (replace(self.config, dt=1e-15),
                       replace(self.config, dt=0),
                       replace(self.config, first_interval=5),
                       replace(self.config, contact_radius=-1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                simulate(self.state, config, "hold")
        with self.assertRaises(ValueError):
            simulate(self.state, self.config, "hold", Draws((0, 0), -5))
        with self.assertRaises(ValueError):
            simulate(self.state, self.config, "hold", Draws((-50, 0), 0))

    def test_evaluator_first_events_and_legal_box(self):
        config = Config(contact_radius=0.5)
        # A keeper contact outside the penalty area is ignored; the first
        # attacker touch occurs before the ball can enter the area.
        path = (Sample(0, (20, 0), (19, 0), (20, 0)),
                Sample(2, (18, 0), (19, 0), (20, 0)))
        self.assertEqual(evaluate(path, config), Outcome("attacker_touch", 0.5))
        # Contact becomes legal exactly at the penalty-area boundary.
        path = (Sample(0, (17, 0), (30, 0), (17, 0)),
                Sample(1, (16, 0), (30, 0), (16, 0)))
        self.assertEqual(evaluate(path, config), Outcome("keeper_interception", 0.5))
        goal = (Sample(0, (1, 0), (10, 10), (10, -10)),
                Sample(1, (0, 0), (10, 10), (10, -10)))
        self.assertEqual(evaluate(goal, config), Outcome("goal", 1.0))
        outside = tuple(replace(s, ball=(s.ball[0], 4)) for s in goal)
        self.assertEqual(evaluate(outside, config), Outcome("unresolved", 1.0))
        short = (Sample(0, (20, 0), (30, 0), (10, 0)),)
        self.assertEqual(evaluate(short, config), Outcome("unresolved", 0))

    def test_symmetric_contacts_and_contested_ties(self):
        config = Config(contact_radius=1)
        keeper_first = (Sample(0, (10, 0), (13, 0), (7, 0)),
                        Sample(2, (10, 0), (12, 0), (9, 0)))
        self.assertEqual(evaluate(keeper_first, config), Outcome("keeper_interception", 2))
        attacker_first = tuple(replace(s, attacker=s.keeper, keeper=s.attacker)
                               for s in keeper_first)
        self.assertEqual(evaluate(attacker_first, config), Outcome("attacker_touch", 2))
        tied = (Sample(0, (10, 0), (12, 0), (8, 0)),
                Sample(2, (10, 0), (10, 0), (10, 0)))
        self.assertEqual(evaluate(tied, config), Outcome("contested", 1))
        at_goal = (Sample(0, (1, 0), (0, 0), (0, 10)),
                   Sample(1, (0, 0), (0, 0), (0, 10)))
        self.assertEqual(evaluate(at_goal, replace(config, contact_radius=0)),
                         Outcome("attacker_touch", 1))

    def test_tangent_contact_survives_floating_point_cancellation(self):
        config = Config(contact_radius=0.6)
        path = (Sample(0, (0, 0), (0.1, 0.6), (10, 0)),
                Sample(1, (1, 0), (0.1, 0.6), (10, 0)))
        result = evaluate(path, config)
        self.assertEqual(result.kind, "attacker_touch")
        self.assertAlmostEqual(result.time, 0.1)
        near_miss = tuple(replace(s, attacker=(0.1, 0.600000001)) for s in path)
        self.assertEqual(evaluate(near_miss, config), Outcome("unresolved", 1))

    def test_timestep_convergence_and_shared_scene_noise(self):
        trajectories = []
        outcomes = []
        for dt in (0.1, 0.05, 0.025, 0.0125):
            config = replace(self.config, dt=dt, attacker_avoidance=1.5)
            branch, = compare_actions(self.state, config, 42, ("forward",))
            trajectories.append(branch.trajectory[-1].attacker)
            outcomes.append(branch.outcome)
            self.assertEqual(sample_draws(self.state, config, 42),
                             sample_draws(self.state, self.config, 42))
        def distance(a, b):
            return math.dist(a, b)
        self.assertLess(distance(trajectories[2], trajectories[3]),
                        distance(trajectories[1], trajectories[3]))
        self.assertLess(distance(trajectories[1], trajectories[3]),
                        distance(trajectories[0], trajectories[3]))
        self.assertEqual(len({o.kind for o in outcomes}), 1)
        self.assertLess(abs(outcomes[2].time - outcomes[3].time), 0.02)

    def test_attacker_responsiveness_is_optional_and_action_dependent(self):
        passive = replace(self.config, attacker_avoidance=0)
        forward = simulate(self.state, passive, "forward")
        retreat = simulate(self.state, passive, "retreat")
        self.assertEqual([s.attacker for s in forward], [s.attacker for s in retreat])
        responsive = replace(passive, attacker_avoidance=3.0)
        forward = simulate(self.state, responsive, "forward")
        retreat = simulate(self.state, responsive, "retreat")
        self.assertGreater(math.dist(forward[-1].attacker, retreat[-1].attacker), 0.001)
        self.assertNotEqual(forward[-1].attacker,
                            simulate(self.state, passive, "forward")[-1].attacker)


if __name__ == "__main__":
    unittest.main()
